import os
import glob
import random
import xml.etree.ElementTree as ET
import numpy as np
import openslide
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib.path import Path

PATCH_SIZE = 28
LEVEL = 3
BAG_SIZE = 100
BAGS_PER_SLIDE = 10
TISSUE_THRESHOLD = 200


def _is_tissue(patch):
    return np.array(patch).mean() < TISSUE_THRESHOLD


class CameleyonBagDataset(Dataset):
    def __init__(self, normal_dir, tumor_dir, annot_dir, bag_size=BAG_SIZE, bags_per_slide=BAGS_PER_SLIDE,
                 transform=None):
        self.normal_files = glob.glob(os.path.join(normal_dir, '*.tif'))
        self.tumor_files = glob.glob(os.path.join(tumor_dir, '*.tif'))
        self.annot_dir = annot_dir
        self.bag_size = bag_size
        self.bags_per_slide = bags_per_slide

        self.tumor_ratio = 0.1
        self.num_tumor_patches = int(self.bag_size * self.tumor_ratio)
        self.num_normal_patches = self.bag_size - self.num_tumor_patches

        self.slide_paths = []
        self.labels = []

        for p in self.normal_files:
            self.slide_paths.append(p)
            self.labels.append(0)

        for p in self.tumor_files:
            self.slide_paths.append(p)
            self.labels.append(1)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.slide_paths) * self.bags_per_slide

    def __getitem__(self, idx):
        slide_idx = idx // self.bags_per_slide
        current_bag_num = idx % self.bags_per_slide

        slide_path = self.slide_paths[slide_idx]
        label = self.labels[slide_idx]

        wsi = openslide.OpenSlide(slide_path)
        level = LEVEL if LEVEL < wsi.level_count else wsi.level_count - 1

        patches = []

        if label == 0:
            patches = self._sample_tissue_random(wsi, level, self.bag_size, polygons_to_avoid=[])

        else:
            xml_path = self._get_xml_path(slide_path)
            tumor_polygons = []

            if xml_path and os.path.exists(xml_path):
                tumor_polygons = self._parse_xml_polygons(xml_path)

            if tumor_polygons:
                patches_tumor = self._sample_tumor_grid(wsi, level, tumor_polygons, self.num_tumor_patches,
                                                        current_bag_num)

                needed_normal = self.bag_size - len(patches_tumor)
                patches_normal = self._sample_tissue_random(wsi, level, needed_normal, polygons_to_avoid=tumor_polygons)

                patches = patches_tumor + patches_normal

            else:
                # patches = self._sample_tissue_random(wsi, level, self.bag_size, polygons_to_avoid=[])
                raise Exception("Tumor XML annotations file not found.")

        # PADDING ZERAMI
        if len(patches) < self.bag_size:
            diff = self.bag_size - len(patches)
            zero_tensor = torch.zeros(1, PATCH_SIZE, PATCH_SIZE)
            patches += [zero_tensor] * diff

        if len(patches) == 0:
            # patches = [torch.zeros(1, PATCH_SIZE, PATCH_SIZE)] * self.bag_size
            raise Exception("No patches generated")

        random.shuffle(patches)

        bag_tensor = torch.stack(patches)
        wsi.close()

        return bag_tensor, torch.tensor([label], dtype=torch.float32)

    def _sample_tissue_random(self, wsi, level, count, polygons_to_avoid=None):
        patches = []
        w_lvl, h_lvl = wsi.level_dimensions[level]
        downsample = wsi.level_downsamples[level]

        attempts = 0
        max_attempts = count * 50

        while len(patches) < count and attempts < max_attempts:
            attempts += 1

            x_lvl = random.randint(0, w_lvl - PATCH_SIZE)
            y_lvl = random.randint(0, h_lvl - PATCH_SIZE)

            x_0 = int(x_lvl * downsample)
            y_0 = int(y_lvl * downsample)

            in_tumor = False
            if polygons_to_avoid:
                center_point = (x_0 + (PATCH_SIZE * downsample) / 2, y_0 + (PATCH_SIZE * downsample) / 2)
                for poly in polygons_to_avoid:
                    if poly.contains_point(center_point):
                        in_tumor = True
                        break

            if in_tumor:
                continue

            try:
                patch = wsi.read_region((x_0, y_0), level, (PATCH_SIZE, PATCH_SIZE))
                patch = patch.convert('L')

                if _is_tissue(patch):
                    patches.append(self.transform(patch))
            except:
                pass

        return patches

    def _sample_tumor_grid(self, wsi, level, polygons, count, bag_num):
        patches = []
        if not polygons:
            return []

        downsample = wsi.level_downsamples[level]
        stride_0 = int(PATCH_SIZE * downsample)  # Bez overlapu

        all_candidates = []
        for poly in polygons:
            (minx, miny, maxx, maxy) = poly.get_extents().bounds
            xs = range(int(minx), int(maxx), stride_0)
            ys = range(int(miny), int(maxy), stride_0)

            for x in xs:
                for y in ys:
                    center_point = (x + stride_0 / 2, y + stride_0 / 2)
                    if poly.contains_point(center_point):
                        all_candidates.append((x, y))

        if not all_candidates:
            return []

        # Determinizm gridu
        random.seed(42)
        random.shuffle(all_candidates)
        random.seed()

        # Wybieramy unikalne dla danego worka (paginacja)
        start_idx = (bag_num * count) % len(all_candidates)

        # Bierzemy kandydatów
        # UWAGA: Tutaj nie zapętlamy na siłę. Jeśli braknie unikalnych,
        # metoda zwróci mniej niż 'count', a resztę dociągnie funkcja _sample_tissue_random w __getitem__
        selected_candidates = all_candidates[start_idx: start_idx + count]

        for (x_0, y_0) in selected_candidates:
            try:
                patch = wsi.read_region((int(x_0), int(y_0)), level, (PATCH_SIZE, PATCH_SIZE))
                patch = patch.convert('L')

                if np.array(patch).mean() < TISSUE_THRESHOLD:
                    patches.append(self.transform(patch))
            except:
                continue

        return patches

    def _get_xml_path(self, slide_path):
        filename = os.path.basename(slide_path)
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        return os.path.join(self.annot_dir, xml_filename)

    def _parse_xml_polygons(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        polygons = []
        for annotation in root.findall('.//Annotation'):
            coords = []
            for coord in annotation.findall('.//Coordinate'):
                x = float(coord.get('X'))
                y = float(coord.get('Y'))
                coords.append((x, y))
            if len(coords) > 2:
                polygons.append(Path(coords))
        return polygons


class CameleyonTestDataset(Dataset):
    def __init__(self, normal_dir, tumor_dir, bag_size=100, bags_per_slide=BAGS_PER_SLIDE, transform=None):
        self.normal_files = glob.glob(os.path.join(normal_dir, '*.tif'))
        self.tumor_files = glob.glob(os.path.join(tumor_dir, '*.tif'))
        self.bag_size = bag_size
        self.bags_per_slide = bags_per_slide

        self.slide_paths = []
        self.labels = []
        for p in self.normal_files:
            self.slide_paths.append(p)
            self.labels.append(0)
        for p in self.tumor_files:
            self.slide_paths.append(p)
            self.labels.append(1)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.slide_paths) * self.bags_per_slide

    def __getitem__(self, idx):
        slide_idx = idx // self.bags_per_slide

        slide_path = self.slide_paths[slide_idx]
        label = self.labels[slide_idx]

        wsi = openslide.OpenSlide(slide_path)
        level = LEVEL if LEVEL < wsi.level_count else wsi.level_count - 1

        patches = self._sample_tissue_patches(wsi, level, self.bag_size)

        # ZMIANA: Padding zerami (Black Tensors) zamiast duplikowania
        if len(patches) < self.bag_size:
            diff = self.bag_size - len(patches)
            # Tworzymy czarny kwadrat [1, H, W]
            zero_tensor = torch.zeros(1, PATCH_SIZE, PATCH_SIZE)
            patches += [zero_tensor] * diff

        if len(patches) == 0:
            # patches = [torch.zeros(1, PATCH_SIZE, PATCH_SIZE)] * self.bag_size
            raise Exception("No patches generated")

        bag_tensor = torch.stack(patches)
        wsi.close()

        return bag_tensor, torch.tensor([label], dtype=torch.float32)

    def _sample_tissue_patches(self, wsi, level, count):
        patches = []
        w_lvl, h_lvl = wsi.level_dimensions[level]
        downsample = wsi.level_downsamples[level]

        attempts = 0
        max_attempts = count * 50

        while len(patches) < count and attempts < max_attempts:
            attempts += 1
            x_lvl = random.randint(0, w_lvl - PATCH_SIZE)
            y_lvl = random.randint(0, h_lvl - PATCH_SIZE)
            x_0 = int(x_lvl * downsample)
            y_0 = int(y_lvl * downsample)

            try:
                patch = wsi.read_region((x_0, y_0), level, (PATCH_SIZE, PATCH_SIZE))
                patch = patch.convert('L')

                if np.array(patch).mean() < TISSUE_THRESHOLD:
                    patches.append(self.transform(patch))
            except:
                continue

        return patches