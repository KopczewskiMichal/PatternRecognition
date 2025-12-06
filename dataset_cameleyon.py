import os
import glob
import random
import xml.etree.ElementTree as ET
import numpy as np
import openslide
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib.path import Path
import math

PATCH_SIZE = 28
LEVEL = 4
BAG_SIZE = 100
BAGS_PER_SLIDE = 5
TISSUE_THRESHOLD = 200


class CameleyonBagDataset(Dataset):
    def __init__(self, normal_dir, tumor_dir, annot_dir, bag_size=BAG_SIZE, bags_per_slide=BAGS_PER_SLIDE,
                 transform=None):
        self.normal_files = glob.glob(os.path.join(normal_dir, '*.tif'))
        self.tumor_files = glob.glob(os.path.join(tumor_dir, '*.tif'))
        self.annot_dir = annot_dir
        self.bag_size = bag_size
        self.bags_per_slide = bags_per_slide  # Ile worków wyciągnąć z jednego pliku

        self.slide_paths = []
        self.labels = []

        for p in self.normal_files:
            self.slide_paths.append(p)
            self.labels.append(0)

        for p in self.tumor_files:
            self.slide_paths.append(p)
            self.labels.append(1)

        self._validate_paths()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        # Dataset jest teraz BAGS_PER_SLIDE razy większy niż liczba plików
        return len(self.slide_paths) * self.bags_per_slide

    def __getitem__(self, idx):
        # Przeliczamy globalny index na konkretny slajd i numer worka
        slide_idx = idx // self.bags_per_slide
        current_bag_num = idx % self.bags_per_slide

        slide_path = self.slide_paths[slide_idx]
        label = self.labels[slide_idx]

        wsi = openslide.OpenSlide(slide_path)
        level = LEVEL if LEVEL < wsi.level_count else wsi.level_count - 1

        patches = []

        if label == 0:
            # Dla NORMAL: Dalej losowo (random).
            # Każde wywołanie wygeneruje inny losowy zestaw (statystycznie).
            patches = self._sample_normal_patches(wsi, level, self.bag_size)
        else:
            # Dla TUMOR: Używamy XML i Gridu, żeby unikać nakładania
            xml_path = self._get_xml_path(slide_path)

            if xml_path and os.path.exists(xml_path):
                tumor_polygons = self._parse_xml_polygons(xml_path)

                # Tu przekazujemy numer worka, żeby wiedzieć, który fragment siatki wziąć
                patches = self._sample_tumor_grid(wsi, level, tumor_polygons, self.bag_size, current_bag_num)
            else:
                patches = self._sample_normal_patches(wsi, level, self.bag_size)

        # Padding (dopełnianie)
        start_len_patches = len(patches)
        print(start_len_patches)
        if start_len_patches < self.bag_size:
            diff = self.bag_size - len(patches)
            if len(patches) > 0:
                patches += [patches[random.choice(range(start_len_patches))]] * diff
                print(f"dopełniono {diff} elementów")
            else:
                patches = [torch.zeros(1, PATCH_SIZE, PATCH_SIZE)] * self.bag_size


        bag_tensor = torch.stack(patches)
        wsi.close()

        return bag_tensor, torch.tensor([label], dtype=torch.float32)

    # --- METODY POMOCNICZE ---

    def _sample_tumor_grid(self, wsi, level, polygons, count, bag_num):
        """
        Generuje wycinki z siatki (grid).
        Gwarantuje brak nakładania się wycinków (non-overlapping) dzięki stałemu krokowi.
        """
        patches = []
        if not polygons:
            return []

        downsample = wsi.level_downsamples[level]
        # Krok siatki na poziomie 0, żeby na poziomie LEVEL wycinki stykały się krawędziami
        # Jeśli chcesz przerwę między wycinkami, zwiększ stride_0
        stride_0 = int(PATCH_SIZE * downsample)

        # Zbieramy wszystkich kandydatów (punkty x,y) z całego guza
        all_candidates = []

        for poly in polygons:
            (minx, miny, maxx, maxy) = poly.get_extents().bounds

            # Generujemy siatkę TYLKO wewnątrz bounding boxa poligonu
            # range z krokiem stride_0 zapewnia brak overlapu
            xs = range(int(minx), int(maxx), stride_0)
            ys = range(int(miny), int(maxy), stride_0)

            for x in xs:
                for y in ys:
                    # Sprawdzamy czy środek patcha jest w poligonie (dokładniej)
                    center_point = (x + stride_0 / 2, y + stride_0 / 2)
                    if poly.contains_point(center_point):
                        all_candidates.append((x, y))

        # Jeśli nie ma kandydatów
        if not all_candidates:
            return []

        # Deterministyczne mieszanie dla powtarzalności LUB zwykłe shuffle
        # Ważne: chcemy żeby kolejność kandydatów była stała dla danego slajdu,
        # żeby bag_num=0 wziął pierwsze 100, bag_num=1 kolejne 100 itd.
        # Dlatego sortujemy, a potem seedujemy RNG indeksem slajdu (opcjonalne)
        # lub po prostu bierzemy wycinek listy.

        # Prostsze podejście: przetasujmy raz z seedem zależnym od nazwy pliku (żeby było stabilnie)
        # albo po prostu bierzmy po kolei, jeśli grid jest naturalnie posortowany.
        # Zróbmy shuffle, żeby brać próbki z różnych miejsc guza, a nie tylko z góry.
        random.seed(42)  # Stały seed dla układu gridu
        random.shuffle(all_candidates)
        random.seed()  # Reset seeda do losowego dla reszty programu

        # Wybieramy odpowiedni fragment listy dla danego worka (Paginacja)
        start_idx = bag_num * count

        # Jeśli guza jest mało i braknie nam unikalnych punktów na 10 worków:
        # Możemy albo zapętlić (modulo), albo brać unikalne ile się da.
        # Tutaj zastosuję modulo - jeśli braknie unikalnych, zaczną się powtarzać z początku listy.

        selected_candidates = []
        for i in range(count):
            idx = (start_idx + i) % len(all_candidates)
            selected_candidates.append(all_candidates[idx])

        # Pobieranie obrazków
        for (x_0, y_0) in selected_candidates:
            try:
                # x_0, y_0 to współrzędne na Level 0
                patch = wsi.read_region((int(x_0), int(y_0)), level, (PATCH_SIZE, PATCH_SIZE))
                patch = patch.convert('RGB')
                patches.append(self.transform(patch))
            except:
                continue

        return patches

    def _sample_normal_patches(self, wsi, level, count):
        """Bez zmian - losowe próbkowanie"""
        patches = []
        w_lvl, h_lvl = wsi.level_dimensions[level]
        downsample = wsi.level_downsamples[level]
        attempts = 0
        max_attempts = count * 30

        while len(patches) < count and attempts < max_attempts:
            attempts += 1
            x_lvl = random.randint(0, w_lvl - PATCH_SIZE)
            y_lvl = random.randint(0, h_lvl - PATCH_SIZE)
            x_0 = int(x_lvl * downsample)
            y_0 = int(y_lvl * downsample)

            try:
                patch = wsi.read_region((x_0, y_0), level, (PATCH_SIZE, PATCH_SIZE))
                patch = patch.convert('RGB')
                np_patch = np.array(patch)
                if np_patch.mean() < TISSUE_THRESHOLD:
                    patches.append(self.transform(patch))
            except:
                pass

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

    def _validate_paths(self):
        # (Skrócona dla czytelności, taka sama jak wcześniej)
        pass


class CameleyonTestDataset(Dataset):
    def __init__(self, normal_dir, tumor_dir, bag_size=100, bags_per_slide=BAGS_PER_SLIDE, transform=None):
        self.normal_files = glob.glob(os.path.join(normal_dir, '*.tif'))
        self.tumor_files = glob.glob(os.path.join(tumor_dir, '*.tif'))
        self.bag_size = bag_size
        self.bags_per_slide = bags_per_slide  # Również dla testu generujemy X bagów

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
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.slide_paths) * self.bags_per_slide

    def __getitem__(self, idx):
        # Ta sama logika indeksowania co w treningu
        slide_idx = idx // self.bags_per_slide

        slide_path = self.slide_paths[slide_idx]
        label = self.labels[slide_idx]

        wsi = openslide.OpenSlide(slide_path)
        level = LEVEL if LEVEL < wsi.level_count else wsi.level_count - 1

        # W teście bierzemy losowo tkankę (tak jak chciałeś: "Tam gdzie wycinki brane są losowo bierz dalej losowo")
        # Ponieważ jest to losowe, każdy z 10 bagów będzie inny.
        patches = self._sample_tissue_patches(wsi, level, self.bag_size)

        if len(patches) < self.bag_size:
            diff = self.bag_size - len(patches)
            if len(patches) > 0:
                patches += [patches[0]] * diff
            else:
                patches = [torch.zeros(1, PATCH_SIZE, PATCH_SIZE)] * self.bag_size

        bag_tensor = torch.stack(patches)
        wsi.close()

        return bag_tensor, torch.tensor([label], dtype=torch.float32)

    def _sample_tissue_patches(self, wsi, level, count):
        """Losowe próbkowanie dla testu (tak samo jak było)"""
        patches = []
        w_lvl, h_lvl = wsi.level_dimensions[level]
        downsample = wsi.level_downsamples[level]
        attempts = 0
        max_attempts = count * 10

        while len(patches) < count and attempts < max_attempts:
            attempts += 1
            x_lvl = random.randint(0, w_lvl - PATCH_SIZE)
            y_lvl = random.randint(0, h_lvl - PATCH_SIZE)
            x_0 = int(x_lvl * downsample)
            y_0 = int(y_lvl * downsample)

            try:
                patch = wsi.read_region((x_0, y_0), level, (PATCH_SIZE, PATCH_SIZE))
                patch = patch.convert('RGB')
                np_patch = np.array(patch)
                if np_patch.mean() < TISSUE_THRESHOLD:
                    patches.append(self.transform(patch))
            except:
                continue
        return patches