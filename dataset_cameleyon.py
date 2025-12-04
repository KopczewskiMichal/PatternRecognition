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

# --- KONFIGURACJA ---
PATCH_SIZE = 28  # Twój model wymaga wejścia 28x28 (wynika z architektury)
LEVEL = 3  # Poziom powiększenia (0 = max, 1 = 4x mniejszy, itd.). 1 jest szybszy.
BAG_SIZE = 100  # Ile wycinków (instancji) w jednym worku. Zmniejsz jeśli braknie VRAM.
TISSUE_THRESHOLD = 220  # Jasność powyżej której uznajemy, że to białe tło (0-255)


class CameleyonBagDataset(Dataset):
    def __init__(self, normal_dir, tumor_dir, annot_dir, bag_size=BAG_SIZE, transform=None):
        self.normal_files = glob.glob(os.path.join(normal_dir, '*.tif'))
        self.tumor_files = glob.glob(os.path.join(tumor_dir, '*.tif'))
        self.annot_dir = annot_dir
        self.bag_size = bag_size

        # Tworzymy listę wszystkich slajdów i ich etykiet
        self.slide_paths = []
        self.labels = []  # 0 = normal, 1 = tumor

        for p in self.normal_files:
            self.slide_paths.append(p)
            self.labels.append(0)

        for p in self.tumor_files:
            self.slide_paths.append(p)
            self.labels.append(1)

        self._validate_paths()

        # Transformacje: Grayscale (bo model ma in_channels=1) + Tensor
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((PATCH_SIZE, PATCH_SIZE)),  # Upewniamy się że jest 28x28
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        slide_path = self.slide_paths[idx]
        label = self.labels[idx]

        # 1. Otwieramy WSI
        wsi = openslide.OpenSlide(slide_path)

        # Sprawdzamy czy żądany poziom istnieje, jak nie to bierzemy ostatni
        level = LEVEL if LEVEL < wsi.level_count else wsi.level_count - 1

        patches = []

        # 2. Strategia pobierania wycinków
        if label == 0:
            # --- NORMAL SLIDE ---
            # Pobieramy losowe wycinki, które nie są tłem
            patches = self._sample_normal_patches(wsi, level, self.bag_size)

        else:
            # --- TUMOR SLIDE ---
            # Pobieramy 50% z guza (bazując na XML) i 50% losowo (zdrowa tkanka w chorym slajdzie)
            xml_path = self._get_xml_path(slide_path)

            if xml_path and os.path.exists(xml_path):
                tumor_polygons = self._parse_xml_polygons(xml_path)

                # Połowa z guza
                n_tumor = self.bag_size // 2
                patches_tumor = self._sample_tumor_patches(wsi, level, tumor_polygons, n_tumor)

                # Połowa "normalna" (lub reszta)
                n_normal = self.bag_size - len(patches_tumor)
                patches_normal = self._sample_normal_patches(wsi, level,
                                                             n_normal)  # Tu przydałoby się sprawdzanie czy nie wpadliśmy w tumor, ale dla szybkości pomijam

                patches = patches_tumor + patches_normal
            else:
                # Jeśli brak XML, traktujemy jako normalne próbkowanie (fallback)
                patches = self._sample_normal_patches(wsi, level, self.bag_size)

        # Jeśli z jakiegoś powodu (np. same białe tło) nie uzbieraliśmy patches, duplikujemy
        if len(patches) < self.bag_size:
            diff = self.bag_size - len(patches)
            # Dopychamy zerami lub duplikatami (tu duplikaty pierwszego)
            if len(patches) > 0:
                patches += [patches[0]] * diff
            else:
                # Skrajny przypadek: pusty czarny obraz
                patches = [torch.zeros(1, PATCH_SIZE, PATCH_SIZE)] * self.bag_size

        # Konwersja listy tensorów na jeden tensor (Bag)
        # Wymiar: [Bag_Size, 1, 28, 28]
        bag_tensor = torch.stack(patches)

        wsi.close()

        return bag_tensor, torch.tensor([label], dtype=torch.float32)

    # --- METODY POMOCNICZE ---

    def _get_xml_path(self, slide_path):
        """Zamienia ścieżkę .tif na ścieżkę .xml w folderze adnotacji"""
        filename = os.path.basename(slide_path)
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        return os.path.join(self.annot_dir, xml_filename)

    def _parse_xml_polygons(self, xml_path):
        """Parsuje Twój format XML do listy obiektów Path z matplotlib"""
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

    def _sample_normal_patches(self, wsi, level, count):
        """Losuje wycinki unikając białego tła"""
        patches = []
        w_lvl, h_lvl = wsi.level_dimensions[level]
        # Przelicznik współrzędnych (bo XML i read_region używają level 0)
        downsample = wsi.level_downsamples[level]

        attempts = 0
        max_attempts = count * 5  # Zabezpieczenie przed pętlą nieskończoną

        while len(patches) < count and attempts < max_attempts:
            attempts += 1
            # Losujemy x, y na poziomie LEVEL
            x_lvl = random.randint(0, w_lvl - PATCH_SIZE)
            y_lvl = random.randint(0, h_lvl - PATCH_SIZE)

            # Przeliczamy na Level 0 (do read_region)
            x_0 = int(x_lvl * downsample)
            y_0 = int(y_lvl * downsample)

            # Czytamy region.
            # UWAGA: read_region zawsze bierze (x,y) z level 0, ale rozmiar podajemy dla docelowego levelu
            patch = wsi.read_region((x_0, y_0), level, (PATCH_SIZE, PATCH_SIZE))
            patch = patch.convert('RGB')

            # Prosty filtr tła: jeśli średnia jasność > progu, to pewnie tło
            np_patch = np.array(patch)
            if np_patch.mean() < TISSUE_THRESHOLD:
                patches.append(self.transform(patch))

        return patches

    def _sample_tumor_patches(self, wsi, level, polygons, count):
        """Losuje wycinki Z WNĘTRZA wielokątów raka"""
        patches = []
        if not polygons:
            return []

        downsample = wsi.level_downsamples[level]
        attempts = 0
        max_attempts = count * 10

        while len(patches) < count and attempts < max_attempts:
            attempts += 1
            # Losujemy poligon
            poly = random.choice(polygons)
            # Pobieramy Bounding Box poligonu dla szybszego losowania
            (minx, miny, maxx, maxy) = poly.get_extents().bounds

            # Losujemy punkt wewnątrz Bounding Boxa
            x_rand = random.uniform(minx, maxx)
            y_rand = random.uniform(miny, maxy)

            # Sprawdzamy czy punkt faktycznie jest w środku poligonu (precyzyjnie)
            if poly.contains_point((x_rand, y_rand)):
                # Znaleźliśmy punkt (lewy górny róg patcha na Level 0)
                x_0 = int(x_rand)
                y_0 = int(y_rand)

                try:
                    patch = wsi.read_region((x_0, y_0), level, (PATCH_SIZE, PATCH_SIZE))
                    patch = patch.convert('RGB')
                    patches.append(self.transform(patch))
                except:
                    continue  # Czasem wyjdzie poza obraz

        return patches

    def _validate_paths(self):
        """Wyświetla ścieżki i liczbę znalezionych plików dla celów diagnostycznych."""

        total_slides = len(self.normal_files) + len(self.tumor_files)

        print("\n--- DIAGNOSTYKA ZBIORU CAMELEYON16 ---")

        # Sprawdzamy, czy katalogi w ogóle istnieją
        if not os.path.isdir(os.path.dirname(self.normal_files[0])) if self.normal_files else False:
            print(f"BŁĄD: Katalog Normal nie istnieje lub jest źle zdefiniowany.")

        # Pliki normalne
        print(
            f"Ścieżka dla NORMAL: {os.path.dirname(self.normal_files[0]) if self.normal_files else '<Nie znaleziono>'}")
        print(f"Znaleziono plików NORMAL (.tif): {len(self.normal_files)}")

        # Pliki tumor
        print(f"Ścieżka dla TUMOR: {os.path.dirname(self.tumor_files[0]) if self.tumor_files else '<Nie znaleziono>'}")
        print(f"Znaleziono plików TUMOR (.tif): {len(self.tumor_files)}")

        # Pliki adnotacji (nie sprawdzamy wszystkich, tylko katalog)
        first_tumor_file = os.path.basename(self.tumor_files[0]) if self.tumor_files else None
        expected_xml = os.path.join(self.annot_dir,
                                    first_tumor_file.replace('.tif', '.xml')) if first_tumor_file else '<Nie dotyczy>'

        print(f"Katalog ADNOTACJI: {self.annot_dir}")
        print(f"Przykładowa ścieżka XML: {expected_xml}")
        print(f"Czy oczekiwany XML istnieje? {'TAK' if os.path.exists(expected_xml) else 'NIE'}")

        print(f"\nŁącznie worków (slajdów) znaleziono: {total_slides}")
        print("-----------------------------------------------------")

        if total_slides == 0:
            raise ValueError("BŁĄD FATALNY: Nie znaleziono żadnych slajdów WSI. Sprawdź ścieżki i rozszerzenia.")


class CameleyonTestDataset(Dataset):
    def __init__(self, normal_dir, tumor_dir, bag_size=100, transform=None):
        self.normal_files = glob.glob(os.path.join(normal_dir, '*.tif'))
        self.tumor_files = glob.glob(os.path.join(tumor_dir, '*.tif'))
        self.bag_size = bag_size

        # Budujemy listę
        self.slide_paths = []
        self.labels = []

        for p in self.normal_files:
            self.slide_paths.append(p)
            self.labels.append(0)  # 0 = Zdrowy

        for p in self.tumor_files:
            self.slide_paths.append(p)
            self.labels.append(1)  # 1 = Chory

        # Standardowa transformacja
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self._validate_paths()

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        slide_path = self.slide_paths[idx]
        label = self.labels[idx]

        wsi = openslide.OpenSlide(slide_path)
        level = LEVEL if LEVEL < wsi.level_count else wsi.level_count - 1

        # W teście NIE używamy XML.
        # Model musi sam znaleźć raka w losowo pobranej tkance.
        patches = self._sample_tissue_patches(wsi, level, self.bag_size)

        # Zabezpieczenie przed pustym workiem
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
        """
        Pobiera losowe wycinki, ale tylko jeśli nie są białym tłem.
        W teście traktujemy tak samo pliki Normal i Tumor.
        """
        patches = []
        w_lvl, h_lvl = wsi.level_dimensions[level]
        downsample = wsi.level_downsamples[level]

        attempts = 0
        max_attempts = count * 10  # Dajemy mu więcej szans na znalezienie tkanki

        while len(patches) < count and attempts < max_attempts:
            attempts += 1
            x_lvl = random.randint(0, w_lvl - PATCH_SIZE)
            y_lvl = random.randint(0, h_lvl - PATCH_SIZE)

            x_0 = int(x_lvl * downsample)
            y_0 = int(y_lvl * downsample)

            try:
                patch = wsi.read_region((x_0, y_0), level, (PATCH_SIZE, PATCH_SIZE))
                patch = patch.convert('RGB')

                # Prosty check: czy to nie jest białe tło?
                np_patch = np.array(patch)
                if np_patch.mean() < TISSUE_THRESHOLD:
                    patches.append(self.transform(patch))
            except:
                continue

        return patches

    def _validate_paths(self):
        print("\n--- TEST DATASET DIAGNOSTICS ---")
        print(f"Test Normal: {len(self.normal_files)} slides")
        print(f"Test Tumor: {len(self.tumor_files)} slides")
        if len(self.normal_files) + len(self.tumor_files) == 0:
            print("WARNING: Pusty zbiór testowy!")