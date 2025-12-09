import os
import glob
import numpy as np
import openslide
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from matplotlib.path import Path as MplPath

LEVEL = 3
PATCH_SIZE = 96
TISSUE_THRESHOLD = 220
PATCHES_PER_CLASS = 10000


def get_xml_polygons(xml_path):
    if not os.path.exists(xml_path):
        return []

    try:
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
                polygons.append(MplPath(coords))
        return polygons
    except:
        return []


def process_slide_task(args):
    slide_path, output_root, annot_dir, is_tumor_source = args

    slide_name = Path(slide_path).stem
    slide_out_dir = os.path.join(output_root, slide_name)

    path_normal = os.path.join(slide_out_dir, 'normal')
    path_tumor = os.path.join(slide_out_dir, 'tumor')
    os.makedirs(path_normal, exist_ok=True)
    if is_tumor_source:
        os.makedirs(path_tumor, exist_ok=True)

    try:
        wsi = openslide.OpenSlide(slide_path)

        eff_level = LEVEL if LEVEL < wsi.level_count else wsi.level_count - 1
        w_lvl, h_lvl = wsi.level_dimensions[eff_level]
        downsample = wsi.level_downsamples[eff_level]

        full_img = wsi.read_region((0, 0), eff_level, (w_lvl, h_lvl)).convert('L')
        full_arr = np.array(full_img)
        wsi.close()

        polygons = []
        if is_tumor_source and annot_dir:
            xml_fname = slide_name + '.xml'
            xml_path = os.path.join(annot_dir, xml_fname)
            polygons = get_xml_polygons(xml_path)

        count_normal = 0
        count_tumor = 0
        attempts = 0
        max_attempts = PATCHES_PER_CLASS * 100

        while attempts < max_attempts:
            attempts += 1

            if is_tumor_source:
                if count_tumor >= PATCHES_PER_CLASS and count_normal >= PATCHES_PER_CLASS:
                    break
            else:
                if count_normal >= PATCHES_PER_CLASS:
                    break

            x_lvl = np.random.randint(0, w_lvl - PATCH_SIZE)
            y_lvl = np.random.randint(0, h_lvl - PATCH_SIZE)

            patch_arr = full_arr[y_lvl: y_lvl + PATCH_SIZE, x_lvl: x_lvl + PATCH_SIZE]

            if np.mean(patch_arr) >= TISSUE_THRESHOLD:
                continue

            is_tumor_patch = False
            if polygons:
                center_x_0 = (x_lvl * downsample) + (PATCH_SIZE * downsample / 2)
                center_y_0 = (y_lvl * downsample) + (PATCH_SIZE * downsample / 2)
                point = (center_x_0, center_y_0)

                for poly in polygons:
                    if poly.contains_point(point):
                        is_tumor_patch = True
                        break

            patch_img = Image.fromarray(patch_arr)

            if is_tumor_patch and count_tumor < PATCHES_PER_CLASS:
                fname = f"t_{count_tumor}_{x_lvl}_{y_lvl}.png"
                patch_img.save(os.path.join(path_tumor, fname))
                count_tumor += 1

            elif not is_tumor_patch and count_normal < PATCHES_PER_CLASS:
                fname = f"n_{count_normal}_{x_lvl}_{y_lvl}.png"
                patch_img.save(os.path.join(path_normal, fname))
                count_normal += 1

        return f"OK: {slide_name} (N:{count_normal}, T:{count_tumor})"

    except Exception as e:
        return f"ERR: {slide_name} -> {str(e)}"


def run_preprocessing(input_dir, output_dir, annot_dir=None, is_tumor_set=False, workers=16):
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    print(f"Preprocessing: {len(tif_files)} slajdów. Tryb Tumor: {is_tumor_set}")

    tasks = []
    for tif in tif_files:
        tasks.append((tif, output_dir, annot_dir, is_tumor_set))
    print(tasks)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_slide_task, tasks), total=len(tasks)))

    errors = [r for r in results if r.startswith("ERR")]
    if errors:
        print(f"\nNapotkano {len(errors)} błędów:")
        for e in errors[:5]: print(e)

def process_some_tasks(tasks: list[tuple[str, str, str, bool]], max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_slide_task, tasks), total=len(tasks)))
    errors = [r for r in results if r.startswith("ERR")]
    if errors:
        print(f"\nNapotkano {len(errors)} błędów:")
        for e in errors[:5]: print(e)


if __name__ == '__main__':
    run_preprocessing(
        input_dir=r"D:\CAMELEYON16\training\normal",
        output_dir=r"D:\CAMELEYON16\preprocessed_L3_96\train\normal",
        annot_dir=None,
        is_tumor_set=False,
        workers=8
    )

    # process_some_tasks([(r'D:\\CAMELEYON16\\training\\tumor\\tumor_008.tif', r'D:\\CAMELEYON16\\preprocessed_L3\\train\\tumor', r'D:\\CAMELEYON16\\training\\lesion_annotations', True), (r'D:\\CAMELEYON16\\training\\tumor\\tumor_010.tif', r'D:\\CAMELEYON16\\preprocessed_L3\\train\\tumor', r'D:\\CAMELEYON16\\training\\lesion_annotations', True), (r'D:\\CAMELEYON16\\training\\tumor\\tumor_011.tif', r'D:\\CAMELEYON16\\preprocessed_L3\\train\\tumor', r'D:\\CAMELEYON16\\training\\lesion_annotations', True), (r'D:\\CAMELEYON16\\training\\tumor\\tumor_012.tif', r'D:\\CAMELEYON16\\preprocessed_L3\\train\\tumor', r'D:\\CAMELEYON16\\training\\lesion_annotations', True), (r'D:\\CAMELEYON16\\training\\tumor\\tumor_013.tif', r'D:\\CAMELEYON16\\preprocessed_L3\\train\\tumor', r'D:\\CAMELEYON16\\training\\lesion_annotations', True)])

    run_preprocessing(
        input_dir=r"D:\CAMELEYON16\training\tumor",
        output_dir=r"D:\CAMELEYON16\preprocessed_L3_96\train\tumor",
        annot_dir=r"D:\CAMELEYON16\training\lesion_annotations",
        is_tumor_set=True,
        workers=8
    )

    run_preprocessing(
        input_dir=r"D:\CAMELEYON16\test\images\tumor",
        output_dir=r"D:\CAMELEYON16\preprocessed_L3_96\test\tumor",
        annot_dir=None,
        is_tumor_set=False, # Co prawda rak jest ale nie mamy adnotacji do niego
        workers=8
    )

    run_preprocessing(
        input_dir=r"D:\CAMELEYON16\test\images\normal",
        output_dir=r"D:\CAMELEYON16\preprocessed_L3_96\test\normal",
        annot_dir=None,
        is_tumor_set=False,
        workers=8
    )