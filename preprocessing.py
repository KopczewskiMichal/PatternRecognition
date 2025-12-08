import os
import glob
import numpy as np
import openslide
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def process_single_slide(args):
    tif_path, output_root, n_patches, patch_size, level, white_thresh = args

    slide_name = Path(tif_path).stem
    slide_out_dir = os.path.join(output_root, slide_name)

    try:
        os.makedirs(slide_out_dir, exist_ok=True)

        slide = openslide.OpenSlide(tif_path)

        eff_level = level if level < slide.level_count else slide.level_count - 1
        w, h = slide.level_dimensions[eff_level]

        img_pil = slide.read_region((0, 0), eff_level, (w, h)).convert('L')
        img_arr = np.array(img_pil)  # Kształt (H, W)

        H, W = img_arr.shape
        saved_count = 0
        attempts = 0
        max_attempts = n_patches * 100

        while saved_count < n_patches and attempts < max_attempts:
            attempts += 1

            x = np.random.randint(0, W - patch_size)
            y = np.random.randint(0, H - patch_size)

            patch_arr = img_arr[y:y + patch_size, x:x + patch_size]

            if np.mean(patch_arr) < white_thresh:
                patch_img = Image.fromarray(patch_arr)
                save_path = os.path.join(slide_out_dir, f"p_{saved_count}_{x}_{y}.png")
                patch_img.save(save_path)
                saved_count += 1

        slide.close()
        return f"OK: {slide_name} ({saved_count}/{n_patches})"

    except Exception as e:
        return f"ERR: {slide_name} -> {str(e)}"


def run_preprocessing(input_dir, output_dir, workers=16):
    LEVEL = 3
    N_PATCHES = 2000
    PATCH_SIZE = 28
    WHITE_THRESH = 200

    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    print(f"Found {len(tif_files)} .tif files in {input_dir}")

    tasks = []
    for tif in tif_files:
        tasks.append((tif, output_dir, N_PATCHES, PATCH_SIZE, LEVEL, WHITE_THRESH))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_single_slide, tasks), total=len(tasks), unit="slide"))


if __name__ == '__main__':
    INPUT_DIR = r"D:\CAMELEYON16\training\normal"

    OUTPUT_DIR = r'D:\CAMELEYON16\preprocessed\training\normal'

    NUM_WORKERS = 16

    run_preprocessing(INPUT_DIR, OUTPUT_DIR, NUM_WORKERS)