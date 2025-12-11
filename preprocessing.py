import os
import glob
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

PATCH_SIZE = 96
WORKERS = 10

WHITE_THRESHOLD = 220
COLOR_VARIANCE = 20


def process_single_slide(args):
    tif_path, out_root, patch_size, level = args

    slide_name = os.path.splitext(os.path.basename(tif_path))[0]
    slide_out_dir = os.path.join(out_root, slide_name)

    try:
        os.makedirs(slide_out_dir, exist_ok=True)

        slide = openslide.OpenSlide(tif_path)

        eff_level = level if level < slide.level_count else slide.level_count - 1
        w_raw, h_raw = slide.level_dimensions[eff_level]

        target_w = (w_raw // patch_size) * patch_size
        target_h = (h_raw // patch_size) * patch_size

        img = slide.read_region((0, 0), eff_level, (w_raw, h_raw)).convert("RGB")
        slide.close()

        if (w_raw != target_w) or (h_raw != target_h):
            img = img.resize((target_w, target_h), resample=Image.BILINEAR)

        img_arr = np.array(img)  # Shape: (H, W, 3)

        saved_count = 0

        for y in range(0, target_h, patch_size):
            for x in range(0, target_w, patch_size):

                patch = img_arr[y: y + patch_size, x: x + patch_size]

                mean_intensity = np.mean(patch)
                if mean_intensity > WHITE_THRESHOLD:
                    continue

                mean_rgb = np.mean(patch, axis=(0, 1))  # np. [200, 100, 200]
                rgb_spread = np.max(mean_rgb) - np.min(mean_rgb)

                if rgb_spread < COLOR_VARIANCE:
                    continue

                patch_pil = Image.fromarray(patch).convert('L')

                out_name = f"{y}_{x}.jpg"
                patch_pil.save(os.path.join(slide_out_dir, out_name))
                saved_count += 1

        return f"OK: {slide_name} -> {saved_count} patches"

    except Exception as e:
        return f"ERR: {slide_name} -> {str(e)}"


def process_dir(input_dir:str, output_dir:str, level = 4):
    files = glob.glob(os.path.join(input_dir, "*.tif"))
    print(f"Found: {len(files)} slides. Start preprocessing on {WORKERS} threads...")
    tasks = [(f, output_dir, PATCH_SIZE, level) for f in files]

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_slide, tasks), total=len(tasks), unit="slide"))

    errors = [r for r in results if r.startswith("ERR")]
    if errors:
        print("\n--- ERRORS ---")
        for err in errors: print(err)



if __name__ == '__main__':
    output_dir = r"D:\CAMELEYON16\preprocessed_lv_5"
    level = 5
    process_dir(r"D:\CAMELEYON16\test\images\normal", os.path.join(output_dir, r"test\normal"), level)
    process_dir(r"D:\CAMELEYON16\test\images\tumor", os.path.join(output_dir, r"test\tumor"), level)
    process_dir(r"D:\CAMELEYON16\training\normal", os.path.join(output_dir, r"training\normal"), level)
    process_dir(r"D:\CAMELEYON16\training\tumor", os.path.join(output_dir, r"training\tumor"), level)