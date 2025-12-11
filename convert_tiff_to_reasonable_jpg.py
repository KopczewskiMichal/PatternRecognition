import openslide
import os


def save_level_as_jpg(tiff_path, output_path, level=4):
    if not os.path.exists(tiff_path):
        print(f"BŁĄD: Nie znaleziono pliku: {tiff_path}")
        return

    try:
        slide = openslide.OpenSlide(tiff_path)

        print(f"Liczba poziomów: {slide.level_count}")
        print(f"Wymiary na poziomie 0 (max): {slide.dimensions}")

        if level >= slide.level_count:
            print(f"UWAGA: Poziom {level} nie istnieje (max to {slide.level_count - 1}).")
            level = slide.level_count - 1
            print(f"Przełączono na poziom: {level}")

        dims = slide.level_dimensions[level]
        print(f"Wymiary na poziomie {level}: {dims}")

        image = slide.read_region((0, 0), level, dims)

        image = image.convert('RGB')

        image.save(output_path, quality=90)
        print(f"SUKCES: Zapisano obraz do: {output_path}")

        slide.close()

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


input_tiff = r"D:\CAMELEYON16\test\images\normal\test_014.tif"
output_jpg = './podglad_level4_im_test_24.jpg'

save_level_as_jpg(input_tiff, output_jpg, level=4)