from tqdm import tqdm
import os
import shutil
import time
import numpy as np
from astropy.visualization import make_lupton_rgb
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

def normalize_image(image):
    scale_min, scale_max = np.percentile(image, [1, 99])
    scaled = np.clip((image - scale_min) / (scale_max - scale_min), 0, 1)
    return scaled

def combine_to_rgb(g_or_u_image, r_image, i_or_z_image, Q, alpha):
    # We resize to avoid images on edges of the full frames being cropped into smaller than 32x32 images
    g_or_u = normalize_image(resize(g_or_u_image, (crop_size, crop_size), preserve_range=True, anti_aliasing=True))
    r = normalize_image(resize(r_image, (crop_size, crop_size), preserve_range=True, anti_aliasing=True))
    i_or_z = normalize_image(resize(i_or_z_image, (crop_size, crop_size), preserve_range=True, anti_aliasing=True))
    return make_lupton_rgb(g_or_u, r, i_or_z, Q=Q, stretch=alpha) # Expects float, see https://docs.astropy.org/en/stable/visualization/rgb.html#astropy-visualization-rgb

def process_fits_file(filename):
    with fits.open(filename, memmap=True) as hdul: # allows the array data of each HDU to be accessed with mmap, rather than being read into memory all at once https://docs.astropy.org/en/stable/io/fits/
        return hdul[0].data.copy()

def process_image(prefix, input_directory, output_directory, band_set, Q, alpha):
    bands = {}
    for band in band_set:
        filename = os.path.join(input_directory, f"{prefix}_band_{band}.fits")
        if os.path.exists(filename):
            bands[band] = process_fits_file(filename)

    if all(band in bands for band in band_set):
        band1 = bands[band_set[0]]
        band2 = bands[band_set[1]]
        band3 = bands[band_set[2]]
        rgb_image = combine_to_rgb(band1, band2, band3, Q, alpha)
        plt.imsave(os.path.join(output_directory, f'{prefix}_{band_set}_RGB.png'), rgb_image, format='png', dpi=300)
        return True
    return False

# Process the images in batches to avoid crashing due to insufficient memory
def process_batch(batch, Q, alpha):
    results = []
    for item in batch:
        prefix, input_directory, gri_output, urz_output, Q, alpha = item
        gri_result = process_image(prefix, input_directory, gri_output, 'gri', Q, alpha)
        urz_result = process_image(prefix, input_directory, urz_output, 'urz', Q, alpha)
        results.append((gri_result, urz_result))
    return results

def process_directory(input_directory, gri_output_directory, urz_output_directory, Q, alpha):
    os.makedirs(gri_output_directory, exist_ok=True)
    os.makedirs(urz_output_directory, exist_ok=True)

    prefixes = set()
    for filename in os.listdir(input_directory):
        if filename.endswith('.fits'):
            prefix = filename.rsplit('_band_', 1)[0] # maxsplit parameter to 1, will return a list with 2 elements https://www.w3schools.com/python/ref_string_rsplit.asp
            prefixes.add(prefix)

    batch_size = 50
    batches = [list(prefixes)[i:i+batch_size] for i in range(0, len(prefixes), batch_size)]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for batch in batches:
            batch_items = [(prefix, input_directory, gri_output_directory, urz_output_directory, Q, alpha) for prefix in batch]
            futures.append(executor.submit(process_batch, batch_items, Q, alpha))

        with tqdm(total=len(prefixes) * 2, desc="Processing images", unit="step") as pbar:
            for future in as_completed(futures):
                results = future.result()
                for gri_result, urz_result in results:
                    if gri_result:
                        pbar.update(1)
                    if urz_result:
                        pbar.update(1)

def sort_files_by_class(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if not os.path.isfile(file_path):
            continue

        star_class = filename[0].upper()
        subdirectory_path = os.path.join(directory, star_class)
        os.makedirs(subdirectory_path, exist_ok=True)
        shutil.move(file_path, os.path.join(subdirectory_path, filename))


datasets = ["dataset_1", "dataset_2", "dataset_3"]
phases = ["train", "val", "test"]
bands = ['u', 'g', 'r', 'i', 'z']
crop_size = 64


# First, create the individual RGB images synthesized from GRI and URZ bands.
for dataset in datasets:
    for phase in phases:
        print(f"Processing directory {phase}_{dataset}_cropped_images_{crop_size}_x_{crop_size}")
        for band in bands:
            try:
                print(f"Processing {band} band")
                process_directory(f"{phase}_{dataset}_cropped_images_{crop_size}_x_{crop_size}", f"{phase}_{dataset}_gri_images_{crop_size}_x_{crop_size}", f"{phase}_{dataset}_urz_images_{crop_size}_x_{crop_size}", 8, 0.02)
                sort_files_by_class(f"{phase}_{dataset}_gri_images_{crop_size}_x_{crop_size}")
                sort_files_by_class(f"{phase}_{dataset}_urz_images_{crop_size}_x_{crop_size}")
                print()
            except Exception as e:
                print(f"Error: {e}")
                continue


# Then, create side-by-side images from the two previously created images.

def get_image_pairs(gri_dir, urz_dir, class_label):
    gri_class_dir = os.path.join(gri_dir, class_label)
    urz_class_dir = os.path.join(urz_dir, class_label)

    gri_images = [f for f in os.listdir(gri_class_dir) if f.endswith('.png')]
    urz_images = [f for f in os.listdir(urz_class_dir) if f.endswith('.png')]
    image_pairs = list(zip(gri_images, urz_images))  # Create image pairs

    return image_pairs, gri_class_dir, urz_class_dir


for dataset in datasets:
    for phase in phases:
        os.makedirs(f"{phase}_{dataset}_side_by_side_images_{crop_size}_x_{crop_size}", exist_ok=True)
        for class_label in ['A', 'B', 'F', 'G', 'K', 'M', 'O']:
            output_class_dir = os.path.join(f"{phase}_{dataset}_side_by_side_images_{crop_size}_x_{crop_size}", class_label)
            os.makedirs(output_class_dir, exist_ok=True)
            
            try:
                image_pairs, gri_class_dir, urz_class_dir = get_image_pairs(gri_dir=f"{phase}_{dataset}_gri_images_{crop_size}_x_{crop_size}", urz_dir=f"{phase}_{dataset}_urz_images_{crop_size}_x_{crop_size}", class_label=class_label)
            except Exception as e:
                print(f"Error: {e}")
                continue
            

            for gri_image_name, urz_image_name in image_pairs:
                gri_image_path = os.path.join(gri_class_dir, gri_image_name)
                urz_image_path = os.path.join(urz_class_dir, urz_image_name)
                gri_image = Image.open(gri_image_path)
                urz_image = Image.open(urz_image_path)

                # Combine the images side by side
                try:
                    combined_img = Image.new('RGB', (128, 64))
                    combined_img.paste(gri_image, (0, 0))
                    combined_img.paste(urz_image, (64, 0))

                    base_name = os.path.basename(gri_image_path).replace('_gri_RGB.png', '_side_by_side.png')
                    output_path = os.path.join(output_class_dir, base_name)
                    combined_img.save(output_path)

                    print(f'Saved {output_path}')
                except Exception as e:
                    print(f"Error saving {output_path}: {e}")


# Finally, upscale images to 224x224


# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
def upscale_and_save_image(image_path, output_path):
    try:
        if os.path.exists(output_path): # Skip if image exists
            return

        with Image.open(image_path) as image:
            upscaled_image = image.resize((224, 224), Image.LANCZOS) # Resize image to 224x224
            upscaled_image.save(output_path)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")



for dataset in datasets:
    for phase in phases:
        for image_type in ["gri", "urz", "side_by_side"]:
            input_dir = f"{phase}_{dataset}_{image_type}_images_{crop_size}_x_{crop_size}"
            output_dir = f"{phase}_{dataset}_{image_type}_images_224_x_224"

            counter = 0
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    input_path = os.path.join(root, file)
                    os.makedirs(output_dir, exist_ok=True)

                    relative_path = os.path.relpath(root, input_dir) # Relative input path
                    output_class_dir = os.path.join(output_dir, relative_path)
                    output_path = os.path.join(output_class_dir, file)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    try:
                        upscale_and_save_image(input_path, output_path)
                        counter += 1
                        if counter % 1000 == 0:
                            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Upscaled {counter} images")
                    except Exception as e:
                        print(f"Error processing {input_path}: {e}")

            print(f"All {image_type} images in phase {phase} for {dataset} have been upscaled and saved to {output_dir}.")
print(f"All images have been upscaled and saved.")