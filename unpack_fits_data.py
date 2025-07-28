import bz2
import shutil
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def unpack_file(filepath, dest_dir):
    filename = os.path.basename(filepath)
    if filepath.endswith('.bz2'):
        newpath = os.path.join(dest_dir, filename[:-4])  # Remove .bz2 extension
        try:
            with bz2.open(filepath, 'rb') as source, open(newpath, 'wb') as dest:
                shutil.copyfileobj(source, dest)
            # os.remove(filepath)  # Optinally, remove the original compressed file
        except EOFError:
            print(f"Corrupted file detected: {filepath}.")
            os.remove(filepath)  # Remove the corrupted file

def unpack_directory(directory, dest_dir):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.bz2'):
                yield os.path.join(root, file), dest_dir

# List of directories to process
directories = [
    ("fits_images_u_band_bz2/", "fits_images_u_band/"),
    ("fits_images_g_band_bz2/", "fits_images_g_band/"),
    ("fits_images_r_band_bz2/", "fits_images_r_band/"),
    ("fits_images_i_band_bz2/", "fits_images_i_band/"),
    ("fits_images_z_band_bz2/", "fits_images_z_band/")
]

files_to_unpack = []
for src_dir, dest_dir in directories:
    os.makedirs(dest_dir, exist_ok=True)
    files_to_unpack.extend(unpack_directory(src_dir, dest_dir))

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(lambda x: unpack_file(*x), files_to_unpack), 
                total=len(files_to_unpack), desc="Unpacking files"))

print("All files have been unpacked.")
