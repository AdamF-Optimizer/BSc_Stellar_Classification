# https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/
# See https://docs.python.org/3/library/concurrent.futures.html, the same as https://docs.python.org/3/library/multiprocessing.html
import os
import pandas as pd
import concurrent.futures
from tqdm import tqdm # Loading bar

def download_file(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    command = f"wget -q -nc -P {save_dir} {url}" # -nc to skip already downloaded files
    os.system(command)

def generate_urls_and_dirs(df):
    for i in range(len(df)):
        rerun = int(df['rerun'][i])
        run1 = int(df['run'][i])
        run2 = str(int(run1)).zfill(6)
        camcol = int(df['camcol'][i])
        field = str(int(df['field'][i])).zfill(4)

        for band in ['u', 'g', 'r', 'i', 'z']:
            image_url = f"https://dr18.sdss.org/sas/dr18/prior-surveys/sdss4-dr17-eboss/photoObj/frames/{rerun}/{run1}/{camcol}/frame-{band}-{run2}-{camcol}-{field}.fits.bz2"
            # print(image_url) # Debugging URL
            image_save_dir = f"fits_images_{band}_band_bz2/"
            yield (image_url, image_save_dir)

        # Catalog info, not used here but it contains optional additional information
        # catalog_url = f"https://dr18.sdss.org/sas/dr18/prior-surveys/sdss4-dr17-eboss/sweeps/dr13_final/{rerun}/calibObj-{run2}-{camcol}-star.fits.gz"
        # catalog_save_dir = f"catalogue_fits_gz/"
        # yield (catalog_url, catalog_save_dir)

fits_entries_csvs = ["MSS_final_cleaned_camera_data_dataset_1.csv", "MSS_final_cleaned_camera_data_dataset_2.csv", "MSS_final_cleaned_camera_data_dataset_3.csv"]

for csv in fits_entries_csvs:
    sampled_unique_fits_entries = pd.read_csv(csv) # Read unique fits image values into DataFrame
    total_downloads = len(sampled_unique_fits_entries) * 5  # 5 image bands
    # print(total_downloads)

    # Use ThreadPoolExecutor to perform parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for url, save_dir in generate_urls_and_dirs(sampled_unique_fits_entries):
            futures.append(executor.submit(download_file, url, save_dir))

        for _ in tqdm(concurrent.futures.as_completed(futures), total=total_downloads, desc=f"Downloading files from {csv}."): # progress bar
            pass