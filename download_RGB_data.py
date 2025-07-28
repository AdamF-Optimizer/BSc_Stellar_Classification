import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Function to download images and classify
def download_image(ra, dec, mainclass, idx, phase, dataset, log_file):
    url = f"http://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.025&width=224&height=224"
    class_dir = os.path.join(f"RGB_{phase}_{dataset}", mainclass)  # Create a subdirectory for each star class
    os.makedirs(class_dir, exist_ok=True)  # Create the subdirectory if it doesn't exist
    image_path = os.path.join(class_dir, f'{mainclass}_class_image_ra_{ra}_dec_{dec}_rgb_image.png')

    with open(log_file, 'a') as log:
        if not os.path.exists(image_path): # Check if the image already exists
            response = requests.get(url)
            with open(image_path, 'wb') as f:
                f.write(response.content)
            log.write(f"Downloaded image {idx} to {image_path}\n")
        else:
            log.write(f"Image {idx} already exists at {image_path}, skipping download.\n")

    # Print time after every 1000 downloads
    if idx > 0 and idx % 1000 == 0:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Downloaded {idx} images")

# Function to handle parallel downloading
def parallel_download(data, phase, dataset, log_file):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx, row in data.iterrows():
            future = executor.submit(download_image, row['ra'], row['dec'], row['mainclass'], idx, phase, dataset, log_file)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                with open(log_file, 'a') as log:
                    log.write(f"An error occurred: {e}\n")

datasets = ["dataset_1", "dataset_2", "dataset_3"]

phases = ["train", "val", "test"]

for dataset in datasets:
    for phase in phases:
        data = pd.read_csv(f"{phase}_entries_{dataset}.csv", delimiter=',')

        # Read ra and dec celestial coordinates as floats
        data['ra'] = data['ra'].astype(float)
        data['dec'] = data['dec'].astype(float)

        os.makedirs(f"RGB_{phase}_{dataset}", exist_ok=True)  # Create the directory if it doesn't exist

        log_file = f"RGB_{phase}_{dataset}_downloading.txt" # Log file for downloading, avoids out of memory errors in browser

        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Processing {len(data)} images...")
        parallel_download(data, phase, dataset, log_file)
        print(f"Completed {phase} dataset {dataset}: {time.strftime('%Y-%m-%d %H:%M:%S')}")