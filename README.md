This repo contains the code for my bachelor's thesis "Stellar Classification of Main Sequence Stars using Vision Transformers". 

In order to run the python files, I assume you have already collected the csv containing all the relevant information from the SDSS SQL database.

Before starting on training/testing the model, preprocessing of the CSV, data collection, and data preprocessing are required. Run the Python files in the following order:
1. process_csv.py
2. download_RGB_data.py
3. download_fits_data.py
4. unpack_fits_data.py
5. extract_stars_from_fits.py
6. synthesize_rgb_from_bands.py
7. TODO (remaining code is on its way, I am working on cleaning it up and ensuring it runs in .py files without any required intervention.)
