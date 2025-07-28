This repo contains the code for my bachelor's thesis "Stellar Classification of Main Sequence Stars using Vision Transformers". 

In order to run the Python files, I assume you have already collected the CSV containing all the relevant information from the SDSS SQL database.

Before starting on training/testing the model, preprocessing of the CSV, data collection, and data preprocessing are required. Run the Python files in the following order:
1. process_csv.py
2. download_RGB_data.py
3. download_fits_data.py
4. unpack_fits_data.py
5. extract_stars_from_fits.py
6. synthesize_rgb_from_bands.py
7. TODO (remaining code is on its way, I am working on cleaning it up and ensuring it runs in .py files without any required intervention.)


Although the code is not identical to the code in the appendix, its functionality remains the same. If you have any questions, I recommend referring to the Methods section of my thesis. If your questions remain unanswered, feel free to open an issue and I'll get back to you ASAP.
