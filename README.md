This repo contains the code for my bachelor's thesis "Stellar Classification of Main Sequence Stars using Vision Transformers". 

In order to run the Python files, I assume you have already collected the CSV containing all the relevant information from the SDSS SQL database.

Before starting on training/testing the model, preprocessing of the CSV, data collection, and data preprocessing are required. Run the Python files in the following order:
1. process_csv.py: Processes the CSV file obtained from the SDSS SQL database in order to obtain class counts, dataset splits, etc.
2. download_RGB_data.py: Downloads RGB imaging data provided by SDSS.
3. download_fits_data.py: Downloads full-frame FITS files for individual band imaging data.
4. unpack_fits_data.py: Unpacks the compressed FITS files.
5. extract_stars_from_fits.py: Crops stars out of the full-frame FITS images.
6. synthesize_rgb_from_bands.py: Synthesizes RGB images from individual band FITS images (gri, urz, side-by-side).
7. dataset_dictionary_creation.py: Converts the data to the correct format expected by the ViT.
8. train_and_test.py: Trains and tests the models for each dataset and image type.

Although the code is not identical to the code in the appendix, its functionality remains the same. If you have any questions, I recommend referring to the Methods section of my thesis. If your questions remain unanswered, feel free to open an issue and I'll get back to you ASAP.
