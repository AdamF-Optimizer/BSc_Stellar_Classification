import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import warnings
from astropy.utils.exceptions import AstropyWarning

# There are deprecated columns in the FITS file headers. They're fixed automatically but print warnings, so I disabled the printing
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

bands = ['u', 'g', 'r', 'i', 'z']


def crop_star_image(csv, phase, dataset, crop_size=64):
    df = pd.read_csv(csv, delimiter=',')
    os.makedirs(f"{phase}_{dataset}_cropped_images_{crop_size}_x_{crop_size}", exist_ok=True)
    df['run'] = df['run'].astype(int).astype(str).apply(lambda x: x.zfill(6)) # Conversion required for filename
    df['field'] = df['field'].astype(int).astype(str).apply(lambda x: x.zfill(4)) # Conversion required for filename
    df['camcol'] = df['camcol'].astype(int)

    # Iterate through each star in the CSV
    for _, star in df.iterrows():
        for band in bands:
            # Construct the filename pattern
            fits_file = f"fits_images_{band}_band/frame-{band}-{star['run']}-{star['camcol']}-{star['field']}.fits"

            if os.path.exists(fits_file):
                try:
                    with fits.open(fits_file) as hdul:
                        image_data = hdul[0].data
                        header = hdul[0].header
                        wcs = WCS(header) # Get WCS information from the FITS header https://docs.astropy.org/en/stable/wcs/
                        star_coord = SkyCoord(ra=star['ra']*u.degree, dec=star['dec']*u.degree) # SkyCoord object for the star
                        cutout = Cutout2D(image_data, star_coord, (64, 64), wcs=wcs) # Make the crop

                        # Create a new FITS Header Data Unit with the cropped data
                        new_hdu = fits.PrimaryHDU(cutout.data)
                        new_hdu.header.update(cutout.wcs.to_header())
                        new_filename = f"{phase}_{dataset}_cropped_images_{crop_size}_x_{crop_size}/{star['mainclass']}_class_cutout_specobjid_{star['specobjid']}_ra_{star['ra']}_dec_{star['dec']}_band_{band}.fits"
                        new_hdu.writeto(new_filename, overwrite=True) # Save the new FITS file
                except Exception as e:
                    print(f"Error processing {fits_file}: {e}")
                    continue
            else:
                print(fits_file)
                print(f"No matching file found for star {star['specobjid']}, ra {star['ra']}, dec {star['dec']}, run {star['run']}, field {star['field']}, camcol {star['camcol']} in band {band}")


datasets = ["dataset_1", "dataset_2", "dataset_3"]
phases = ["train", "val", "test"]

for dataset in datasets:
    for phase in phases:
        crop_star_image(f"{phase}_entries_{dataset}.csv", phase, dataset, crop_size=64)