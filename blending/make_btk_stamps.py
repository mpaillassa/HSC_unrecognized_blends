import os

import btk
import galsim
import numpy as np
from astropy.io import fits

###
suffix = "_release_run"
data_dir = "../data"
for_ml_training = True
if for_ml_training:
    max_shift = 0.168 * 8
    out_dir = os.path.join(data_dir, f"blending/set_with_hst_ms8{suffix}")
    seed = 1
else:
    max_shift = 0.168 * 4
    out_dir = os.path.join(data_dir, f"blending/set_with_hst_ms4{suffix}")
    seed = 2

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
nb_stamps = 200000
batch_size = 8

### catalog
cat_dir = os.path.join(data_dir, f"galsim_cosmos_dataset/full{suffix}")
catalog_names = [
    os.path.join(cat_dir, "real_galaxy_catalog_26_extension_detrefined.fits"),
    os.path.join(cat_dir, "real_galaxy_catalog_26_extension_detrefined_fits.fits"),
]
catalog = btk.catalog.CosmosCatalog.from_file(catalog_names, exclusion_level="none")

### sampling function
stamp_size = 16.0  # 95x95 pixels
max_number = 2
sampling_function = btk.sampling_functions.DefaultSampling(
    max_number=max_number, stamp_size=stamp_size, max_shift=max_shift, seed=seed
)

### surveys
hst_survey = btk.survey.get_surveys("COSMOS")
hsc_survey = btk.survey.get_surveys("HSC")
filters = hsc_survey.available_filters

# setup custom PSF
psf_dir = os.path.join(data_dir, "hsc_psfs")
for f in filters:
    filt = hsc_survey.get_filter(f)
    filt.psf = lambda: btk.survey.get_psf_from_file(
        os.path.join(psf_dir, f), hsc_survey
    )

# setup sky brightnesses
# this was experimentally set to match mag/snr plots of sources measured in real HSC coadds
offsets = [-0.15, 0.15, 0.75, 1.4, 0.5]
for i, f in enumerate(filters):
    filt = hsc_survey.get_filter(f)
    val = filt.sky_brightness.value
    filt.sky_brightness = val + offsets[i]

### draw generator
draw_generator = btk.draw_blends.CosmosGenerator(
    catalog,
    sampling_function,
    [hsc_survey, hst_survey],
    batch_size=batch_size,
    stamp_size=stamp_size,
    cpus=1,
    add_noise="all",
    verbose=False,
    gal_type="real",
    seed=seed,
)

# drawing loop
nb_steps = nb_stamps // batch_size
for k in range(nb_steps):
    if not k % 500:
        print(f"{k}/{nb_steps}")

    # generate a batch of stamps
    batch = next(draw_generator)

    # for each survey
    for surv in batch["blend_images"].keys():

        # for each element of the batch
        for b in range(batch_size):

            # get images and infos
            image = batch["blend_images"][surv][b]
            tab = batch["blend_list"][surv][b]

            # fill header
            hdu = fits.CompImageHDU(image)
            for obj in range(len(tab)):
                col_names = tab.columns
                for name in col_names:
                    hdu.header[f"HIERARCH {name}_{obj}"] = tab[obj][name]

            # save file
            if surv == "COSMOS":
                out_name = os.path.join(out_dir, f"{k*batch_size+b}_hst_stamp.fits")
            else:
                out_name = os.path.join(out_dir, f"{k*batch_size+b}_stamp.fits")
            hdu.writeto(out_name, overwrite=True)

            # get and save isolated images
            isolated_images = batch["isolated_images"][surv][b]
            nb_images = isolated_images.shape[0]
            hdul = fits.HDUList()
            for im in range(nb_images):
                hdu = fits.CompImageHDU(isolated_images[im])
                hdul.append(hdu)
            if surv == "COSMOS":
                out_name = os.path.join(out_dir, f"{k*batch_size+b}_hst_isostamp.fits")
            else:
                out_name = os.path.join(out_dir, f"{k*batch_size+b}_isostamp.fits")
            hdul.writeto(out_name, overwrite=True)
