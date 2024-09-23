import math
import os
import random as rd

import btk
import numpy as np
import tqdm
from astropy.io import fits
from astropy.table import Table
from galcheat.utilities import mag2counts, mean_sky_level
from scipy.signal import convolve as conv

COADD_SIZE = 4100


def make_star_map(nx, ny):
    """Makes a map with regularly spaced locations for stars.

    Args:
        nx, ny (int, int): size of the map.

    Returns:
        star_map (np.ndarray): star map.
    """

    star_map = np.zeros([nx, ny])

    k = 1
    for y in range(ny):
        for x in range(nx):
            if y == 0 or y == ny - 1:
                if not x % 10:
                    star_map[y, x] = 1
            if x == 0 or x == nx - 1:
                if not y % 10:
                    star_map[y, x] = 1
            if x == nx - 1 and y == ny - 1:
                star_map[y, x] = 1

            if x != 0 and y != 0 and x != nx - 1 and y != ny - 1:
                if not y % 2:
                    if not y % 4:
                        if not x % 10:
                            star_map[y, x] = 1
                    else:
                        if not (x + 5) % 10:
                            star_map[y, x] = 1
            k += 1

    return star_map


def make_star_im(point, btk_survey, filters, pix_stamp_size):
    """Makes the image of star using a similar process that BTK does for galaxies.

    Args:
        point (np.ndarray): image of a point.
        btk_survey (btk.survey.Survey): btk survey object.
        filters (list): list of galcheat.filter.Filter filter objects.
        pix_stamp_size (int): size of the image stamp in pixels.
    Returns:
        star_im (np.ndarray): image of a star.
    """

    # star image
    star_im = np.zeros([len(filters), pix_stamp_size, pix_stamp_size], dtype=np.float32)

    # pick a random magnitude for the star
    star_mag = rd.uniform(19, 21)

    # generate image in each band
    for ifilt, filt in enumerate(filters):
        # compute star flux
        flux = mag2counts(star_mag, btk_survey, filt).value

        # make the star image
        if callable(btk_survey.get_filter(filt).psf):
            psf = btk_survey.get_filter(filt).psf().drawImage().array
        else:
            psf = btk_survey.get_filter(filt).psf.drawImage().array
        star_im[ifilt] = conv(point, psf, "same", "fft")
        star_im[ifilt] /= np.sum(star_im[ifilt])
        star_im[ifilt] *= flux

        # add background and noise
        sky_flux = mean_sky_level(btk_survey, filt).value
        star_im[ifilt] += sky_flux
        star_im[ifilt] = np.random.poisson(star_im[ifilt]).astype(np.float32)
        star_im[ifilt] -= sky_flux

    return star_im


def make_btk_im(btk_survey, btk_catalog, btk_sampling_function, gal_idx, stamp_size):
    """Makes the image of the given galaxy with BTK.

    Args:
        btk_survey (btk.survey.Survey): btk survey object.
        btk_catalog (btk.catalog.Catalog): btk catalog object.
        btk_sampling_function (btk.sampling_functions.SamplingFunction): btk sampling function object.
        gal_idx (int): index of the galaxy to simulate.
        stamp_size (float): size of the image (in arcsec).

    Returns:
        image (np.ndarray): image of the galaxy.
        x, y (float, float): coordinates of the peak position of the galaxy.
    """

    # draw generator
    # this is not optimal to instantiate it at each sample
    # but this is the only way to go to make all samples by manually setting shifts and indexes ?
    # could use a batch_size>1 to make less instantiations
    indexes = [[gal_idx]]
    draw_generator = btk.draw_blends.CosmosGenerator(
        btk_catalog,
        btk_sampling_function,
        [btk_survey],
        batch_size=1,
        stamp_size=stamp_size,
        shifts=[[[0], [0]]],
        indexes=indexes,
        cpus=1,
        add_noise="all",
        gal_type="real",
        seed=gal_idx,
    )

    # generate stamps
    batch = next(draw_generator)

    # get images and pos
    image = batch["blend_images"][0]
    tab = batch["blend_list"][0]
    x, y = tab["x_peak"], tab["y_peak"]

    return image, x, y


def make_fake_coadds(
    margin,
    nx,
    ny,
    stamp_size,
    nb_galaxies,
    btk_catalog,
    btk_sampling_function,
    btk_survey,
    filters,
    out_dir,
):
    """Makes fake coadd images along with their variance images.

    Args:
        margin (int): margin to set around the grid of stamps.
        nx, ny (int, int): size of the grid of stamps.
        stamp_size (float): stamp sizes (in arcsec).
        nb_galaxies (int): number of galaxies to put in coadds.
        btk_catalog (btk.catalog.Catalog): btk catalog object.
        btk_sampling_function (btk.sampling_functions.SamplingFunction): btk sampling function object.
        btk_survey (btk.survey.Survey): btk survey object.
        filters (list): list of galcheat.filter.Filter filter objects.
        out_dir (string): output directory to save coadd and variance images.
    """

    # make star map
    star_map = make_star_map(nx, ny)

    # make a point to draw stars later
    pix_stamp_size = int(stamp_size / btk_survey.pixel_scale.value)
    point = np.zeros([pix_stamp_size, pix_stamp_size])
    point[pix_stamp_size // 2, pix_stamp_size // 2] = 1

    # compute the number of fake coadds to make
    nb_stars_per_coadd = np.sum(star_map)
    nb_gal_per_coadd = nx * ny - nb_stars_per_coadd
    nb_coadds = math.ceil(nb_galaxies / nb_gal_per_coadd)

    # make the mask of the image margins
    border_mask = np.zeros([COADD_SIZE, COADD_SIZE], dtype=np.uint8)
    (
        border_mask[:margin, :],
        border_mask[-margin:, :],
        border_mask[:, :margin],
        border_mask[:, -margin:],
    ) = (1, 1, 1, 1)
    idx = np.where(border_mask)

    # rows of the catalog to save for easier further matching after detection and measurements
    row_names = ["galsim_idx", "coadd_nb", "x", "y"]
    rows = []

    # make fake coadds
    for coadd in tqdm.tqdm(range(nb_coadds), desc=f"Building fake coadds"):

        # get the initial galaxy id
        gal_idx = int(coadd * nb_gal_per_coadd)

        # initialize the current fake coadd image
        coadd_image = np.zeros([len(filters), COADD_SIZE, COADD_SIZE])

        # for each stamp to put in the fake coadd
        for y in range(ny):
            for x in range(nx):

                # stop when we reach the end of stamps
                if gal_idx == nb_galaxies:
                    break

                # compute the position of the current stamp in the fake coadd
                yb = y * pix_stamp_size + margin
                ye = (y + 1) * pix_stamp_size + margin
                xb = x * pix_stamp_size + margin
                xe = (x + 1) * pix_stamp_size + margin

                # make either a star or a galaxy stamp
                if star_map[y, x]:
                    im = make_star_im(point, btk_survey, filters, pix_stamp_size)
                else:
                    im, xs, ys = make_btk_im(
                        btk_survey,
                        btk_catalog,
                        btk_sampling_function,
                        gal_idx,
                        stamp_size,
                    )

                    rows.append([gal_idx, coadd, xb + xs, yb + ys])
                    gal_idx += 1

                # fill the coadd image
                coadd_image[:, yb:ye, xb:xe] = im

        # make a fake sky image in each band to fill margin areas in the coadds and compute variance
        for ifilt, filt in enumerate(filters):
            # make sky image
            sky_flux = mean_sky_level(btk_survey, filt).value
            fake_sky = np.ones([COADD_SIZE, COADD_SIZE]) * sky_flux
            fake_sky = np.random.poisson(fake_sky).astype(np.float32) - sky_flux

            # fill coadd image margin areas
            coadd_image[ifilt, idx[0], idx[1]] = fake_sky[idx[0], idx[1]]

            # make variance image
            var = np.std(fake_sky) ** 2
            var_image = np.ones([COADD_SIZE, COADD_SIZE]) * var

            # save coadd and variance images
            hdu = fits.PrimaryHDU(coadd_image[ifilt])
            hdu.writeto(
                os.path.join(out_dir, "coadd_%s_%05d.fits" % (filt, coadd)),
                overwrite=True,
            )
            hdu = fits.PrimaryHDU(var_image)
            hdu.writeto(
                os.path.join(out_dir, "var_%s_%05d.fits" % (filt, coadd)),
                overwrite=True,
            )

    # recording
    final_table = Table(rows=rows, names=row_names)
    final_table_path = os.path.join(out_dir, "catalog.fits")
    final_table.write(final_table_path, overwrite=True)


def main():

    # COSMOS catalog
    data_dir = "../../../data"
    suffix = "_release_run"
    cat_dir = os.path.join(data_dir, f"galsim_cosmos_dataset/full{suffix}")
    catalog_names = [
        os.path.join(cat_dir, "real_galaxy_catalog_26_extension.fits"),
        os.path.join(cat_dir, "real_galaxy_catalog_26_extension_fits.fits"),
    ]

    # BTK stuff
    btk_catalog = btk.catalog.CosmosCatalog.from_file(
        catalog_names, exclusion_level="none"
    )
    stamp_size = 16  # arcsec, 95 in pixels
    btk_sampling_function = btk.sampling_functions.DefaultSampling(
        max_number=1, stamp_size=stamp_size, max_shift=0
    )
    hsc_survey = btk.survey.get_surveys("HSC")

    # setup PSFs
    filters = hsc_survey.available_filters
    psf_dir = os.path.join(data_dir, "hsc_psfs")
    for f in filters:
        filt = hsc_survey.get_filter(f)
        filt.psf = lambda: btk.survey.get_psf_from_file(
            os.path.join(psf_dir, f), hsc_survey
        )

    # modify sky brightnesses, this was experimentally set to match mag/snr plots of sources measured in real HSC coadds
    offsets = [-0.15, 0.15, 0.75, 1.4, 0.5]
    for i, f in enumerate(filters):
        filt = hsc_survey.get_filter(f)
        val = filt.sky_brightness.value
        filt.sky_brightness = val + offsets[i]

    # number of galaxies
    with fits.open(catalog_names[0]) as hducat:
        cat = hducat[1].data
        nb_galaxies = len(cat)

    # fake coadd arragement
    # the size of a fake coadd must be 4100x4100
    # we use a margin of 150 and 40x40 stamp per coadd
    # with the btk stamp_size being 16 arcsec = 95 pixels for HSC
    # we effectively end up with 150*2 + 40*95 = 300 + 3800 = 4100
    margin = 150
    nx = 40
    ny = 40

    # output directory
    out_dir = os.path.join(
        data_dir, f"galsim_cosmos_dataset/validation_process/fake_coadds{suffix}"
    )
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # make fake coadds
    make_fake_coadds(
        margin,
        nx,
        ny,
        stamp_size,
        nb_galaxies,
        btk_catalog,
        btk_sampling_function,
        hsc_survey,
        filters,
        out_dir,
    )


if __name__ == "__main__":
    main()
