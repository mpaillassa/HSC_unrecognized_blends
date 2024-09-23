import math
import os
import random as rd

import btk
import galcheat
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


def main():

    # data directories
    data_dir = "../data"
    psf_dir = os.path.join(data_dir, "galsim_cosmos_dataset/sanity_check/hsc_psfs")

    # set directories
    set_name = "_ms8"
    suffix = "_release_run"
    inp_dir = os.path.join(data_dir, f"blending/set_with_hst{set_name}{suffix}")
    out_dir = os.path.join(
        data_dir, f"blending/set_with_hst{set_name}_fake_coadds{suffix}"
    )
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # number of stamps to make fake coadds for
    # this should be in agreement with make_btk_stamps.py
    nb_stamps = 200000

    # fake coadd arragement
    # the size of a fake coadd must be 4100x4100
    # we use a margin of 150 and 40x40 stamp per coadd
    # with the btk stamp_size being 16 arcsec = 95 pixels for HSC
    # we effectively end up with 150*2 + 40*95 = 300 + 3800 = 4100
    stamp_size = 16  # arcsec
    margin = 150
    nx = 40
    ny = 40

    # get survey object
    hsc_survey = btk.survey.get_surveys("HSC")

    # make a point to draw stars later
    pix_stamp_size = int(stamp_size / hsc_survey.pixel_scale.value)
    point = np.zeros([pix_stamp_size, pix_stamp_size])
    point[pix_stamp_size // 2, pix_stamp_size // 2] = 1

    # star map
    star_map = make_star_map(nx, ny)

    # compute the number of fake coadds to make
    nb_stars_per_coadd = np.sum(star_map)
    nb_stamps_per_coadd = nx * ny - nb_stars_per_coadd
    nb_coadds = math.ceil(nb_stamps / nb_stamps_per_coadd)

    # modify sky brightnesses, this was experimentally set to match mag/snr plots of sources measured in real HSC coadds
    filters = hsc_survey.available_filters
    offsets = [-0.15, 0.15, 0.75, 1.4, 0.5]
    for i, f in enumerate(filters):
        filt = hsc_survey.get_filter(f)
        val = filt.sky_brightness.value
        filt.sky_brightness = val + offsets[i]

    # make the mask of the image margins
    border_mask = np.zeros([COADD_SIZE, COADD_SIZE], dtype=np.uint8)
    (
        border_mask[:margin, :],
        border_mask[-margin:, :],
        border_mask[:, :margin],
        border_mask[:, -margin:],
    ) = (1, 1, 1, 1)
    idx = np.where(border_mask)

    # for each coadd
    for coadd in tqdm.tqdm(range(nb_coadds), desc="Fake coadds"):

        # get stamp idx
        stamp_idx = int(coadd * nb_stamps_per_coadd)

        # prepare coadd image
        coadd_image = np.zeros([len(filters), COADD_SIZE, COADD_SIZE])

        # loop through stamps
        for y in range(ny):
            for x in range(nx):

                # stop when we reach the end of stamps
                if stamp_idx == nb_stamps:
                    break

                # compute position of each stamp
                yb = y * pix_stamp_size + margin
                ye = (y + 1) * pix_stamp_size + margin
                xb = x * pix_stamp_size + margin
                xe = (x + 1) * pix_stamp_size + margin

                # add either a star or a galaxy in the stamp
                if star_map[y, x]:
                    im = make_star_im(point, hsc_survey, filters, pix_stamp_size)
                else:
                    btk_file_path = os.path.join(inp_dir, f"{stamp_idx}_stamp.fits")
                    im = fits.getdata(btk_file_path)
                    stamp_idx += 1

                # fill the coadd image
                coadd_image[:, yb:ye, xb:xe] = im

        # make a fake sky image in each band to fill margin areas in the coadds and compute variance
        for ifilt, filt in enumerate(filters):
            # make sky image
            sky_flux = mean_sky_level(hsc_survey, filt).value
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


if __name__ == "__main__":
    main()
