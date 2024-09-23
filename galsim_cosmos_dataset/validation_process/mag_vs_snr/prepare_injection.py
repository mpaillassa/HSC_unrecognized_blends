import functools
import os
import random as rd
import sys

import galsim
import lsst.daf.persistence as dafPersist
import numpy as np
import scipy.ndimage
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import label
from scipy.spatial import KDTree


def main():

    # data directories
    data_dir = "../../../data"
    suffix = "_release_run"

    # data for source injection
    tract = "9813"
    first_patch = 37  # so that the first patch is 4,1 (arbitrary choice)
    bands = ["g", "r", "i", "z", "y"]

    # to simulate images of sources
    ident_psf = np.zeros([21, 21])
    ident_psf[10, 10] = 1
    galsim_ident_psf = galsim.InterpolatedImage(
        galsim.Image(ident_psf), scale=0.168
    ).withFlux(1.0)
    gal_cat = galsim.COSMOSCatalog(
        os.path.join(
            data_dir,
            f"galsim_cosmos_dataset/full{suffix}/real_galaxy_catalog_26_extension.fits",
        ),
        exclusion_level="none",
    )

    # second galsim catalog to get the EL COSMOS true magnitudes
    with fits.open(
        os.path.join(
            data_dir,
            f"galsim_cosmos_dataset/full{suffix}/real_galaxy_catalog_26_extension_fits.fits",
        )
    ) as fitscat:
        galsim_fitcatalog = fitscat[1].data

    # butler to read data with DM stack (especially masks and WCS to inject sources in empty areas)
    butler = dafPersist.Butler(root=os.path.join(data_dir, "hsc_data/rerun/pdr2_wide"))

    # injection catalog info
    injcat_col_names = [
        "ra",
        "dec",
        "x",
        "y",
        "g_mag",
        "r_mag",
        "i_mag",
        "z_mag",
        "y_mag",
        "gimFilename",
        "rimFilename",
        "iimFilename",
        "zimFilename",
        "yimFilename",
        "sourceType",
        "bulge_n",
        "bulge_pa",
        "disk_n",
        "disk_pa",
        "select",
        "bulge_axis_ratio",
        "bulge_semimajor",
        "disk_axis_ratio",
        "disk_semimajor",
    ]
    injcat_col_types = [
        "f8",
        "f8",
        "i8",
        "i8",
        "f8",
        "f8",
        "f8",
        "f8",
        "f8",
        "S128",
        "S128",
        "S128",
        "S128",
        "S128",
        "S32",
        "f8",
        "f8",
        "f8",
        "f8",
        "f8",
        "f8",
        "f8",
        "f8",
        "f8",
    ]

    # pick a random subset of galaxies
    nb_sources_to_inject = 50000
    gal_indexes = np.random.randint(0, len(galsim_fitcatalog), nb_sources_to_inject)

    out_dir = os.path.join(data_dir, f"for_magsnr_source_injection{suffix}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # write the selected galaxy indices for backup
    np.save(os.path.join(out_dir, "galaxy_selection.npy"), gal_indexes)

    nb_sources_done = 0
    current_patch = first_patch
    while nb_sources_done != nb_sources_to_inject:

        i1, i2 = current_patch // 9, current_patch % 9
        patch_s = f"{i1},{i2}"
        print(f"Adding sources in patch {patch_s}")

        # get coadd info
        im = butler.get(
            "deepCoadd_calexp",
            dataId={"tract": int(tract), "patch": patch_s, "filter": "HSC-I"},
        )
        wcs = im.getWcs()
        bbox = im.getBBox()
        minx, miny = bbox.getBeginX(), bbox.getBeginY()
        maxx, maxy = bbox.getEndX(), bbox.getEndY()

        mask = im.getMask().array
        struct = scipy.ndimage.generate_binary_structure(2, 1)
        mask = scipy.ndimage.morphology.binary_dilation(
            mask, struct, iterations=5
        ).astype(np.uint8)

        # prepare the images and injection catalog entries for the given patch
        patch_done = False
        rows = []
        while not patch_done:
            # pick a position in the empty areas
            available_idx = np.where(1 - mask)
            nb_idx = len(available_idx[0])
            r_idx = rd.randint(0, nb_idx - 1)
            rx, ry = available_idx[1][r_idx], available_idx[0][r_idx]

            # remove 100x100 pixels around this position from empty areas
            # to prevent injecting sources at the same location
            yb = max(0, ry - 50)
            ye = min(mask.shape[0], ry + 50)
            xb = max(0, rx - 50)
            xe = min(mask.shape[1], rx + 50)
            mask[yb:ye, xb:xe] = 1

            # convert it to world coordinates for the injection catalog
            rx, ry = rx + 200 + minx, ry + 200 + miny
            ra, dec = wcs.pixelToSky(rx, ry)

            # get the corresponding galsim catalog entry index
            k = gal_indexes[nb_sources_done]

            # make the injection catalog entry
            im_name = f"{k}.fits"
            im_path = os.path.join(out_dir, im_name)
            rows.append(
                [
                    ra,
                    dec,
                    rx - minx,
                    ry - miny,
                    galsim_fitcatalog[k]["HSC_g"],
                    galsim_fitcatalog[k]["HSC_r"],
                    galsim_fitcatalog[k]["HSC_i"],
                    galsim_fitcatalog[k]["HSC_z"],
                    galsim_fitcatalog[k]["HSC_y"],
                    im_path,
                    im_path,
                    im_path,
                    im_path,
                    im_path,
                    "galaxy",
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )

            # make the image
            if not os.path.isfile(im_path):
                gal_im = gal_cat.makeGalaxy(k, gal_type="real", noise_pad_size=0)
                gal_im_conv = galsim.Convolve(gal_im, galsim_ident_psf)
                hdu = fits.PrimaryHDU(gal_im_conv.drawImage(scale=0.168).array)
                hdu.writeto(im_path, overwrite=True)

            # increment the number of sources done
            nb_sources_done += 1
            if not nb_sources_done % 200:
                print(nb_idx)

            # if we have done all sources, write catalog and end
            if nb_sources_done == nb_sources_to_inject:
                patch_done = True
                all_done = True
                final_cat = Table(
                    rows=rows, names=injcat_col_names, dtype=injcat_col_types
                )
                final_cat.write(
                    os.path.join(out_dir, f"{tract}_{patch_s}.fits"), overwrite=True
                )
                print(
                    f"{patch_s} done, {nb_sources_done} done out of {nb_sources_to_inject}"
                )
                break

            # if there are no more empty areas, write catalog and go to next patch
            if np.all(mask):
                patch_done = True
                final_cat = Table(
                    rows=rows, names=injcat_col_names, dtype=injcat_col_types
                )
                final_cat.write(
                    os.path.join(out_dir, f"{tract}_{patch_s}.fits"), overwrite=True
                )
                print(
                    f"{patch_s} done, {nb_sources_done} done out of {nb_sources_to_inject}"
                )
                current_patch += 1
                rows = []


if __name__ == "__main__":
    main()
