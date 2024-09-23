import math
import os
import random as rd

import galsim
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import KDTree


def make_data_set(catalog, whole_hst_catalog, out_dir, div, lim, data_dir):
    """Makes a GalSim COSMOS data set.
    This writes the given images, PSF images and catalog in the corresponding output directory.

    Args:
        catalog (string): path to hst catalog subset to pick sources in.
        whole_hst_catalog (string): path to the whole hst catalog to get neighbors.
        out_dir (string): path to output directory.
        div (int): number of stamps per output image file.
        lim (int): number of outputs to generate, set to -1 to make the whole set.
        data_dir (string): path to the directory containig additional data like HST mosaic files or PSF models
    """

    # get the catalog from which getting galaxies
    with fits.open(catalog) as fitscat:
        cat = fitscat[1].data

    # get the whole hst catalog data to get neighbor information
    with fits.open(whole_hst_catalog) as wcat:
        hst_cat = wcat[1].data

    # build the HST tree to later look at neighbors
    tree_hst = build_hst_tree(hst_cat)

    # HST mosaic file to get image cutouts
    mosaic_dir = os.path.join(data_dir, "HST_mosaic")
    hst_mosaic = read_hst_mosaic(mosaic_dir)

    # get the PSF images
    psf_dir = os.path.join(data_dir, "HST_PSF")
    psf_models = read_psf_models(psf_dir)

    # final tables
    rows = []  # for the first GalSim COSMOS catalog
    rows2 = []  # for second GalSim COSMOS catalog

    gen_count = 0  # general counter
    file_count = 0  # file stamp counter
    galaxy_stamps = fits.HDUList()  # current FITS images file
    psf_stamps = fits.HDUList()  # curren FITS psf images file
    hdu_count = 0  # hdu file stamp counter

    # loop through cat to make galaxy stamps
    for source in cat:

        # works only if lim < div
        # because hdu_count cannot be greater than div
        if hdu_count == lim:
            break

        # get image stamp and wcs
        im_name = os.path.join(out_dir, "side_images/iobj%i_HST.fits" % gen_count)
        if not os.path.isfile(im_name):
            # if the cutout have not been extracted yet, do it
            ok, im, im_wcs = get_image_cutout(source, hst_mosaic)
            extracted = True
        else:
            # else read it directly
            im_hdu = fits.open(im_name)
            im = im_hdu[0].data.copy()
            im_wcs = wcs.WCS(im_hdu[0].header)
            ok = True
            extracted = False

        if not ok:
            # if cutout could not be extracted
            print("Cutout extraction failed (out of CCD): skipping the source")
        else:
            # if cutout could be retrieved
            if extracted:
                # if it has been extracted, save it
                hdu = fits.PrimaryHDU(im)
                hdu.header.update(im_wcs.to_header())
                hdu.writeto(im_name, overwrite=True)

            # get neighboring sources
            h, w = im.shape
            r_arcsec = math.sqrt((0.5 * h + 1) ** 2 + (0.5 * w + 1) ** 2) * 0.03
            r_deg = r_arcsec / 3600.0
            r_rad = r_deg * 2 * np.pi / 360.0
            dist = 2.0 * np.sin(0.5 * r_rad)
            source_ids = get_neighboring_sources(source, dist, tree_hst)

            # get pixels to replace by noise
            ma_name = im_name.replace(".fits", ".mask.fits")
            if not os.path.isfile(ma_name):
                # if masks do not exist
                neighbors, main_source_mask = make_source_masks(
                    source_ids, hst_cat, source, im_wcs, h, w
                )
                masks = np.zeros([2, h, w], dtype=np.uint8)
                masks[0] = main_source_mask
                masks[1] = neighbors
                hdu = fits.PrimaryHDU(masks)
                hdu.writeto(ma_name, overwrite=True)
            else:
                # else just read them
                ma_hdu = fits.open(ma_name)
                ma = ma_hdu[0].data.copy()
                neighbors, main_source_mask = ma[1], ma[0]

            if np.any(np.logical_and(neighbors, main_source_mask)):
                print("Neighbor overlaps with the main source: skipping the source")
            else:
                # replace neighbor pixels by noise
                noise_mean, noise_std = replace_pixels(neighbors, main_source_mask, im)

                # change files if full
                # be careful, not the expected format if div < lim
                if hdu_count == div:
                    galaxy_stamps.writeto(
                        os.path.join(
                            out_dir,
                            "real_galaxy_images_extension_n%i.fits" % file_count,
                        )
                    )
                    psf_stamps.writeto(
                        os.path.join(
                            out_dir,
                            "real_galaxy_PSF_images_extension_n%i.fits" % file_count,
                        )
                    )
                    galaxy_stamps = fits.HDUList()
                    psf_stamps = fits.HDUList()
                    file_count += 1
                    hdu_count = 0

                # add galaxy stamp
                galaxy_stamps.append(fits.ImageHDU(im))

                # add PSF stamp
                psf = get_psf_stamp(psf_models, source, psf_dir)
                psf_stamps.append(fits.ImageHDU(psf))

                # fill first catalog
                ra, dec, mag = (
                    source["ALPHA_J2000_1"],
                    source["DELTA_J2000_1"],
                    source["MAG_AUTO_1"],
                )
                if lim == 100:
                    rows.append(
                        [
                            gen_count,
                            ra,
                            dec,
                            mag,
                            "F814W",
                            1.0,
                            "real_galaxy_images_extension.fits",
                            "real_galaxy_PSF_images_extension.fits",
                            hdu_count,
                            hdu_count,
                            0.03,
                            noise_mean,
                            noise_std * noise_std,
                            "acs_I_unrot_sci_20_cf.fits",
                            source["FLUX_AUTO_1"],
                        ]
                    )
                else:
                    rows.append(
                        [
                            gen_count,
                            ra,
                            dec,
                            mag,
                            "F814W",
                            1.0,
                            "real_galaxy_images_extension_n%i.fits" % file_count,
                            "real_galaxy_PSF_images_extension_n%i.fits" % file_count,
                            hdu_count,
                            hdu_count,
                            0.03,
                            noise_mean,
                            noise_std * noise_std,
                            "acs_I_unrot_sci_20_cf.fits",
                            source["FLUX_AUTO_1"],
                        ]
                    )
                hdu_count += 1

                # fill second catalog
                rows2.append(
                    [
                        gen_count,
                        source["FLUX_RADIUS_1"][1],
                        source["g_HSC"],
                        source["r_HSC"],
                        source["i_HSC"],
                        source["z_HSC"],
                        source["y_HSC"],
                        source["ZPHOT_1"],
                    ]
                )

        gen_count += 1

    # write the first catalog
    col_names = [
        "IDENT",
        "RA",
        "DEC",
        "MAG",
        "BAND",
        "WEIGHT",
        "GAL_FILENAME",
        "PSF_FILENAME",
        "GAL_HDU",
        "PSF_HDU",
        "PIXEL_SCALE",
        "NOISE_MEAN",
        "NOISE_VARIANCE",
        "NOISE_FILENAME",
        "stamp_flux",
    ]
    col_types = [
        "i4",
        "f8",
        "f8",
        "f8",
        "S5",
        "f8",
        "S64",
        "S64",
        "i4",
        "i4",
        "f8",
        "f8",
        "f8",
        "S32",
        "f8",
    ]
    final_table = Table(rows=rows, names=col_names, dtype=col_types)
    if lim == 100:
        final_table.write(
            os.path.join(out_dir, "real_galaxy_catalog_26_extension_example.fits"),
            overwrite=True,
        )
    else:
        final_table.write(
            os.path.join(out_dir, "real_galaxy_catalog_26_extension.fits"),
            overwrite=True,
        )

    # write the second catalog
    col_names2 = [
        "IDENT",
        "flux_radius",
        "HSC_g",
        "HSC_r",
        "HSC_i",
        "HSC_z",
        "HSC_y",
        "ZPHOT",
    ]
    col_types2 = ["i4", "f8", "f8", "f8", "f8", "f8", "f8", "f8"]
    final_table2 = Table(rows=rows2, names=col_names2, dtype=col_types2)
    if lim == 100:
        final_table2.write(
            os.path.join(out_dir, "real_galaxy_catalog_26_extension_example_fits.fits"),
            overwrite=True,
        )
    else:
        final_table2.write(
            os.path.join(out_dir, "real_galaxy_catalog_26_extension_fits.fits"),
            overwrite=True,
        )

    # don't forget to write the last (not full) image and PSF image files
    if len(galaxy_stamps) and len(psf_stamps):
        assert len(galaxy_stamps) == len(psf_stamps)
        if lim == 100:
            galaxy_stamps.writeto(
                os.path.join(out_dir, "real_galaxy_images_extension.fits"),
                overwrite=True,
            )
            psf_stamps.writeto(
                os.path.join(out_dir, "real_galaxy_PSF_images_extension.fits"),
                overwrite=True,
            )
        else:
            galaxy_stamps.writeto(
                os.path.join(
                    out_dir, "real_galaxy_images_extension_n%i.fits" % file_count
                ),
                overwrite=True,
            )
            psf_stamps.writeto(
                os.path.join(
                    out_dir, "real_galaxy_PSF_images_extension_n%i.fits" % file_count
                ),
                overwrite=True,
            )


def build_hst_tree(hst_cat):
    """Builds the HST tree to get neighbors of a given source.

    Args:
        hst_cat (astropy.io.fits.fitsrec.FITS_rec): HST catalog.

    Returns:
        tree_hst (scipy.spatial.KDTree): HST tree.
    """

    ra = hst_cat["ALPHA_J2000"]
    dec = hst_cat["DELTA_J2000"]
    phi = np.radians(ra)
    theta = np.pi / 2.0 - np.radians(dec)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    data = np.array([x, y, z]).transpose()
    tree_hst = KDTree(data)

    return tree_hst


def read_hst_mosaic(mosaic_dir):
    """Reads the HSC mosaic image files.

    Args:
        mosaic_dir (string): path to the HST mosaic file directory.

    Returns:
        ret (list): list of HST mosaic images and WCS.
    """

    hst_files = os.listdir(mosaic_dir)
    hst_files = [os.path.join(mosaic_dir, x) for x in hst_files if "fits" in x]
    ret = []
    for hst_file in hst_files:
        file_data = fits.getdata(hst_file)
        file_header = fits.getheader(hst_file)
        file_wcs = wcs.WCS(file_header)
        ret.append([file_data, file_wcs])

    return ret


def read_psf_models(psf_dir):
    """Reads the PSF image files.

    Args:
        psf_dir (string): path to the PSF file directory.
    Returns:
        psf_im (dict): dictionary where keys are focus values and values are PSF images.
    """

    psf_m = dict()
    for fv in range(-10, 2):
        psf_m[fv] = fits.getdata(os.path.join(psf_dir, "TinyTim_f%i.fits.gz" % fv))

    return psf_m


def get_image_cutout(source, hst_mosaic):
    """Gets the HST image cutout from the mosaic file.

    Args:
        source (astropy.io.fits.fitsrec.FITS_record): source HST catalog entry.
        hst_files (list): list of HST mosaic images and WCS.
    Returns:
        ok (bool): whether if the cutout could be extracted or not.
        im (np.ndarray): cutout image if ok=True, None if ok=False.
        im_wcs (astropy.wcs.wcs.WCS): cutout wcs if found=True, None if found=False
    """

    # compute temporary (larger) size
    tmp_size_arcsec = compute_stamp_size_official(source)
    tmp_size_pix = int(tmp_size_arcsec / 0.03)

    # get the final size
    tmp_source_mask = get_source_mask(
        source, tmp_size_pix // 2, tmp_size_pix // 2, tmp_size_pix, tmp_size_pix
    )
    idx = np.where(tmp_source_mask)
    ymin, ymax = np.min(idx[0]), np.max(idx[0])
    xmin, xmax = np.min(idx[1]), np.max(idx[1])
    ysize = ymax - ymin
    xsize = xmax - xmin
    size_pix = int(1.25 * max(ysize, xsize))

    # find in which file is the source
    ra, dec = source["ALPHA_J2000_1"], source["DELTA_J2000_1"]
    ok = False
    for file_data, file_wcs in hst_mosaic:
        x, y = file_wcs.all_world2pix([[ra, dec]], 0)[0].astype(np.int)
        xb, xe = x - size_pix // 2, x + size_pix // 2
        yb, ye = y - size_pix // 2, y + size_pix // 2
        if xb >= 0 and xe < 20480 and yb >= 0 and ye < 20480:
            ok = True
            im = file_data[yb:ye, xb:xe]
            im_wcs = file_wcs[yb:ye, xb:xe]
            break

    if ok:
        return ok, im, im_wcs
    else:
        return ok, None, None


def compute_stamp_size_official(source):
    """Computes the image stamp size of a given source using the paper formula.

    Args:
        source (int): HST id of the source.

    Returns:
        size (float): stamp size in arcsec.
    """

    # formula from the official galsim data set paper
    r = float(source["FLUX_RADIUS_1"][1])
    size = (
        11
        * math.sqrt(1.5 * 1.5 * r * r + (1.2 * 1.2) / (0.03 * 0.03 * 2.35 * 2.35))
        * 0.03
    )

    return size


def get_neighboring_sources(source, dist, tree_hst):
    """Gets the neighboring sources of a given source.

    Args:
        source (astropy.io.fits.fitsrec.FITS_record): source HST catalog entry.
        dist (float): maximal distance to look for in arcsec.
        tree_hst (scipy.spatial.KDTree): HST tree.

    Returns:
        sources (list): HST neighboring source ids.
    """

    ra, dec = source["ALPHA_J2000_1"], source["DELTA_J2000_1"]

    phi = np.radians(ra)
    theta = np.pi / 2.0 - np.radians(dec)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    data = np.array([x, y, z]).transpose()
    sources = tree_hst.query_ball_point(data, dist)

    return sources


def make_source_masks(source_ids, hst_cat, main_source, im_wcs, h, w):
    """Makes the masks of requested sources.

    Args:
        source_ids (list): list of source ids to make masks for.
        hst_cat (astropy.io.fits.fitsrec.FITS_rec): HST catalog.
        main_source (astropy.io.fits.fitsrec.FITS_record): main source HST catalog entry.
        im_wcs (astropy.wcs.wcs.WCS): image wcs to convert the coords to pixels.
        h, w (int, int): height and width of the image stamp.

    Returns:
        neighbors (np.ndarray): mask of the neighboring sources.
        main_source (np.ndarray): mask of the main source.
    """

    neighbors = np.zeros([h, w])
    main_ra, main_dec = main_source["ALPHA_J2000_1"], main_source["DELTA_J2000_1"]
    for source_id in source_ids:
        source = hst_cat[source_id]
        s_ra = source["ALPHA_J2000"]
        s_dec = source["DELTA_J2000"]
        x, y = im_wcs.all_world2pix([[s_ra, s_dec]], 0)[0].astype(np.int)
        if s_ra == main_ra and s_dec == main_dec:  # main source
            main_source = get_source_mask(source, x, y, h, w)
        else:  # neighbor
            neighbor = get_source_mask(source, x, y, h, w)
            neighbors = np.logical_or(neighbors, neighbor)

    return neighbors, main_source


def get_source_mask(source, x, y, h, w):
    """Gets the mask of a given source.

    Args:
        source (astropy.io.fits.fitsrec.FITS_record): source HST catalog entry.
        x, y (int, int): source position in the stamp.
        h, w (int, int): height and width of the image stamp.

    Returns:
        source_mask (np.ndarray): mask of the source.
    """

    source_mask = np.zeros([h, w], dtype=np.uint8)

    # following columns are only in the Leauthaud catalog so there is no issue
    # calling this function on cut on whole hst catalog (no need to _1 or _2)
    cyy = source["CXX_IMAGE"]
    cxx = source["CYY_IMAGE"]
    cxy = source["CXY_IMAGE"]
    for yy in range(h):
        for xx in range(w):
            if (
                yy >= 0
                and yy < h
                and xx >= 0
                and xx < w
                and cxx * (xx - x) ** 2
                + cyy * (yy - y) ** 2
                - cxy * (xx - x) * (yy - y)
                < 16
            ):
                source_mask[yy, xx] = 1

    return source_mask


def replace_pixels(masks, source_mask, im):
    """Replaces masked pixels by noise in the image.
    Modifies the image array <im> on-the-fly.

    Args:
        masks (np.ndarray): masked pixels to replace.
        source_mask (np.ndarray): mask of the main source.
        im (np.ndarray): image to replace pixels in.

    Returns:
        im_noise_mean, im_noise_std (float, float): statistics of the source image noise.
    """

    # get the image std
    full_mask = masks + source_mask
    idx = np.where(1 - full_mask)
    im_noise_mean = np.mean(im[idx[0], idx[1]])
    im_noise_std = np.std(im[idx[0], idx[1]])

    # get noise field:
    h, w = im.shape
    noise_field = get_noise_array(1, h, w)
    noise_std = np.std(noise_field)

    # replace pixels
    idx = np.where(masks)
    im[idx[0], idx[1]] = (im_noise_std / noise_std) * noise_field[idx[0], idx[1]]

    return im_noise_mean, im_noise_std


def get_noise_array(rng, h, w):
    """Gets a noise array for neighbor replacement.

    Args:
        rng (int): seed for galsim rng.
        h, w (int, int): height and width of the noise array to generate.

    Returns:
        noise_arr (np.ndarray): noise array.
    """

    if rng:
        rng = galsim.BaseDeviate(rng)
        noise = galsim.getCOSMOSNoise(rng=rng)
    else:
        noise = galsim.getCOSMOSNoise()
    noise_image = galsim.Image(h, w)
    noise_image.addNoise(noise)
    noise_arr = noise_image.array

    return noise_arr


def get_psf_stamp(psf_models, source, psf_dir, X=51, Y=65):
    """Gets the PSF stamp of a given source.

    Args:
        psf_models (dict): dictionary where keys are focus values and values are PSF images.
        source (astropy.io.fits.fitsrec.FITS_record): source HST catalog entry.
        psf_dir (string): path to the PSF file directory.
        X, Y (int, int): width and height of the PSF stamp to retrieve.

    Returns:
        psf_stamp (np.ndarray): psf stamp image.
    """

    x, y, focus = source["X_IMAGE_1"], source["Y_IMAGE_1"], source["FOCUS_MODEL"]
    with open(os.path.join(psf_dir, "TinyTim_f%i.stars.dat" % focus)) as fd:
        lines = fd.readlines()

    dist = 10000000
    psfx, psfy = -1, -1
    for pos in lines:
        pos_x, pos_y = pos.split()
        pos_dist = (float(pos_x) - x) ** 2 + (float(pos_y) - y) ** 2
        if pos_dist < dist:
            dist = pos_dist
            psfx, psfy = round(float(pos_x)), round(float(pos_y))

    psf_stamp = psf_models[focus][
        psfy - Y // 2 : psfy + Y // 2 + 1, psfx - X // 2 : psfx + X // 2 + 1
    ]

    return psf_stamp


def make_hst_cut(hst_cat_path):
    """Makes the HST catalog cut.
    This writes the corresponding catalog to disk.

    Args:
        hst_cat_path (string): path to the hst catalog.
    """

    selected_from_hst = []
    with fits.open(hst_cat_path) as hst:
        hst_cat = hst[1].data

        hst_cut = fits.HDUList()
        hst_cut.append(hst[0])

        for obj in hst_cat:
            if (
                obj["MU_CLASS"] == 1
                and obj["CLEAN"] == 1
                and obj["MAG_AUTO"] < 26
                and obj["GOOD"] == 1
            ):
                selected_from_hst.append(obj)

    init_cols = hst[1].columns
    col_names = []
    for col in range(len(init_cols)):
        col_names.append(init_cols[col].name)

    tab = Table(rows=selected_from_hst, names=col_names)
    tab.write("hst_cut_26_good.fits", overwrite=True)


def main():

    data_dir = "../data"
    galsim_cosmos_data_dir = os.path.join(data_dir, "galsim_cosmos_dataset")

    # full original HST ACS catalog from Leauthaud et al. 2007
    hst_catalog = os.path.join(
        galsim_cosmos_data_dir, "COSMOS_ACS_catalog/acs_clean_only.fits"
    )

    # this is a match between:
    # the original HST catalog cut with the make_hst_cut function,
    # the EL-COSMOS catalog to get HSC multiband magnitudes and photo-z,
    # the lensing14.fits HST catalog provided by R. Mandelbaum to get focus PSF information
    # the matches were made with topcat
    # (the original HST catalog and lensing14.fits are slightly different so may come from
    #  different SExtractor run, should have I had only used lensing14.fits to make it simpler ?)
    hst_match_lensing14 = os.path.join(
        galsim_cosmos_data_dir, "hst_cut26_el_cosmos_lensing14_match.fits"
    )

    # make a galsim COSMOS data set
    """
    # for the small example catalog of 100 entries
    out_dir = os.path.join(galsim_cosmos_data_dir, "example_release_run") # check that it's the same than the one already done
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(os.path.join(out_dir, "side_images")):
        os.mkdir(os.path.join(out_dir, "side_images"))
    make_data_set(hst_match_lensing14, hst_catalog, out_dir, 100, 100, data_dir)
    """
    # for the full catalog
    out_dir = os.path.join(galsim_cosmos_data_dir, "full_release_run")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(os.path.join(out_dir, "side_images")):
        os.mkdir(os.path.join(out_dir, "side_images"))
    make_data_set(hst_match_lensing14, hst_catalog, out_dir, 5000, -1, data_dir)


if __name__ == "__main__":
    main()
