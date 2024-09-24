import os
import random as rd

import matplotlib as mpl
import numpy as np
from astropy.io import fits

mpl.use("Agg")
import math

import galcheat
import matplotlib.pyplot as plt
import tqdm
from astropy import wcs
from astropy.table import Table

# this should be consistent with make_fake_coadds.py
nb = 298662
nb_coadds = 197
nx = 40
ny = 40
margin = 150
cell_size = 95


def retrieve_det_meas(rerun_dir, gt_cat_dir, band):
    """Retrieves detections and measurements for each isolated source drawned with BTK.

    For each stamp is recorded how many detections are in this stamp, and how many matches are in this stamp.
    Additional measurements are recorded for the best matching detection.
    Some are used for this validation process and some are used later to compare measurements when isolated and when blended.

    Args:
        rerun_dir (string): rerun directory containing HSC pipeline output products.
        gt_cat_dir (string): directory containing the ground truth catalog.
        band (string): band to process.

    Returns:
        detected (np.ndarray): number of detection and matches per stamp.
        meas_flux (np.ndarray): measured flux for each stamp.
        meas_fluxerr (np.ndarray): measured flux error for each stamp.
        unforced_meas_flux (np.ndarray): unforced measured flux for each stamp.
        unforced_meas_fluxerr (np.ndarray): unforced measured flux error for each stamp.
        shapes (np.ndarray): measured shape parameters for each stamp.
        kron_radius (np.ndarray): measured kron radius for each stamp.
    """

    # 2 dims for number of detections in the stamp and number of matches
    detected = np.zeros([nb, 2], dtype=np.int32)

    # also records fluxes, flux errors, shape parameters and kron radius
    meas_flux = np.zeros([nb])
    meas_flux_err = np.zeros([nb])
    unforced_meas_flux = np.zeros([nb])
    unforced_meas_flux_err = np.zeros([nb])
    shapes = np.zeros([nb, 5])
    kron_radius = np.zeros([nb, 2])

    gt_cat = fits.getdata(os.path.join(gt_cat_dir, "catalog.fits"))
    for coadd in tqdm.tqdm(range(nb_coadds), desc="coadds"):

        # get gt cat
        coadd_gt_cat = gt_cat[gt_cat["coadd_nb"] == coadd]

        # get meas cat
        with fits.open(
            os.path.join(
                rerun_dir,
                f"deepCoadd-results/HSC-{band.upper()}/{coadd}/0,0/forced_src-HSC-{band.upper()}-{coadd}-0,0.fits",
            )
        ) as fitscat:
            meas_cat = fitscat[1].data
        with fits.open(
            os.path.join(
                rerun_dir,
                f"deepCoadd-results/HSC-{band.upper()}/{coadd}/0,0/meas-HSC-{band.upper()}-{coadd}-0,0.fits",
            )
        ) as fitscat:
            unforced_meas_cat = fitscat[1].data

        # for each ground truth source in this coadd
        for gt_src in tqdm.tqdm(
            coadd_gt_cat, desc=f"gt src check in coadd {coadd} {band}"
        ):
            galsim_idx, x, y = gt_src["galsim_idx"], gt_src["x"], gt_src["y"]

            cond_x = np.logical_and(
                meas_cat["base_SdssCentroid_x"] < x + cell_size // 2,
                meas_cat["base_SdssCentroid_x"] > x - cell_size // 2,
            )
            cond_y = np.logical_and(
                meas_cat["base_SdssCentroid_y"] < y + cell_size // 2,
                meas_cat["base_SdssCentroid_y"] > y - cell_size // 2,
            )
            cond = np.logical_and(cond_x, cond_y)
            meas_to_check = meas_cat[cond]

            # check how many measured sources are matching it
            matched_meas = []
            dist_l = []
            for meas_src in meas_to_check:
                meas_x, meas_y = (
                    meas_src["base_SdssCentroid_x"],
                    meas_src["base_SdssCentroid_y"],
                )
                dist = (meas_x - x) ** 2 + (meas_y - y) ** 2

                if dist < 9:
                    matched_meas.append(meas_src)
                    dist_l.append(dist)

            if len(matched_meas):
                detected[galsim_idx, 0] = len(meas_to_check)
                detected[galsim_idx, 1] = len(matched_meas)
                meas_flux[galsim_idx] = matched_meas[dist_l.index(min(dist_l))][
                    "modelfit_CModel_instFlux"
                ]
                meas_flux_err[galsim_idx] = matched_meas[dist_l.index(min(dist_l))][
                    "modelfit_CModel_instFluxErr"
                ]
                pipe_id = matched_meas[dist_l.index(min(dist_l))]["id"]
                unforced_meas_flux[galsim_idx] = unforced_meas_cat[
                    unforced_meas_cat["id"] == pipe_id
                ]["modelfit_CModel_instFlux"]
                unforced_meas_flux_err[galsim_idx] = unforced_meas_cat[
                    unforced_meas_cat["id"] == pipe_id
                ]["modelfit_CModel_instFluxErr"]

                shape_xx = matched_meas[dist_l.index(min(dist_l))]["base_SdssShape_xx"]
                shape_yy = matched_meas[dist_l.index(min(dist_l))]["base_SdssShape_yy"]
                shape_xy = matched_meas[dist_l.index(min(dist_l))]["base_SdssShape_xy"]
                shape_x = matched_meas[dist_l.index(min(dist_l))]["base_SdssShape_x"]
                shape_y = matched_meas[dist_l.index(min(dist_l))]["base_SdssShape_y"]
                shapes[galsim_idx] = [shape_xx, shape_yy, shape_xy, shape_x, shape_y]

                kron1 = matched_meas[dist_l.index(min(dist_l))][
                    "ext_photometryKron_KronFlux_radius"
                ]
                kron2 = matched_meas[dist_l.index(min(dist_l))][
                    "ext_photometryKron_KronFlux_radius_for_radius"
                ]
                kron_radius[galsim_idx] = [kron1, kron2]

    return (
        detected,
        meas_flux,
        meas_flux_err,
        unforced_meas_flux,
        unforced_meas_flux_err,
        shapes,
        kron_radius,
    )


def make_magnitude_plots(
    detected, meas_flux, true_mags, band, color, exp_time, hsczp, plot_dir
):
    """Makes magnitude plots, i.e. measured versus true magnitudes.

    Plots are written in <plot_dir>.

    Args:
        detected (np.ndarray): number of detection and matches per stamp.
        meas_flux (np.ndarray): measured flux for each stamp.
        true_mags (np.ndarray): true magnitude for each stamp.
        band (string): band to process (used for plot labels only).
        color (string): color to use for plots.
        exp_time (int): exposure time to convert fluxes to magnitudes.
        hsczp (float): zeropoint to convert fluxes to magnitudes.
        plot_dir (string): output dir to write plots.
    """

    # get exactly detected objects
    m = np.logical_and(detected[:, 0] == 1, detected[:, 1] == 1)
    meas_flux = meas_flux[m]
    true_mags = true_mags[m]

    # get rid of possible nan measurement
    mnan = np.logical_not(np.isnan(meas_flux))
    print("nan", np.sum(np.logical_not(mnan)), len(meas_flux))
    meas_flux = meas_flux[mnan]
    true_mags = true_mags[mnan]

    # get rid of possible negative flux
    mneg = np.logical_not(meas_flux <= 0)
    print("neg", np.sum(np.logical_not(mneg)))
    meas_flux = meas_flux[mneg]
    true_mags = true_mags[mneg]

    # compute magnitudes
    meas_mags = hsczp - 2.5 * np.log10(meas_flux) + 2.5 * math.log10(exp_time)

    # plot
    mini = 20
    maxi = 27
    plt.plot(
        np.arange(mini, maxi + 1), np.arange(mini, maxi + 1), color="black", alpha=0.25
    )
    plt.plot(
        np.arange(mini, maxi + 1),
        np.arange(mini, maxi + 1) - 0.5,
        color="black",
        alpha=0.25,
    )
    plt.plot(
        np.arange(mini, maxi + 1),
        np.arange(mini, maxi + 1) + 0.5,
        color="black",
        alpha=0.25,
    )
    plt.hist2d(
        true_mags,
        meas_mags,
        bins=(int((maxi - mini) / 0.02), int((maxi - mini) / 0.02)),
        range=[[mini, maxi], [mini, maxi]],
        cmap=color,
    )
    plt.colorbar()
    plt.xlabel(f"True {band.upper()} magnitude", fontsize=14)
    plt.ylabel(f"Measured {band.upper()} magnitude", fontsize=14)
    # plt.title(f"{band.upper()} band magnitude measurement validation")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mag_{band}_validation.png"))
    plt.savefig(os.path.join(plot_dir, f"mag_{band}_validation.pdf"))
    plt.gcf().clear()


def make_detection_plots(detected, true_mags, band, color, plot_dir):
    """Makes detection plots, i.e. histograms of true magnitudes depending on detection results.

    This is mainly for curiosity and miss some cases.

    Args:
        detected (np.ndarray): number of detection and matches per stamp.
        true_mags (np.ndarray): true magnitude for each stamp.
        band (string): band to process (used for plot labels only).
        color (string): color to use for plots.
        plot_dir (string): output dir to write plots.
    """

    # no detection at all
    m = detected[:, 0] == 0
    if np.any(m == True):
        not_detected = true_mags[m]
        mini, maxi = np.min(not_detected), np.max(not_detected)
        n, bins, _ = plt.hist(
            not_detected,
            bins=np.arange(mini, maxi, 0.2),
            color=color,
            alpha=0.5,
            log=True,
        )
        plt.xlabel(f"True {band.upper()} magnitude")
        plt.ylabel("Counts")
        prop_not_detected = len(not_detected) / len(true_mags)
        plt.title(
            f"{band.upper()} band true magnitude histogram of undetected sources\nTotal: {len(not_detected)}. Proportion: {prop_not_detected:.4f}"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"maghist_{band}_undetected.png"))
        plt.savefig(os.path.join(plot_dir, f"maghist_{band}_undetected.pdf"))
        plt.gcf().clear()

    # one or more false detections (with or without the main source exactly detected)
    m = detected[:, 0] > 1
    if np.any(m == True):
        false_detected = true_mags[m]
        mini, maxi = np.min(false_detected), np.max(false_detected)
        n, bins, _ = plt.hist(
            false_detected,
            bins=np.arange(mini, maxi, 0.2),
            color=color,
            alpha=0.5,
            log=True,
        )
        plt.xlabel(f"True {band.upper()} magnitude")
        plt.ylabel("Counts")
        prop_false_detected = len(false_detected) / len(true_mags)
        plt.title(
            f"{band.upper()} band true magnitude histogram of false sources\nTotal: {len(false_detected)}. Proportion: {prop_false_detected:.4f}"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"maghist_{band}_false.png"))
        plt.savefig(os.path.join(plot_dir, f"maghist_{band}_false.pdf"))
        plt.gcf().clear()

    # exactly one good detection
    m = np.logical_and(detected[:, 0] == 1, detected[:, 1] == 1)
    if np.any(m == True):
        well_detected = true_mags[m]
        mini, maxi = np.min(well_detected), np.max(well_detected)
        n, bins, _ = plt.hist(
            well_detected,
            bins=np.arange(mini, maxi, 0.2),
            color=color,
            alpha=0.5,
            log=True,
        )
        plt.xlabel(f"True {band.upper()} magnitude")
        plt.ylabel("Counts")
        prop_well_detected = len(well_detected) / len(true_mags)
        plt.title(
            f"{band.upper()} band true magnitude histogram of exactly detected sources\nTotal: {len(well_detected)}. Proportion: {prop_well_detected:.4f}"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"maghist_{band}_exactlydetected.png"))
        plt.savefig(os.path.join(plot_dir, f"maghist_{band}_exactlydetected.pdf"))
        plt.gcf().clear()


def get_measurements(
    validation_cat_path, bands, rerun_dir, gt_cat_dir, galsim_cat_path
):
    """Gets the measurements for the validation plots.

    If the validation catalog already exists, just read the measurements from it, otherwise recovers everything and writes it to disk.
    This enables to iterate on plot modification faster.

    Args:
        validation_cat_path (string): path to the validation catalog.
        bands (list): list of bands to process.
        rerun_dir (string): rerun directory containing HSC pipeline output products.

    Returns:
        detected (np.ndarray): number of detection and matches per stamp.
        true_mags (np.ndarray): true magnitude for each stamp.
        meas_flux (np.ndarray): measured flux for each stamp.
        fully_undetected (np.ndarray): sources not detected in any band.
    """

    # get the source ident, later processing assumes sources are simulated in order
    # i.e that source at row k has the k-th ident
    with fits.open(galsim_cat_path) as fitscat:
        cat = fitscat[1].data
        ident = cat["IDENT"]

    # compute detection and measurements
    if not os.path.isfile(validation_cat_path):
        detected = np.zeros([len(bands), nb, 2])
        meas_flux = np.zeros([len(bands), nb])
        meas_flux_err = np.zeros([len(bands), nb])
        unforced_meas_flux = np.zeros([len(bands), nb])
        unforced_meas_flux_err = np.zeros([len(bands), nb])
        shapes = np.zeros([len(bands), nb, 5])
        radius = np.zeros([len(bands), nb, 2])
        for bidx, band in enumerate(bands):
            (
                detected[bidx],
                meas_flux[bidx],
                meas_flux_err[bidx],
                unforced_meas_flux[bidx],
                unforced_meas_flux_err[bidx],
                shapes[bidx],
                radius[bidx],
            ) = retrieve_det_meas(rerun_dir, gt_cat_dir, band)

        # write information in catalog so that we do not need to compute it again
        rows = []
        for k in range(nb):
            r = [ident[k]]
            for band in bands:
                r.append(detected[bands.index(band), k, 0])
                r.append(detected[bands.index(band), k, 1])
                r.append(meas_flux[bands.index(band), k])
                r.append(meas_flux_err[bands.index(band), k])
                r.append(unforced_meas_flux[bands.index(band), k])
                r.append(unforced_meas_flux_err[bands.index(band), k])
                for s in range(5):
                    r.append(shapes[bands.index(band), k, s])
                r.append(radius[bands.index(band), k, 0])
                r.append(radius[bands.index(band), k, 1])
            rows.append(r)

        # name and types of catalog columns
        col_names = ["IDENT"]
        col_types = ["i4"]
        for band in bands:
            col_names.append(f"{band}_total_det")
            col_types.append("i4")
            col_names.append(f"{band}_matched_det")
            col_types.append("i4")
            col_names.append(f"{band}_flux")
            col_types.append("f8")
            col_names.append(f"{band}_flux_err")
            col_types.append("f8")
            col_names.append(f"{band}_unforced_flux")
            col_types.append("f8")
            col_names.append(f"{band}_unforced_flux_err")
            col_types.append("f8")
            for i, s in enumerate(["xx", "yy", "xy", "x", "y"]):
                col_names.append(f"{band}_shape_{s}")
                col_types.append("f8")
            col_names.append(f"{band}_kron_radius")
            col_types.append("f8")
            col_names.append(f"{band}_kron_radius_for_radius")
            col_types.append("f8")

        # write it
        final_table = Table(rows=rows, names=col_names, dtype=col_types)
        final_table.write(validation_cat_path, overwrite=True)

    # retrieve detections and measurements from already written catalog
    else:
        with fits.open(validation_cat_path) as fitscat:
            cat = fitscat[1].data

        # retrieve detection and measurements
        detected = np.zeros([len(bands), nb, 2])
        meas_flux = np.zeros([len(bands), nb])
        for band in bands:
            detected[bands.index(band), :, 0] = cat[f"{band}_total_det"]
            detected[bands.index(band), :, 1] = cat[f"{band}_matched_det"]
            meas_flux[bands.index(band)] = cat[f"{band}_flux"]

    # retrieve true magnitudes
    true_mags = np.zeros([len(bands), nb])
    with fits.open(galsim_cat_path) as fitscat:
        cat = fitscat[1].data
        for band in bands:
            true_mags[bands.index(band)] = cat[f"HSC_{band}"]

    # retrieve fully undetected sources
    fully_undetected = np.sum(detected[:, :, 1], 0) == 0  # no match and possibly a fp

    return detected, true_mags, meas_flux, fully_undetected


def main():

    # data directory
    data_dir = "../../../data"

    # catalogs and directories
    base_suffix = "_release_run"
    suffix = "_release_run"
    galsim_cat_path = os.path.join(
        data_dir,
        f"galsim_cosmos_dataset/full{base_suffix}/real_galaxy_catalog_26_extension_fits.fits",
    )
    gt_cat_dir = os.path.join(
        data_dir, f"galsim_cosmos_dataset/validation_process/fake_coadds{suffix}"
    )
    rerun_dir = os.path.join(data_dir, f"my_data/rerun/validation_process{suffix}")

    # galcheat
    hsc_survey = galcheat.get_survey("HSC")
    filters = hsc_survey.available_filters

    # get detections and measurements
    validation_cat_path = os.path.join(
        data_dir,
        f"galsim_cosmos_dataset/validation_process/validation_process{suffix}.fits",
    )
    detected, true_mags, meas_flux, fully_undetected = get_measurements(
        validation_cat_path, filters, rerun_dir, gt_cat_dir, galsim_cat_path
    )

    # things for plots
    colors = ["purple", "cyan", "green", "orange", "red"]
    cmaps = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
    plot_dir = f"plots{suffix}"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    # make detection plots
    for i, filt in enumerate(filters):
        make_detection_plots(detected[i], true_mags[i], filt, colors[i], plot_dir)

    # make magnitude plots
    for i, filt in enumerate(filters):
        exp_time = hsc_survey.get_filter(filt).full_exposure_time.value
        zeropoint = hsc_survey.get_filter(filt).zeropoint.value
        make_magnitude_plots(
            detected[i],
            meas_flux[i],
            true_mags[i],
            filt,
            cmaps[i],
            exp_time,
            zeropoint,
            plot_dir,
        )

    # make refined catalogs without fully undetected sources
    with fits.open(galsim_cat_path) as fitscathdu:
        fitscat = fitscathdu[1].data

    # make detection selection
    nb = detected.shape[1]
    sel = np.zeros([nb], dtype=np.int32)
    for k in range(nb):
        # one match max in each band (avoid multi matches of the main object)
        # AND at least one band has one match (avoid cases with no detection at all)
        # AND one det max in the stamp (avoid good detection + fp cases)
        if (
            np.all(detected[:, k, 1] <= 1)
            and np.sum(detected[:, k, 1], axis=0)
            and np.all(detected[:, k, 0] <= 1)
        ):
            sel[k] = 1
    print(f"selection: {np.sum(sel)}")

    # save refined catalog
    idx = np.where(sel)
    newfitscat = Table(fitscat[idx[0]])
    newfitscat.write(
        galsim_cat_path.replace("_fits.fits", "_detrefined_fits.fits"), overwrite=True
    )
    with fits.open(galsim_cat_path.replace("_fits", "")) as cathdu:
        cat = cathdu[1].data
    newcat = Table(cat[idx[0]])
    newcat.write(
        galsim_cat_path.replace("_fits.fits", "_detrefined.fits"), overwrite=True
    )

    # make measurement selection ?


if __name__ == "__main__":
    main()
