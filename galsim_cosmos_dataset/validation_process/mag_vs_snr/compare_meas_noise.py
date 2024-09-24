import math
import os
import pickle
import random as rd
import sys

import galsim
import matplotlib as mpl
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from scipy import stats
from scipy.spatial import KDTree

mpl.use("Agg")
import lsst.daf.persistence as dafPersist
import matplotlib.pyplot as plt
from lsst.geom import Angle, Point2D, SpherePoint, degrees, radians
from scipy import asarray as ar
from scipy import exp
from scipy.optimize import curve_fit

eps = 1e-08


def gauss(x, a, x0, sigma):
    """Gaussian function.

    Args:
        x (float or np.ndarray): input to the function.
        a (float or np.ndarray): amplitude of the Gaussian.
        x0 (float or np.ndarray): mean of the Gaussian.
        sigma (float or np.ndarray): sigma of the Gaussian.

    Returns:
        (float or np.ndarray): corresponding Gaussian results.
    """

    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def gaussfit(x, y, init):
    """Gaussian fit function.

    Args:
        x, y (np.ndarray, np.ndarray): data to use for fitting.
        init (list): initial guess of the Gaussian parameters (a, x0, sigma).

    Returns:
        popt (list): Gaussian parameters fit on (x, y).
    """

    popt, pcov = curve_fit(gauss, x, y, p0=init)
    return popt


def main():

    # data directories
    suffix = "_release_run"
    tract = 9813
    data_dir = "../../../data"

    # data to compute magnitudes (retrieve from galcheat ?)
    bands = ["g", "r", "i", "z", "y"]
    hsczp = [28.90, 28.86, 28.61, 27.68, 27.33]
    exp_time = [600, 600, 1200, 1200, 1200]

    # second galsim cosmos catalog to get EL COSMOS true magnitudes
    with fits.open(
        os.path.join(
            data_dir,
            f"galsim_cosmos_dataset/full{suffix}/real_galaxy_catalog_26_extension_fits.fits",
        )
    ) as fitscat:
        galsim_fitcatalog = fitscat[1].data

    # BTK detection and measurement validation process catalog
    with fits.open(
        os.path.join(
            data_dir,
            f"galsim_cosmos_dataset/validation_process/validation_process{suffix}.fits",
        )
    ) as fitscat:
        btk_meas_catalog = fitscat[1].data

    # a small parameter to possibly add features to the plots
    no_sig = True
    if no_sig:
        out_dir = f"meas_noise_plots{suffix}_nosig"
    else:
        out_dir = f"meas_noise_plots{suffix}"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    patch_dir = os.path.join(
        data_dir,
        f"hsc_data/rerun/pdr2_wide_magsnr_inj{suffix}_2/deepCoadd-results/HSC-I/9813",
    )
    patches = os.listdir(patch_dir)

    print(f"Recovering injected measurements")
    print("Lost sources are due to non detection or invalid measurement\n")
    tot_sources = np.zeros([len(bands)], dtype=np.int32)

    band_list = []

    # for each band
    for i_band, band in enumerate(bands):
        print(f"Band {band}")
        recovered_fluxes = dict()
        recovered_flux_errors = dict()

        # go through all the injected sources of the given band in all patches
        for patch_s in patches:

            # ground truth catalog of injected sources in that patch
            inj_gt_cat = fits.getdata(
                os.path.join(
                    data_dir,
                    f"for_magsnr_source_injection{suffix}/{tract}_{patch_s}.fits",
                )
            )
            tot_sources[i_band] += len(inj_gt_cat)

            # measurement catalog of that patch
            inj_meas_cat = fits.getdata(
                os.path.join(
                    data_dir,
                    f"hsc_data/rerun/pdr2_wide_magsnr_inj{suffix}_2/deepCoadd-results/HSC-{band.upper()}/{tract}/{patch_s}/meas-HSC-{band.upper()}-{tract}-{patch_s}.fits",
                )
            )

            # build tree of injected measurement sources
            ra = inj_meas_cat["coord_ra"]
            dec = inj_meas_cat["coord_dec"]
            phi = ra
            theta = np.pi / 2 - dec
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            data = np.array([x, y, z]).transpose()
            tree_inj_meas_hsc = KDTree(data)

            # set a distance
            r_arcsec = 0.34  # to relate to HSC pix size ~0.17
            r_deg = r_arcsec / 3600.0
            r_rad = r_deg * 2 * np.pi / 360.0
            dist = r_rad

            # find the measurement of each injected source
            c = 0
            for src in inj_gt_cat:
                im_path = src[f"{band}imFilename"]
                galsim_id = int(im_path.split("/")[-1].replace(".fits", ""))

                # get source candidates
                ra, dec = src["ra"], src["dec"]
                phi = ra
                theta = np.pi / 2 - dec
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                data = np.array([x, y, z]).transpose()
                sources = tree_inj_meas_hsc.query_ball_point(data, dist)

                # if there is one source we directly assume it's the good one
                if len(sources) == 1:
                    meas_flux = inj_meas_cat[sources[0]]["modelfit_CModel_instFlux"]
                    if meas_flux > 0:
                        recovered_fluxes[galsim_id] = meas_flux
                        recovered_flux_errors[galsim_id] = inj_meas_cat[sources[0]][
                            "modelfit_CModel_instFluxErr"
                        ]
                # sometimes there are two 'identical' matches and we recover only one of them
                elif len(sources) == 2:
                    for k in range(2):
                        if inj_meas_cat[sources[k]]["deblend_nChild"] == 0:
                            meas_flux = inj_meas_cat[sources[k]][
                                "modelfit_CModel_instFlux"
                            ]
                            if meas_flux > 0:
                                recovered_fluxes[galsim_id] = meas_flux
                                recovered_flux_errors[galsim_id] = inj_meas_cat[
                                    sources[k]
                                ]["modelfit_CModel_instFluxErr"]

        band_list.append([recovered_fluxes, recovered_flux_errors])
        print(
            f"Recovered {len(recovered_fluxes)} injected sources out of {tot_sources[i_band]}\n"
        )

    # now get the BTK validation process stamp measurement of each injected source we recovered
    print(
        f"Matching recovered injected source measurements with their (hscPipe) BTK stamp measurements"
    )
    print(
        "Lost sources are due to non detection in BTK stamps or invalid BTK stamp measurement\n"
    )

    # for each band
    for i_band, band in enumerate(bands):
        print(f"Band {band}")

        btk_mags = []
        btk_fluxes = []
        btk_errors = []

        inj_mags = []
        inj_fluxes = []
        inj_errors = []

        recovered_fluxes = band_list[i_band][0]
        recovered_flux_errors = band_list[i_band][1]
        tot_sources = band_list[i_band]

        nb_bad_btk = 0
        for k in recovered_fluxes.keys():
            if btk_meas_catalog[k][f"{band}_unforced_flux"] > 0:
                inj_mags.append(27 - 2.5 * math.log10(recovered_fluxes[k]))
                inj_fluxes.append(recovered_fluxes[k])
                inj_errors.append(recovered_flux_errors[k])

                btk_mags.append(
                    hsczp[i_band]
                    - 2.5
                    * math.log10(
                        btk_meas_catalog[k][f"{band}_unforced_flux"] / exp_time[i_band]
                    )
                )
                btk_fluxes.append(btk_meas_catalog[k][f"{band}_unforced_flux"])
                btk_errors.append(btk_meas_catalog[k][f"{band}_unforced_flux_err"])
            else:
                nb_bad_btk += 1
        print(
            f"Matched {len(inj_mags)} sources out of {len(recovered_fluxes)} injected sources recovered (out of {len(tot_sources)})\n"
        )

        # mag snr plots
        inj_snrs = np.array(inj_fluxes) / np.array(inj_errors)
        btk_snrs = np.array(btk_fluxes) / np.array(btk_errors)
        inj_mags = np.array(inj_mags)
        btk_mags = np.array(btk_mags)

        step = 10
        new_x, new_y, y_err = [], [], []
        for k in range(0, 200, step):
            idx = np.where(np.logical_and(inj_snrs > k, inj_snrs < k + step))

            if len(idx[0]) > 2:
                new_x.append(k + step / 2)
                new_y.append(np.mean(inj_mags[idx[0]]))
                y_err.append(np.std(inj_mags[idx[0]]))

        plt.errorbar(
            new_x,
            new_y,
            yerr=y_err,
            fmt="o",
            label="From real coadd injection",
            linestyle="-",
        )

        new_x, new_y, y_err = [], [], []
        for k in range(0, 200, step):
            idx = np.where(np.logical_and(btk_snrs > k, btk_snrs < k + step))

            if len(idx[0]) > 2:
                new_x.append(k + step / 2 + 1)
                new_y.append(np.mean(btk_mags[idx[0]]))
                y_err.append(np.std(btk_mags[idx[0]]))

        plt.errorbar(
            new_x,
            new_y,
            yerr=y_err,
            fmt="o",
            label="From BTK simulations",
            linestyle="-",
        )

        plt.legend(fontsize=12)
        plt.xlabel(f"{band.upper()} SNR", fontsize=14)
        plt.ylabel(f"{band.upper()} Magnitude", fontsize=14)
        # plt.title(f"{band.upper()} band")
        plt.savefig(os.path.join(out_dir, f"{band}_mag_snr_errbar.png"))
        plt.gcf().clear()


if __name__ == "__main__":
    main()
