import math
import os
import pickle

import galcheat
import numpy as np
import tqdm
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import linear_sum_assignment


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


def get_gt_data(filt, stamp_dir, stamp_idx, zp, exp_time, validation_cat):
    """Gets the ground truth data of a stamp.

    Args:
        - filt (string): filter to consider.
        - stamp_dir (string): directory containing the stamps.
        - stamp_idx (int): index of the stamp to consider.
        - zp (float): zeropoint of the given filter.
        - exp_time (float): exposure time of the given filter.
        - validation_cat (astropy.io.fits.fitsrec.FITS_rec): validation catalog to get isolated measurements of the sources.

    Returns:
        - nb_gt (int): number of groud truth objects in the stamp.
        - gt_coords (np.ndarray): ground truth coordinates of the objects.
        - gt_mags (np.ndarray): magnitudes of the objects.
        - iso_mags (np.ndarray): magnitudes of the objects measured when isolated.
        - iso_flux_err (np.ndarray): flux error of the objects measured when isolated.
    """

    # read stamp header
    stamp_header = fits.getheader(os.path.join(stamp_dir, f"{stamp_idx}_stamp.fits"), 1)

    # get number of gt objects
    try:
        _ = stamp_header["IDENT_1_1"]
        nb_gt = 2
    except KeyError:
        nb_gt = 1

    # retrieve coords and mags
    gt_coords = np.zeros([nb_gt, 2])
    gt_mags = np.zeros([nb_gt])
    for gt_obj in range(nb_gt):
        gt_coords[gt_obj, 0] = stamp_header[f"x_peak_{gt_obj}"]
        gt_coords[gt_obj, 1] = stamp_header[f"y_peak_{gt_obj}"]
        gt_mags[gt_obj] = stamp_header[f"HSC_{filt}_{gt_obj}"]

    # retrieve isolated measurements from validation process
    iso_mags = np.zeros([nb_gt])
    iso_flux_err = np.zeros([nb_gt])
    for gt_obj in range(nb_gt):
        gt_ident = stamp_header[f"IDENT_1_{gt_obj}"]
        validation_row = validation_cat[validation_cat["IDENT"] == gt_ident]
        iso_flux = validation_row[f"{filt}_flux"]
        if iso_flux > 0:
            iso_mags[gt_obj] = zp - 2.5 * math.log10(iso_flux / exp_time)
        else:
            iso_mags[gt_obj] = -1
        iso_flux_err[gt_obj] = validation_row[f"{filt}_flux_err"]

    return nb_gt, gt_coords, gt_mags, iso_mags, iso_flux_err


def get_pred_data(coadd_pred_cat, xb, yb, window, pix_stamp_size, zp, exp_time):
    """Gets the hscPipe predictions of a stamp.

    Args:
        - coadd_pred_cat (astropy.io.fits.fitsrec.FITS_rec): hscPipe given coadd catalog.
        - xb, yb (int, int): bottom left stamp position to look for in the coadd.
        - window (int): window to look for around the centered position.
        - pix_stamp_size (int): stamp size in pixels.
        - zp (float): zeropoint of the given filter.
        - exp_time (float): exposure time of the given filter.

    Returns:
        - nb_preds (int): number of predicted objects in the stamp by hscPipe.
        - pred_coords (np.ndarray): hscPipe predicted coordinates of the objects.
        - pred_mags (np.ndarray): hscPipe predicted magnitudes of the objects.
        - pred_ids (list): hscPipe ids of the objects.
        - pred_flux_err (np.ndarray): hscPipe predicted flux error of the objects.
    """

    # selection in coadd catalog
    xc, yc = xb + pix_stamp_size // 2, yb + pix_stamp_size // 2
    cond_x = np.logical_and(
        coadd_pred_cat["base_SdssCentroid_x"] < xc + window,
        coadd_pred_cat["base_SdssCentroid_x"] > xc - window,
    )
    cond_y = np.logical_and(
        coadd_pred_cat["base_SdssCentroid_y"] < yc + window,
        coadd_pred_cat["base_SdssCentroid_y"] > yc - window,
    )
    cond = np.logical_and(cond_x, cond_y)
    stamp_pred_cat = coadd_pred_cat[cond]
    stamp_pred_cat = stamp_pred_cat[stamp_pred_cat["deblend_nChild"] == 0]

    # retrieve coords and mags
    nb_pred = len(stamp_pred_cat)
    pred_coords = np.zeros([nb_pred, 2])
    pred_mags = np.zeros([nb_pred])
    pred_fluxerr = np.zeros([nb_pred])
    pred_ids = []
    for pred_obj in range(nb_pred):
        pred_coords[pred_obj] = (
            stamp_pred_cat[pred_obj]["base_SdssCentroid_x"] - xb,
            stamp_pred_cat[pred_obj]["base_SdssCentroid_y"] - yb,
        )
        pred_flux = stamp_pred_cat[pred_obj]["modelfit_CModel_instFlux"]
        if pred_flux > 0:
            pred_mags[pred_obj] = zp - 2.5 * math.log10(pred_flux / exp_time)
        else:
            pred_mags[pred_obj] = -1
        pred_fluxerr[pred_obj] = stamp_pred_cat[pred_obj]["modelfit_CModel_instFluxErr"]
        pred_ids.append(stamp_pred_cat[pred_obj]["id"])

    return nb_pred, pred_coords, pred_mags, pred_ids, pred_fluxerr


def make_det_matrix(gt_coords, pred_coords, matching_dist):
    """Computes detection matrix given ground truth and predicted coordinates.
    This matrix tells which predicted object is assigned to which ground truth object.

    Args:
        - gt_coords (np.ndarray): ground truth coordinates.
        - pred_coords (np.ndarray): predicted coordinates.
        - matching_dist (float): minimal distance for matching ground truth and predicted objects.
    Returns:
        - det_matrix (np.ndarray): detection matrix.
    """

    # compute assignment cost matrix
    cost_matrix = compute_cost_matrix(gt_coords, pred_coords)
    if cost_matrix.size == 0:
        det_matrix = None
    else:
        # solve the assignment problem
        gt_out, pred_out = linear_sum_assignment(cost_matrix)
        det_matrix = np.zeros_like(cost_matrix)
        for i, gt_idx in enumerate(gt_out):
            if cost_matrix[gt_idx, pred_out[i]] < matching_dist**2:
                det_matrix[gt_idx, pred_out[i]] = 1

        # manage the possible optimal assignments that would lead to no matching while one object is well detected
        # (this can happen when one predicted object is very far)
        if np.all(det_matrix == 0):
            for gt in range(cost_matrix.shape[0]):
                for pred in range(cost_matrix.shape[1]):
                    if cost_matrix[gt, pred] < matching_dist**2:
                        det_matrix[gt, pred] = 1

    return det_matrix


def compute_cost_matrix(gt_coords, pred_coords):
    """Computes to cost matrix assignment problem to make the detection matrix.

    Args:
        - gt_coords (np.ndarray): ground truth coordinates.
        - pred_coords (np.ndarray): predicted coordinates.

    Returns:
        - cost_matrix (np.ndarray): cost matrix.
    """

    nb_gt = len(gt_coords)
    nb_pred = len(pred_coords)
    cost_matrix = np.zeros([nb_gt, nb_pred])

    for gt in range(nb_gt):
        for pred in range(nb_pred):
            dist = (gt_coords[gt][0] - pred_coords[pred][0]) ** 2 + (
                gt_coords[gt][1] - pred_coords[pred][1]
            ) ** 2
            cost_matrix[gt, pred] = dist

    return cost_matrix


def main():

    # data directory
    data_dir = "../../data"
    out_dir = os.path.join(data_dir, "blending")
    suffix = "_release_run"

    # validation catalog to retrieve isolated source measurements
    validation_cat_path = os.path.join(
        data_dir,
        f"galsim_cosmos_dataset/validation_process/validation_process{suffix}.fits",
    )
    validation_cat = fits.getdata(validation_cat_path)

    # fake coadd characteristics
    margin = 150
    nx, ny = 40, 40
    pix_stamp_size = 95
    nb_stamps = 200000
    nb_coadds = 132
    star_map = make_star_map(nx, ny)
    nb_stamps_per_coadd = int(nx * ny - np.sum(star_map))

    # survey
    hsc_survey = galcheat.get_survey("HSC")
    filters = hsc_survey.available_filters

    # blending directories
    set_name = "_ms4"
    rerun_dir = os.path.join(data_dir, f"my_data/rerun/set_with_hst{set_name}")
    stamp_dir = os.path.join(data_dir, f"blending/set_with_hst{set_name}{suffix}")

    # mag cuts, taken from pdr2 depth
    # https://hsc-release.mtk.nao.ac.jp/doc/index.php/sample-page/pdr2/
    mag_cuts = [26.6, 26.2, 26.2, 25.3, 24.5]

    ### analysis

    # matching distance criterion in pixel
    matching_dist = 3

    # window to look for in each stamp
    window = 40

    # loop through each stamp across bands/coadds like in make_fake_coadds.py
    # this gets all the hscPipe results in each band independently
    # the resulting tmp_results dict has the given nested structure:
    #   keys: filter strings ("g", "r", "i", "z", "y")
    #   values: dict with keys: stamp index
    #                     values: all infos about ground truth and hscPipe predicted objects for the given filter and stamp
    tmp_results = dict()
    for filt in filters:

        # init band results
        tmp_results[filt] = dict()

        # get parameters for hscPipe magnitude computation
        cur_zp = hsc_survey.get_filter(filt).zeropoint.value
        cur_exp_time = hsc_survey.get_filter(filt).full_exposure_time.value

        for coadd in tqdm.tqdm(range(nb_coadds), desc=f"{filt.upper()} coadds"):

            # get coadd catalog
            coadd_pred_cat = fits.getdata(
                os.path.join(
                    rerun_dir,
                    f"deepCoadd-results/HSC-{filt.upper()}/{coadd}/0,0/forced_src-HSC-{filt.upper()}-{coadd}-0,0.fits",
                )
            )

            # init stamp indexes of this coadd
            stamp_idx = coadd * nb_stamps_per_coadd

            # for each stamp
            for y in tqdm.tqdm(range(ny), desc="Coadd rows"):
                for x in range(nx):

                    # stop when we reach the last stamp index
                    if stamp_idx == nb_stamps:
                        break

                    # consider the stamp if it's not a star
                    if not star_map[y, x]:

                        # get gt data
                        nb_gt, gt_coords, gt_mags, iso_mags, iso_flux_err = get_gt_data(
                            filt,
                            stamp_dir,
                            stamp_idx,
                            cur_zp,
                            cur_exp_time,
                            validation_cat,
                        )

                        # get hscPipe pred data
                        xb = x * pix_stamp_size + margin
                        yb = y * pix_stamp_size + margin
                        (
                            nb_pred,
                            pred_coords,
                            pred_mags,
                            pred_ids,
                            pred_flux_err,
                        ) = get_pred_data(
                            coadd_pred_cat,
                            xb,
                            yb,
                            window,
                            pix_stamp_size,
                            cur_zp,
                            cur_exp_time,
                        )

                        # make detection matrix
                        det_matrix = make_det_matrix(
                            gt_coords, pred_coords, matching_dist
                        )

                        # store stamp results
                        tmp_results[filt][stamp_idx] = {
                            "det_matrix": det_matrix,
                            "gt_mags": gt_mags,
                            "pred_mags": pred_mags,
                            "gt_coords": gt_coords,
                            "pred_coords": pred_coords,
                            "pred_ids": pred_ids,
                            "pred_flux_err": pred_flux_err,
                            "iso_mags": iso_mags,
                            "iso_flux_err": iso_flux_err,
                        }

                        # iterate
                        stamp_idx += 1

    # rearrange the hscPipe predictions by object ids
    # the rearranged rearranged_results dict has the following nested structure
    #  keys: stamp indices
    #  values: dict with keys: "pred_ids"
    #                          filter strings ("g", "r", "i", "z", "y")
    #                    values: list of hscPipe predicted object ids
    #                            dict with keys: ground truth infos
    #                                            hscPipe predicted object ids
    #                                      values: ground truth infos
    #                                              dict with keys: hscPipe predicted infos
    #                                                        values: hscPipe predicted infos
    rearranged_results = dict()
    for stamp_idx in tqdm.tqdm(range(nb_stamps), desc="Rearranging"):

        # get all the source ids across bands
        all_ids = []
        for filt in filters:
            filt_ids = tmp_results[filt][stamp_idx]["pred_ids"]
            for pred_id in filt_ids:
                if pred_id not in all_ids:
                    all_ids.append(pred_id)

        # build the new dict for this given stamp
        stamp_dict = dict()
        stamp_dict["pred_ids"] = all_ids

        for filt in filters:
            filt_dict = dict()

            filt_dict["det_matrix"] = tmp_results[filt][stamp_idx]["det_matrix"]
            filt_dict["gt_coords"] = tmp_results[filt][stamp_idx]["gt_coords"]
            filt_dict["gt_mags"] = tmp_results[filt][stamp_idx]["gt_mags"]
            filt_dict["iso_mags"] = tmp_results[filt][stamp_idx]["iso_mags"]
            filt_dict["iso_flux_err"] = tmp_results[filt][stamp_idx]["iso_flux_err"]

            for pred_id in all_ids:
                pred_dict = dict()
                try:
                    idx = tmp_results[filt][stamp_idx]["pred_ids"].index(pred_id)
                    pred_dict["pred_coords"] = tmp_results[filt][stamp_idx][
                        "pred_coords"
                    ][idx]
                    pred_dict["pred_mag"] = tmp_results[filt][stamp_idx]["pred_mags"][
                        idx
                    ]
                    pred_dict["pred_flux_err"] = tmp_results[filt][stamp_idx][
                        "pred_flux_err"
                    ][idx]
                except:
                    pred_dict["pred_coords"] = -1
                    pred_dict["pred_mag"] = -1
                    pred_dict["pred_flux_err"] = -1

                filt_dict[pred_id] = pred_dict

            stamp_dict[filt] = filt_dict

        rearranged_results[stamp_idx] = stamp_dict

    # make a second loop to check for bad objects and remove them
    # this removes predicted objects with bad magnitudes by hscPipe (i.e. negative or above some cuts)
    for stamp_idx in tqdm.tqdm(range(nb_stamps), desc="Cleaning"):

        # make a list of bad objects in each band
        all_ids = rearranged_results[stamp_idx]["pred_ids"]
        to_remove = [0] * len(all_ids)
        for i, pred_id in enumerate(all_ids):
            for j, filt in enumerate(filters):
                mag = rearranged_results[stamp_idx][filt][pred_id]["pred_mag"]
                if mag > mag_cuts[j] or mag == -1:
                    to_remove[i] += 1

        # remove objects that have negative or to high magnitude in at least 3 bands
        have_removed = False
        for i, pred_id in enumerate(all_ids):
            if to_remove[i] >= 3:
                have_removed = True
                for filt in filters:
                    if all_ids[i] in rearranged_results[stamp_idx][filt].keys():
                        del rearranged_results[stamp_idx][filt][all_ids[i]]
                rearranged_results[stamp_idx]["pred_ids"].remove(all_ids[i])

        # recompute the detection matrix in each filter for this stamp now that some hscPipe predicted objects have been removed
        if have_removed:
            for filt in filters:
                gt_coords = rearranged_results[stamp_idx][filt]["gt_coords"]

                pred_coords = []
                for pred_id in rearranged_results[stamp_idx]["pred_ids"]:
                    coords = rearranged_results[stamp_idx][filt][pred_id]["pred_coords"]
                    if isinstance(coords, np.ndarray):
                        pred_coords.append(coords)

                det_matrix = make_det_matrix(gt_coords, pred_coords, matching_dist)

                rearranged_results[stamp_idx][filt]["det_matrix"] = det_matrix

    # save results
    pickle_path = os.path.join(
        out_dir, f"set_with_hst{set_name}{suffix}_pp_analysis.pickle"
    )
    with open(pickle_path, "wb") as pf:
        pickle.dump(rearranged_results, pf)


if __name__ == "__main__":
    main()
