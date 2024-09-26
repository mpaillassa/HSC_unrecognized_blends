import os
import pickle
import random as rd
import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from astropy.io import fits
from matplotlib import gridspec


def single_classification(pp_results, bands):
    """Classifies hscPipe preprocessed results of single object images from the results computed in preprocess_hscpipe_meas.py.

    Args:
        - pp_results (dict): dictionary of the preprocessed results computed in preprocess_hscpipe_meas.py.
        - bands (list): list of bands.

    Returns:
        - single_res (dict): dictionary having the classification cases as keys and the stamp indices as values.
    """

    # classification dict
    single_cl = {
        "0": [],  # no detection at all
        "1": [],  # single object matched
        "0_fp": [],  # single object not matched and object(s) detected elsewhere
        "1_fp": [],  # single object matched and object(s) detected elsewhere
        "2_fp": [],  # single object matched with at least 2 objects
    }

    # for each stamp
    for stamp_key in tqdm.tqdm(pp_results.keys(), desc="Single classification"):

        # get stamp res
        stamp_res = pp_results[stamp_key]

        # get number of gt objects
        gt_mags = stamp_res["i"]["gt_mags"]
        nb_gt = len(gt_mags)

        # if there is one gt object
        if nb_gt == 1:

            # check if there are detections
            no_det = True
            for band in bands:
                if stamp_res[band]["det_matrix"] is None:
                    pass
                else:
                    no_det = False

            # if there is no detection
            if no_det:
                single_cl["0"].append(stamp_key)

            # else there is at least one predicted object
            else:

                # check each predicted object
                pred_class = dict()
                for obj_id in stamp_res["pred_ids"]:

                    gt_matches = dict()
                    for band in bands:
                        det_matrix = stamp_res[band]["det_matrix"]
                        obj_idx = list(stamp_res[band].keys())[5:].index(
                            obj_id
                        )  # assumes the order of keys and the order in the detection matrix are the same
                        if stamp_res[band][obj_id]["pred_mag"] != -1:
                            if np.any(det_matrix[:, obj_idx]):
                                gt_idx = np.argmax(det_matrix[:, obj_idx])
                                if gt_idx not in gt_matches.keys():
                                    gt_matches[gt_idx] = 1
                                else:
                                    gt_matches[gt_idx] += 1

                    gt_found = gt_matches.keys()
                    if len(gt_found) == 0:  # this pred did not match any gt
                        pred_class[obj_id] = "fp"
                    elif (
                        len(gt_found) == 1
                    ):  # this pred matched the gt in at least one band
                        pred_class[obj_id] = "tp"

                # assign a label to the stamp
                pred_values = list(pred_class.values())
                nb_tp = pred_values.count("tp")
                nb_fp = pred_values.count("fp")
                if nb_tp == 1 and nb_fp == 0:
                    single_cl["1"].append(stamp_key)
                elif nb_tp == 0 and nb_fp >= 1:
                    single_cl["0_fp"].append(stamp_key)
                elif nb_tp == 1 and nb_fp >= 1:
                    single_cl["1_fp"].append(stamp_key)
                elif nb_tp >= 2:
                    single_cl["2_fp"].append(stamp_key)

    return single_cl


def blend_classification(pp_results, bands):
    """Classifies hscPipe preprocessed results of blend object images from the results computed in preprocess_hscpipe_meas.py.

    Args:
        - pp_results (dict): result dictionary computed in preprocess_hscpipe_meas.py.
        - bands (list): list of bands.

    Returns:
        - blend_cl (dict): dictionary having the classification cases as keys and the stamp indices as values.
    """

    # classification dict
    blend_cl = {
        "0": [],  # no detection at all
        "1": [],  # one object matched
        "0_fp": [],  # no object matched and object(s) detected elsewhere
        "1_fp": [],  # one object matched and object(s) detected elsewhere
        "2": [],  # both objects matched
        "2_fp": [],  # both objects matched and object(s) detected elsewhere
        "cf": [],  # confusion across bands, i.e. one predicted object is matched to different ground truth objects in different bands
    }

    shorter_det_matrix_fails = []

    # for each stamp
    for stamp_key in tqdm.tqdm(pp_results.keys(), desc="Blend classification"):

        # get stamp res
        stamp_res = pp_results[stamp_key]

        # get number of gt objects
        gt_mags = stamp_res["i"]["gt_mags"]
        nb_gt = len(gt_mags)

        # if there is one gt object
        if nb_gt == 2:

            # check if there are detections
            no_det = True
            for band in bands:
                if stamp_res[band]["det_matrix"] is None:
                    pass
                else:
                    no_det = False

            # if there is no detection
            if no_det:
                blend_cl["0"].append(stamp_key)

            # else there is at least one predicted object
            else:

                # check each predicted object
                pred_class = dict()
                for obj_id in stamp_res["pred_ids"]:

                    gt_matches = dict()
                    for band in bands:
                        det_matrix = stamp_res[band]["det_matrix"]
                        obj_idx = list(stamp_res[band].keys())[5:].index(
                            obj_id
                        )  # assumes the order of keys and the order in the detection matrix are the same
                        if stamp_res[band][obj_id]["pred_mag"] != -1:
                            if obj_idx >= det_matrix.shape[1]:
                                shorter_det_matrix_fails.append(
                                    [stamp_key, band, obj_idx, det_matrix.shape[1]]
                                )
                            else:
                                if np.any(det_matrix[:, obj_idx]):
                                    gt_idx = np.argmax(det_matrix[:, obj_idx])
                                    if gt_idx not in gt_matches.keys():
                                        gt_matches[gt_idx] = 1
                                    else:
                                        gt_matches[gt_idx] += 1

                    gt_found = gt_matches.keys()
                    if len(gt_found) == 0:  # this pred did not match any gt
                        pred_class[obj_id] = "fp"
                    elif (
                        len(gt_found) == 1
                    ):  # this pred matched the gt in at least one band
                        pred_class[obj_id] = "tp"
                    elif (
                        len(gt_found) == 2
                    ):  # this pred matched two different gts across bands
                        pred_class[obj_id] = "confused"

                # assign a label to the stamp
                pred_values = list(pred_class.values())
                nb_tp = pred_values.count("tp")
                nb_fp = pred_values.count("fp")
                nb_cf = pred_values.count("confused")
                if nb_cf:
                    blend_cl["cf"].append(stamp_key)
                else:
                    if nb_tp == 2 and nb_fp == 0:
                        blend_cl["2"].append(stamp_key)
                    if nb_tp == 2 and nb_fp >= 1:
                        blend_cl["2_fp"].append(stamp_key)
                    if nb_tp == 1 and nb_fp == 0:
                        blend_cl["1"].append(stamp_key)
                    if nb_tp == 1 and nb_fp >= 1:
                        blend_cl["1_fp"].append(stamp_key)
                    if nb_tp == 0 and nb_fp >= 1:
                        blend_cl["0_fp"].append(stamp_key)

    print("Too short detetection matrix on sample(s): ", shorter_det_matrix_fails)

    return blend_cl


def make_figures(
    nb_fig,
    bands,
    pp_results,
    cl_results,
    stamp_dir,
    set_name,
    stamp_type,
    with_table,
    cnn_res,
):
    """Makes figures for all classification cases of single or blend stamps. Each figure is a randomly picked stamp sample and includes images with coordinate annotations, hscPipe result information table (optionally) and CNN blend probability predictions (optionally).

    Args:
        - nb_fig (int): number of figures to make.
        - bands (list): list of bands.
        - pp_results (dict): dictionary of results from preprocess_hscpipe_meas.py.
        - cl_results (dict): dictionary of classification results computed by single_classification or blend_classification.
        - stamp_dir (string): directory containing the stamps.
        - set_name (string): name of the set for the output directory.
        - stamp_type (string): type of stamps to make figures for (single or blend).
        - with_table (bool): bool to decide if adding the table in the figure.
        - cnn_res (np.ndarray or None): blend probabilities from CNN.
    """

    row_size = 25
    fig = plt.figure(figsize=(row_size, row_size))

    # for each classification case
    for k in cl_results.keys():

        out_dir = f"figures/set{set_name}/{stamp_type}/{k}"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # pick a random sample of cases
        if len(cl_results[k]) < nb_fig:
            nb_fig = len(cl_results[k])
        res_samples = rd.sample(cl_results[k], nb_fig)

        for stamp_idx in res_samples:

            # make an rgb image
            im_path = os.path.join(stamp_dir, f"{stamp_idx}_stamp.fits")
            full_im = fits.getdata(im_path)
            rgb_im, rgb_bands = make_rgb_im(full_im)
            im_size = rgb_im.shape[0]

            # prepare figure layout
            if with_table:
                spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 1])
                ax1, ax2 = fig.add_subplot(spec[0]), fig.add_subplot(spec[1])
                ax1.imshow(rgb_im)
            else:
                ax1 = plt
                ax1.imshow(rgb_im)

            # add coords annotations
            add_coord_annotations(ax1, pp_results, stamp_idx, im_size)
            ax1.axis("off")

            # if table requested
            if with_table:

                # make the table
                data_table = make_table(pp_results, stamp_idx, bands)

                # make the column labels of the table
                col_lab = ["Quantity / Band"]
                for band in bands:
                    col_lab.append(band)

                # add colors to the column labels to specify the bands picked for rgb channels
                rgb_colors = ["red", "green", "blue"]
                colors = ["white"]
                idx = 0
                for band_idx, band in enumerate(bands):
                    if (
                        band_idx == rgb_bands[idx]
                    ):  # assumes rgb_bands are in ascending order
                        band_col = rgb_colors[idx]
                        idx += 1
                    else:
                        band_col = "white"
                    colors.append(band_col)

                # add the table to the plot
                table = ax2.table(
                    cellText=data_table,
                    colLabels=col_lab,
                    cellLoc="center",
                    loc="center",
                    colColours=colors,
                )
                table.auto_set_font_size(False)
                table.set_fontsize(30)
                table.scale(1.5, 2.2)
                ax2.axis("off")

                # add cnn probability result if requested
                if cnn_res is not None:
                    ax2.annotate(
                        f"Latest CNN prediction: {cnn_res[stamp_idx]:.2f} probability to be a blend",
                        (0.01, 0.97),
                        fontsize="medium",
                    )

            fig.tight_layout()
            fig_path = os.path.join(out_dir, f"{stamp_idx}.png")
            fig.canvas.start_event_loop(sys.float_info.min)
            plt.savefig(fig_path, bbox_inches="tight")
            plt.gcf().clear()


def make_rgb_im(full_im, selected_bands=[0, 2, 4]):
    """Makes an array ready to be plot as rgb from a grizy fits image.

    Args:
        - full_im (np.ndarray): image to make rgb.
        - bands (list): list of indices of bands to select for rgb channels. Default is 0,2,4 for g,i,y.
    Returns:
        - res_im (np.ndarray): resulting rgb ready image.
        - bands (list): list of indices of bands selected to make the rgb channels.
    """

    res_im = full_im[selected_bands, :, :]
    for ch in range(len(selected_bands)):
        res_im[ch] = np.flipud(normalization(res_im[ch]))

    res_im = np.transpose(res_im, (1, 2, 0)).astype(np.uint8)

    return res_im, selected_bands


def normalization(x):
    """Normalizes the input array for rgb format (0, 255).

    Args:
        - x (np.ndarray): input array to normalize.
    Returns:
        - x_norm (np.ndarray): normalized array.
    """

    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = 255 * (x - x_min) / (x_max - x_min)

    return x_norm


def add_coord_annotations(
    ax1,
    pp_results,
    stamp_idx,
    im_size,
    fontsize=50,
    default_band="i",
    gt_color="black",
    pred_color="red",
):
    """Adds HscPipe predicted and ground truth object coordinate annotations in the image figure.

    Args:
        - ax1 (matplotlib.axes.Axes): axes of the matplotlib figure.
        - pp_results (dict): dictionary of results from preprocess_hscpipe_meas.py.
        - stamp_idx (int): index of the stamp.
        - im_size (int): image size.
        - fontsize (int): font size to use for annotations
        - default_band (string): band to select coordinates to display. Default is i band.
        - gt_color (string): color for ground truth coordinate annotations.
        - pred_color (string): color for HscPipe predicted coordinate annotations.
    """

    gt_coords = pp_results[stamp_idx][default_band]["gt_coords"]
    k = 1
    for gt in gt_coords:
        ax1.plot(gt[0], im_size - gt[1], "+", color=gt_color, mew=5, ms=30)
        ax1.annotate(
            str(k),
            xy=(gt[0] - 4, im_size - gt[1]),
            color=gt_color,
            weight="bold",
            fontsize=fontsize,
        )
        k += 1

    pred_coords = []
    pred_ids = list(pp_results[stamp_idx]["i"].keys())[5:]
    if len(pred_ids):
        for pred_id in pred_ids:
            pred_coords.append(
                pp_results[stamp_idx][default_band][pred_id]["pred_coords"]
            )
        k = 1
        for pred in pred_coords:
            if not isinstance(pred, int):
                ax1.plot(
                    pred[0], im_size - pred[1], "x", color=pred_color, mew=5, ms=30
                )
                ax1.annotate(
                    str(k),
                    xy=(pred[0] + 2, im_size - pred[1]),
                    color=pred_color,
                    weight="bold",
                    fontsize=fontsize,
                )
            k += 1


def make_table(pp_results, stamp_idx, bands, default_band="i"):
    """Makes the information table for the figure.

    Args:
        - pp_results (dict): dictionary of results from preprocess_hscpipe_meas.py.
        - stamp_idx (int): index of the stamp.
        - bands (list): list of bands.
        - default_band (string): band to select coordinates to display. Default is i band.
    """

    # number of gt and pred
    nb_gt = len(pp_results[stamp_idx][default_band]["gt_coords"])
    nb_pred = len(pp_results[stamp_idx]["pred_ids"])

    # empty table
    # 3 times nb_gt because we display gt coords, gt mags, det matrix
    # 2 times nb_pred because we display pred coords, pred mags
    data_table = []
    for i in range(3 * nb_gt + 2 * nb_pred):
        row = [None] * (1 + len(bands))
        data_table.append(row)

    # now fill the rows with add_quantities_in_table(...) in the following order: gt coords, pred coords, gt mags, pred mags, det matrix
    # using the same function for gt, preds and different quantities may be confusing due to different formats and cases...

    # gt coords
    offset = 0
    add_quantities_in_table(
        data_table, "gt_coords", nb_gt, offset, bands, pp_results, stamp_idx, None, None
    )
    offset += nb_gt

    # pred coords
    pred_idx = 1
    for pred_id in pp_results[stamp_idx]["pred_ids"]:
        add_quantities_in_table(
            data_table,
            "pred_coords",
            1,
            offset,
            bands,
            pp_results,
            stamp_idx,
            pred_id,
            pred_idx,
        )
        offset += 1
        pred_idx += 1

    # gt mags
    add_quantities_in_table(
        data_table, "gt_mags", nb_gt, offset, bands, pp_results, stamp_idx, None, None
    )
    offset += nb_gt

    # pred mags
    pred_idx = 1
    for pred_id in pp_results[stamp_idx]["pred_ids"]:
        add_quantities_in_table(
            data_table,
            "pred_mag",
            1,
            offset,
            bands,
            pp_results,
            stamp_idx,
            pred_id,
            pred_idx,
        )
        offset += 1
        pred_idx += 1

    # det matrix
    for i in range(nb_gt):
        data_table[i + offset][0] = f"det_matrix{i+1}"

        for j, band in enumerate(bands):
            det_matrix = pp_results[stamp_idx][band]["det_matrix"]
            if det_matrix is not None:
                data_table[i + offset][1 + j] = det_matrix[i]
            else:
                data_table[i + offset][1 + j] = "None"

    return data_table


def add_quantities_in_table(
    data_table, key, nb, offset, bands, pp_results, stamp_idx, pred_id, pred_no
):
    """Adds quantities to the data table. This modifies the data table on-the-fly.

    Args:
        - data_table (list of list): data table.
        - key (string): name of the quantity.
        - nb (int): number of quantities to add.
        - offset (int): row offset
        - bands (list): bands to process.
        - pp_results (): result dictionary computed in preprocess_hscpipe_meas.py.
        - stamp_idx (int): index of the stamp.
        - pred_id (int or None): hscPipe predicted id of the object. None if the quantity to add is not related to hscPipe predicted object.
        - pred_no (int or None): number of the hscPipe predicted object. None if the quantity to add is not related to hscPipe predicted object.
    """

    # for each quantity to add
    for q_nb in range(nb):

        # add data table row names
        if (
            pred_no is None
        ):  # this is gt quantity and we add it for all objects at once using the number of quantities
            data_table[q_nb + offset][0] = f"{key}{q_nb+1}"
        else:  # this is pred quantity and we add it for each object one by one using pred_id and pred_no
            data_table[q_nb + offset][0] = f"{key}{pred_no}"

        # add data table row values
        for band_idx, band in enumerate(bands):
            if (
                pred_id is None
            ):  # this is gt quantity, always comes for all objects at once and as np.ndarray even for single values
                vals = pp_results[stamp_idx][band][key]
                rounded_vals = round_values(vals)
                for val_nb, rounded_val in enumerate(rounded_vals):
                    data_table[val_nb + offset][1 + band_idx] = rounded_val
            else:  # this is pred quantity, always comes for one object at a time but not always as np.ndarray...
                vals = pp_results[stamp_idx][band][pred_id][key]
                if isinstance(vals, np.ndarray):
                    rounded_vals = round_values(vals)
                    data_table[q_nb + offset][1 + band_idx] = rounded_vals
                else:
                    data_table[q_nb + offset][1 + band_idx] = round(vals, 2)


def round_values(vals, preci=2):
    """Rounds values to a specified precision.

    Args:
        - vals (np.ndarray): input values to round.
        - preci (int): number of digits for rounding. Default is 2.
    Returns:
        - rounded_vals (np.ndarray): same size array than vals with rounded values.
    """

    rounded_vals = np.zeros_like(vals)

    if len(rounded_vals.shape) == 2:
        for i in range(rounded_vals.shape[0]):
            for j in range(rounded_vals.shape[1]):
                rounded_vals[i, j] = round(vals[i, j], preci)
    elif len(rounded_vals.shape) == 1:
        for i in range(rounded_vals.shape[0]):
            rounded_vals[i] = round(vals[i], preci)

    return rounded_vals


def main():

    # data directory
    data_dir = "../../data"

    # blending directories
    suffix = "_release_run"
    set_name = "_ms8"
    blend_dir = os.path.join(data_dir, "blending")
    stamp_dir = os.path.join(blend_dir, f"set_with_hst{set_name}{suffix}")

    # preprocessed results with preprocess_hscpipe_meas.py
    pp_results_pickle_path = os.path.join(
        blend_dir, f"set_with_hst{set_name}{suffix}_pp_analysis.pickle"
    )
    with open(pp_results_pickle_path, "rb") as pf:
        pp_results = pickle.load(pf)

    # bands to consider
    bands = ["g", "r", "i", "z", "y"]

    # make the case classification of each single stamp and some figures
    single_cl_pickle_path = os.path.join(
        blend_dir, f"set_with_hst{set_name}{suffix}_single_classification.pickle"
    )
    if not os.path.isfile(single_cl_pickle_path):
        single_cl = single_classification(pp_results, bands)
        with open(single_cl_pickle_path, "wb") as pf:
            pickle.dump(single_cl, pf)
    else:
        with open(single_cl_pickle_path, "rb") as pf:
            single_cl = pickle.load(pf)

    for k in single_cl.keys():
        print(k, len(single_cl[k]))

    # paper figure
    paper_cl = {"1": [59881]}
    make_figures(
        1,
        bands,
        pp_results,
        paper_cl,  # single_cl,
        stamp_dir,
        set_name,
        stamp_type="single",
        cnn_res=None,
        with_table=True,
    )

    # make the case classification of each blend stamp and some figures
    blend_cl_pickle_path = os.path.join(
        blend_dir, f"set_with_hst{set_name}{suffix}_blend_classification.pickle"
    )
    if not os.path.isfile(blend_cl_pickle_path):
        blend_cl = blend_classification(pp_results, bands)
        with open(blend_cl_pickle_path, "wb") as pf:
            pickle.dump(blend_cl, pf)
    else:
        with open(blend_cl_pickle_path, "rb") as pf:
            blend_cl = pickle.load(pf)

    for k in blend_cl.keys():
        print(k, len(blend_cl[k]))

    # paper figure
    paper_cl = {"2": [19749]}
    make_figures(
        1,
        bands,
        pp_results,
        paper_cl,  # blend_cl,
        stamp_dir,
        set_name,
        stamp_type="blend",
        cnn_res=None,
        with_table=True,
    )

    paper_cl = {"0_fp": [113036]}
    make_figures(
        1,
        bands,
        pp_results,
        paper_cl,  # blend_cl,
        stamp_dir,
        set_name,
        stamp_type="blend",
        cnn_res=None,
        with_table=True,
    )

    # check if we missed any sample
    classified = np.zeros([len(pp_results.keys())])
    for k in single_cl.keys():
        for stamp_idx in single_cl[k]:
            classified[stamp_idx] = 1
    for k in blend_cl.keys():
        for stamp_idx in blend_cl[k]:
            classified[stamp_idx] = 1
    missed = np.where(classified == 0)
    print("Missed samples in classification...: ", missed[0])


if __name__ == "__main__":
    main()
