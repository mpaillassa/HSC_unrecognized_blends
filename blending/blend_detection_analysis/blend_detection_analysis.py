import math
import os
import pickle
import sys

import matplotlib as mpl
import numpy as np
import tqdm
from astropy.io import fits
from astropy.table import Table
from matplotlib.colors import LogNorm
from scipy.stats.stats import pearsonr
from sklearn_som.som import SOM

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def get_hscpipe_predictions_and_labels(nb_train, nb_test, final_results_dict):
    """Gets hscpipe predictions and ground truth labels.

    Args:
        - nb_train (int): number of training samples.
        - nb_test (int): number of testing samples.
        - final_results_dict (dict): final dict from make_final_dict.py
    Returns:
        - hscpipe_preds (np.ndarray): hscpipe predictions.
        - gt_labels (np.ndarray): ground truth labels.
    """

    hscpipe_preds = []
    gt_labels = []
    for k in range(nb_train, nb_train + nb_test):
        blend_bool = final_results_dict[k]["blend_flag"]
        gt_labels.append(blend_bool)

        hscpipe_pred = final_results_dict[k]["pipe_flag"]
        # if the sample is a single source
        if blend_bool == 0:
            if hscpipe_pred == "1":
                hscpipe_preds.append(0)
            else:
                hscpipe_preds.append(1)
        # if the sample is a blend
        if blend_bool == 1:
            if hscpipe_pred == "2":
                hscpipe_preds.append(1)
            else:
                hscpipe_preds.append(0)

    hscpipe_preds = np.array(hscpipe_preds)
    gt_labels = np.array(gt_labels)

    return hscpipe_preds, gt_labels


def make_basic_plots(
    set_name, gt_labels, hscpipe_preds, ml_preds, ml_model_name, out_dir
):
    """Makes some basic performance plots.

    Args:
        - set_name (string): name of the set to make plots for.
        - gt_labels (np.ndarray): ground truth labels.
        - hscpipe_preds (np.ndarray): hscpipe predictions.
        - ml_preds (np.ndarray): ml predictions.
        - ml_model_name (string): ml model name.
        - out_dir (string): output directory.
    """

    # make ml probability histogram
    make_proba_hist(set_name, gt_labels, ml_preds, ml_model_name, out_dir)

    # make ml confusion matrix
    make_cf_matrix(set_name, gt_labels, ml_preds, ml_model_name, out_dir)

    # make hscpipe confusion matrix
    make_cf_matrix(set_name, gt_labels, hscpipe_preds, "hscpipe", out_dir)

    # make roc for ml and hscpipe
    make_roc(set_name, gt_labels, ml_preds, hscpipe_preds, out_dir)


def make_proba_hist(set_name, gt_labels, ml_preds, ml_model_name, out_dir):
    """Makes histograms of the ml predicted probabilities.

    Args:
        - set_name (string): name of the corresponding set.
        - gt_labels (np.ndarray): ground truth labels.
        - ml_preds (np.ndarray): ml predictions.
        - ml_model_name (string): ml model name.
        - out_dir (string): output directory.
    """

    # get predictions of single objects and blends
    idx = np.where(gt_labels == 0)
    singles = ml_preds[idx]
    idx = np.where(gt_labels == 1)
    blends = ml_preds[idx]

    # make plots
    plt.hist(singles, bins=np.arange(0, 1.01, 0.01), label="Single", alpha=0.25)
    plt.hist(blends, bins=np.arange(0, 1.01, 0.01), label="Blend", alpha=0.25)
    plt.legend()
    plt.xlabel("ML model probability")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{ml_model_name}{set_name}_proba_hist.png")
    plt.gcf().clear()


def make_cf_matrix(set_name, gt_labels, preds, model_name, out_dir, threshold=0.5):
    """Makes confusion matrix.

    Args:
        - set_name (string): name of the corresponding set.
        - gt_labels (np.ndarray): ground truth labels.
        - preds (np.ndarray): predictions.
        - model_name (string): model name.
        - out_dir (string): output directory.
        - threshold (float): probability threshold to use for predictions.
    """

    # compute cf matrix
    cf_matrix = np.zeros([2, 2])
    for i, g in enumerate(gt_labels):
        if preds[i] > threshold:
            p = 0
        else:
            p = 1
        if g == 1:
            g = 0
        else:
            g = 1
        cf_matrix[g, p] += 1
    cf_matrix /= np.sum(cf_matrix, axis=1)

    # plot cf matrix
    gt_names = ["Blend", "Single"]
    pred_names = ["Predicted blend", "Predicted single"]
    fig, ax = plt.subplots()
    im = ax.imshow(cf_matrix)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(pred_names)))
    ax.set_xticklabels(pred_names, fontsize=14)
    ax.set_yticks(np.arange(len(gt_names)))
    ax.set_yticklabels(gt_names, fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(gt_names)):
        for j in range(len(pred_names)):
            if cf_matrix[i, j] > 0.25:
                text = ax.text(
                    j,
                    i,
                    f"{cf_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )
            else:
                text = ax.text(
                    j,
                    i,
                    f"{cf_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{model_name}{set_name}_cf.png")
    plt.gcf().clear()
    plt.close(fig)


def make_roc(set_name, gt_labels, ml_preds, hscpipe_preds, out_dir):
    """Makes ROC curve.

    Args:
        - set_name (string): name of the corresponding set.
        - gt_labels (np.ndarray): ground truth labels.
        - ml_preds (np.ndarray): ml predictions.
        - hscpipe_preds (np.ndarray): hscpipe predictions.
        - out_dir (string): output directory.
    """

    # make ml preds roc curve
    ind = np.argsort(ml_preds)
    sorted_pred = ml_preds[ind]
    sorted_gt = gt_labels[ind]

    nb_test = len(gt_labels)
    nb_pos = np.sum(sorted_gt)
    nb_neg = nb_test - nb_pos

    tp = nb_pos
    fp = nb_neg

    tprs = np.zeros([nb_test], dtype=np.float32)
    fprs = np.zeros([nb_test], dtype=np.float32)

    for k in range(nb_test):
        if sorted_gt[k]:
            tp -= 1
        else:
            fp -= 1
        tn = nb_neg - fp
        fn = nb_pos - tp

        tprs[k] = tp / nb_pos
        fprs[k] = fp / nb_neg

    auc = np.trapz(tprs, fprs)

    # compute hscpipe point
    hscpipe_tp = np.sum(np.logical_and(gt_labels, hscpipe_preds))
    hscpipe_tpr = hscpipe_tp / np.sum(gt_labels)
    hscpipe_fp = np.sum(np.logical_and(gt_labels == 0, hscpipe_preds))
    hscpipe_fpr = hscpipe_fp / np.sum(gt_labels == 0)

    ### ROC plot
    xminlim = 1e-08
    plt.plot([xminlim, 3], [0, 0], linestyle="--", color="black")
    plt.plot([xminlim, 3], [1, 1], linestyle="--", color="black")
    plt.plot(fprs, tprs, label="ML model")
    plt.plot(hscpipe_fpr, hscpipe_tpr, "o", label="HscPipe")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc{set_name}.png")
    plt.gcf().clear()


def get_blend_sample_data(
    nb_blend_samples,
    nb_train,
    nb_test,
    final_results_dict,
    hscpipe_preds,
    ml_preds,
    bands=["g", "r", "i", "z", "y"],
):
    """Gets the blend sample data in a handy way for further plots.

    Args:
        - nb_blend_samples (int): number of blend samples.
        - nb_train (int): number of training samples.
        - nb_test (int): number of testing samples.
        - final_results_dict (dict): final dictionary made with make_final_dict.py.
        - hscpipe_preds (np.ndarray): hscpipe predictions.
        - ml_preds (np.ndarray): ml predictions.
        - bands (list): list of bands.
    Returns:
        - blend_sample_data (dict): dictionary containing blend sample data (configuration and predictions).
    """

    # format with arrays across samples
    blend_sample_data = {
        "distance": np.zeros([nb_blend_samples]),
        "size_kron": np.zeros([nb_blend_samples, 2]),
        "size_moment": np.zeros([nb_blend_samples, 2]),
        "photoz": np.zeros([nb_blend_samples, 2]),
        "mags": np.zeros([nb_blend_samples, 5, 2]),
        "hscpipe_pred": np.zeros([nb_blend_samples]),
        "ml_pred": np.zeros([nb_blend_samples]),
    }

    # get data
    cur_idx = 0
    for k in tqdm.tqdm(range(nb_test), desc="Getting blend sample data"):
        samp_idx = k + nb_train

        if final_results_dict[samp_idx]["blend_flag"]:
            # distance
            distance = final_results_dict[samp_idx]["pix_distance"]
            blend_sample_data["distance"][cur_idx] = distance

            # size kron
            size_kron = final_results_dict[samp_idx]["i_kron_size"]
            blend_sample_data["size_kron"][cur_idx] = size_kron

            # size moment
            size_moment = final_results_dict[samp_idx]["i_moment_size"]
            blend_sample_data["size_moment"][cur_idx] = size_moment

            # photo-z
            photoz = final_results_dict[samp_idx]["elcosmos_photoz"]
            blend_sample_data["photoz"][cur_idx] = photoz

            for band_idx, band in enumerate(bands):
                band_mag = final_results_dict[samp_idx][band]["gt_mags"]
                blend_sample_data["mags"][cur_idx, band_idx] = band_mag

            blend_sample_data["hscpipe_pred"][cur_idx] = hscpipe_preds[k]
            blend_sample_data["ml_pred"][cur_idx] = ml_preds[k]
            cur_idx += 1

    # track invalid data
    invalid_data = np.zeros([nb_blend_samples])
    for key in blend_sample_data.keys():
        data = blend_sample_data[key]
        if len(data.shape) == 1:
            invalid_data = np.logical_or(
                invalid_data, np.logical_or(data < 0, np.isnan(data))
            )
        elif len(data.shape) == 2:
            for dim in range(data.shape[1]):
                invalid_data = np.logical_or(
                    invalid_data,
                    np.logical_or(data[:, dim] < 0, np.isnan(data[:, dim])),
                )
        elif len(data.shape) == 3:
            for dim1 in range(data.shape[1]):
                for dim2 in range(data.shape[2]):
                    invalid_data = np.logical_or(
                        invalid_data,
                        np.logical_or(
                            data[:, dim1, dim2] < 0, np.isnan(data[:, dim1, dim2])
                        ),
                    )

    # keep only valid data
    valid_data = np.logical_not(invalid_data)
    for key in blend_sample_data.keys():
        blend_sample_data[key] = blend_sample_data[key][valid_data]

    return blend_sample_data


def make_all_1d_plots(
    set_name, ml_model_name, blend_sample_data, blend_parameters, out_dir
):
    """Makes all blend detection accuracy vs 1d blend configuration parameter plots.

    Args:
        - set_name (string): name of the set.
        - ml_model_name (string): name of the ml model.
        - blend_sample_data (dict): dictionary containing blend sample data (configuration and predictions).
        - blend_parameters (dict): 1d blend configuration parameters to make 1d plots for. Keys are short names, values are for plot axis description.
        - out_dir (string): output directory.
    """

    for blend_parameter_name, blend_parameter_desc in blend_parameters.items():
        make_1d_plot(
            set_name,
            ml_model_name,
            blend_sample_data,
            blend_parameter_name,
            blend_parameter_desc,
            out_dir,
        )


def make_1d_plot(
    set_name,
    ml_model_name,
    blend_sample_data,
    blend_parameter_name,
    blend_parameter_desc,
    out_dir,
    threshold=0.5,
):
    """Makes the corresponding ml model and hscpipe blend detection accuracy vs 1d blend configuration parameter plot.
    Also makes an histogram of the corresponding 1d blend configuration parameter.

    Args:
        - set_name (string): name of the set.
        - ml_model_name (string): name of the ml model.
        - blend_sample_data (dict): dictionary containing blend sample data (configuration and predictions).
        - blend_parameter_name (string): 1d blend configuration parameter name to make 1d plots for.
        - blend_parameter_desc (string): 1d blend configuration parameter description to make 1d plots for.
        - out_dir (string): output directory.
        - threshold (float): probability threshold to use for predictions.
    """

    # get the blend configuration parameter values
    blend_parameter = get_1d_blend_parameter(blend_sample_data, blend_parameter_name)

    # make an historam of the blend configuration parameter
    mini_p, maxi_p = np.min(blend_parameter), np.max(blend_parameter)
    if blend_parameter_name == "norm_distance":
        step = (maxi_p - mini_p) / 800
        nb_bins = 50
    else:
        step = (maxi_p - mini_p) / 50
        nb_bins = 10
    counts, _, _ = plt.hist(
        blend_parameter, bins=np.arange(mini_p, maxi_p + step, step)
    )
    if blend_parameter_name == "norm_distance":
        plt.plot([0.01, 0.01], [0, np.max(counts)], color="black")
        plt.xlim([0, 0.5])
    fig_path = os.path.join(out_dir, f"{blend_parameter_name}_histogram{set_name}.png")
    plt.savefig(fig_path)
    plt.gcf().clear()

    # get the ml and hscpipe predictions
    ml_preds = blend_sample_data["ml_pred"]
    hscpipe_preds = blend_sample_data["hscpipe_pred"]

    # compute blend detection accuracy vs blend configuration parameter
    mini, maxi = np.min(blend_parameter), np.max(blend_parameter)
    step = (maxi - mini) / nb_bins
    ml_acc = np.zeros([nb_bins])
    hscpipe_acc = np.zeros([nb_bins])
    bin_values = np.zeros([nb_bins])
    for b in range(nb_bins):
        min_p = mini + b * step
        max_p = min_p + step

        idx = np.where(
            np.logical_and(blend_parameter > min_p, blend_parameter <= max_p)
        )
        if len(idx[0]):
            ml_bin_preds = ml_preds[idx[0]]
            np.place(ml_bin_preds, ml_bin_preds > threshold, 1)
            np.place(ml_bin_preds, ml_bin_preds < threshold, 0)
            ml_acc[b] = np.sum(ml_bin_preds) / len(idx[0])

            hscpipe_bin_preds = hscpipe_preds[idx[0]]
            np.place(
                hscpipe_bin_preds, hscpipe_bin_preds < threshold, 0
            )  # counts -1 cases as failures
            hscpipe_acc[b] = np.sum(hscpipe_bin_preds) / len(idx[0])
        else:
            ml_acc[b] = -1
            hscpipe_acc[b] = -1

        bin_values[b] = min_p + step / 2

    # make figure
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(bin_values, ml_acc, "o", linestyle="-", label="CNN")
    ax1.plot(bin_values, hscpipe_acc, "o", linestyle="-", label="HscPipe")
    ax1.set_xlabel(blend_parameter_desc, fontsize=14)
    ax1.set_ylabel("Blend detection accuracy", fontsize=14)
    ax1.legend(fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(
        out_dir, f"{blend_parameter_name}_{ml_model_name}{set_name}.png"
    )
    plt.savefig(fig_path)
    plt.gcf().clear()
    plt.close(fig)


def get_1d_blend_parameter(blend_sample_data, blend_parameter_name):
    """Gets 1d blend configuration parameter values.

    Args:
        - blend_sample_data (dict): dictionary containing blend configuration data and predictions.
        - blend_parameter_name (string): blend configuration parameter name to get values for.
    Returns:
        - blend_parameter (np.ndarray): blend configuration parameter values.
    """

    if blend_parameter_name == "distance":
        blend_parameter = blend_sample_data["distance"]
    elif blend_parameter_name == "norm_distance":
        distance = blend_sample_data["distance"]
        size_moment = blend_sample_data["size_moment"]
        blend_parameter = distance / np.sqrt(
            np.power(size_moment[:, 0], 2) + np.power(size_moment[:, 1], 2)
        )
        blend_parameter /= 2.355
    elif blend_parameter_name == "i_mag_diff":
        i_mag = blend_sample_data["mags"][:, 2]
        blend_parameter = np.absolute(i_mag[:, 0] - i_mag[:, 1])
    elif blend_parameter_name == "size_ratio_moment":
        size_moment = blend_sample_data["size_moment"]
        min_size_moment = np.min(size_moment, axis=1)
        max_size_moment = np.max(size_moment, axis=1)
        blend_parameter = min_size_moment / max_size_moment
    elif blend_parameter_name == "size_ratio_kron":
        size_kron = blend_sample_data["size_kron"]
        min_size_kron = np.min(size_kron, axis=1)
        max_size_kron = np.max(size_kron, axis=1)
        blend_parameter = min_size_kron / max_size_kron
    elif blend_parameter_name == "photoz_diff":
        photoz = blend_sample_data["photoz"]
        blend_parameter = np.absolute(photoz[:, 0] - photoz[:, 1])
    elif blend_parameter_name == "min_size_moment":
        size_moment = blend_sample_data["size_moment"]
        blend_parameter = np.min(size_moment, axis=1)
    elif blend_parameter_name == "min_i_mag":
        i_mag = blend_sample_data["mags"][:, 2]
        blend_parameter = np.min(i_mag, axis=1)

    return blend_parameter


def make_all_2d_plots(
    set_name, ml_model_name, blend_sample_data, blend_parameters, out_dir
):
    """Makes all blend detection accuracy vs 2d blend configuration parameter plots.

    Args:
        - set_name (string): name of the set.
        - ml_model_name (string): name of the ml model.
        - blend_sample_data (dict): dictionary containing blend sample data (configuration and predictions).
        - blend_parameters (dict): 2d blend configuration parameters to make 1d plots for. Keys are short names, values are for plot axis description.
        - out_dir (string): output directory.
    """

    for blend_parameter_name, blend_parameter_desc in blend_parameters.items():
        make_2d_plot(
            set_name,
            ml_model_name,
            blend_sample_data,
            blend_parameter_name,
            blend_parameter_desc,
            out_dir,
        )


def make_2d_plot(
    set_name,
    ml_model_name,
    blend_sample_data,
    blend_parameter_name,
    blend_parameter_desc,
    out_dir,
    nb_bins=25,
    threshold=0.5,
):
    """Makes the corresponding ml model and hscpipe blend detection accuracy vs 2d blend configuration parameter plot.
    Also makes an histogram of the corresponding 2d blend configuration parameter.

    Args:
        - set_name (string): name of the set.
        - ml_model_name (string): name of the ml model.
        - blend_sample_data (dict): dictionary containing blend sample data (configuration and predictions).
        - blend_parameter_name (string): 2d blend configuration parameter name to make 2d plots for.
        - blend_parameter_desc (list): 2d blend configuration parameter description to make 2d plots for.
        - out_dir (string): output directory.
        - threshold (float): probability threshold to use for predictions.
        - nb_bins (int): number of sampling bins to use.
    """

    # get blend configuration parameter values
    blend_parameter1, blend_parameter2 = get_2d_blend_parameter(
        blend_sample_data, blend_parameter_name
    )

    # define boundaries for binning
    if blend_parameter_name == "distance_photoz_diff":
        mini1 = np.min(blend_parameter2)
        maxi1 = np.max(blend_parameter2)
        mini2 = np.min(blend_parameter1)
        maxi2 = np.max(blend_parameter1)
    else:
        if blend_parameter_name == "min_i_mag_max_i_mag":
            mini = min(np.min(blend_parameter1), np.min(blend_parameter2))
            maxi = max(np.max(blend_parameter1), np.max(blend_parameter2))
        elif blend_parameter_name == "min_size_max_size_moment":
            mini = min(np.min(blend_parameter1), np.min(blend_parameter2))
            maxi = 7
        elif blend_parameter_name == "min_size_max_size_kron":
            mini = min(np.min(blend_parameter1), np.min(blend_parameter2))
            maxi = 22

        mini1, maxi1 = mini, maxi
        mini2, maxi2 = mini, maxi

    step1 = (maxi1 - mini1) / nb_bins
    step2 = (maxi2 - mini2) / nb_bins

    # compute blend detection accuracy vs 2d blend configuration parameter and bin counts for 2d histograms
    ml_acc = np.zeros([nb_bins, nb_bins])
    hscpipe_acc = np.zeros([nb_bins, nb_bins])
    bin_counts = np.zeros([nb_bins, nb_bins])

    # get the ml and hscpipe predictions
    ml_preds = blend_sample_data["ml_pred"]
    hscpipe_preds = blend_sample_data["hscpipe_pred"]

    # because ax.imshow is plotting dim1 top down and dim2 left right (y, x)
    # we just flipud to plot dim1 bottom up (y) and dim2 left right (x)
    # and as we specify (param1, param2) to be (x, y) then
    # param1 is dim2 when param2 is dim1
    # similarly yticks are bottom up so dim1 and xticks dim2
    xticks = np.zeros([nb_bins])
    yticks = np.zeros([nb_bins])
    for b1 in range(nb_bins):
        for b2 in range(nb_bins):
            min_p1 = mini1 + b1 * step1
            max_p1 = min_p1 + step1
            min_p2 = mini2 + b2 * step2
            max_p2 = min_p2 + step2

            m1 = np.logical_and(blend_parameter2 > min_p1, blend_parameter2 < max_p1)
            m2 = np.logical_and(blend_parameter1 > min_p2, blend_parameter1 < max_p2)
            idx = np.where(np.logical_and(m1, m2))

            if len(idx[0]):
                ml_bin_preds = ml_preds[idx[0]]
                np.place(ml_bin_preds, ml_bin_preds > threshold, 1)
                np.place(ml_bin_preds, ml_bin_preds < threshold, 0)
                ml_acc[b1, b2] = np.sum(ml_bin_preds) / len(idx[0])

                hscpipe_bin_preds = hscpipe_preds[idx[0]]
                np.place(
                    hscpipe_bin_preds, hscpipe_bin_preds < threshold, 0
                )  # counts -1 cases as failures
                hscpipe_acc[b1, b2] = np.sum(hscpipe_bin_preds) / len(idx[0])
            else:
                ml_acc[b1, b2] = -1
                hscpipe_acc[b1, b2] = -1

            bin_counts[b1, b2] = len(idx[0])

            xticks[b2] = min_p2 + step2 / 2
        yticks[nb_bins - b1 - 1] = min_p1 + step1 / 2

    # make all figures
    make_2d_fig(
        set_name,
        ml_model_name,
        ml_acc,
        blend_parameter_name,
        blend_parameter_desc,
        out_dir,
        xticks,
        yticks,
        nb_bins,
    )
    make_2d_fig(
        set_name,
        "hscpipe",
        hscpipe_acc,
        blend_parameter_name,
        blend_parameter_desc,
        out_dir,
        xticks,
        yticks,
        nb_bins,
    )
    make_2d_fig(
        set_name,
        "histogram",
        bin_counts,
        blend_parameter_name,
        blend_parameter_desc,
        out_dir,
        xticks,
        yticks,
        nb_bins,
    )


def get_2d_blend_parameter(blend_sample_data, blend_parameter_name):
    """Gets 2d blend configuration parameter values.

    Args:
        - blend_sample_data (dict): dictionary containing blend configuration data and predictions.
        - blend_parameter_name (string): blend configuration parameter name to get values for.
    Returns:
        - blend_parameter1 (np.ndarray): first blend configuration parameter values.
        - blend_parameter2 (np.ndarray): second blend configuration parameter values.
    """

    if blend_parameter_name == "min_i_mag_max_i_mag":
        i_mag = blend_sample_data["mags"][:, 2]
        blend_parameter1 = np.min(i_mag, axis=1)
        blend_parameter2 = np.max(i_mag, axis=1)
    elif blend_parameter_name == "min_size_max_size_moment":
        size_moment = blend_sample_data["size_moment"]
        blend_parameter1 = np.min(size_moment, axis=1)
        blend_parameter2 = np.max(size_moment, axis=1)
    elif blend_parameter_name == "min_size_max_size_kron":
        size_kron = blend_sample_data["size_kron"]
        blend_parameter1 = np.min(size_kron, axis=1)
        blend_parameter2 = np.max(size_kron, axis=1)
    elif blend_parameter_name == "distance_photoz_diff":
        blend_parameter1 = blend_sample_data["distance"]
        photoz = blend_sample_data["photoz"]
        blend_parameter2 = np.absolute(photoz[:, 0] - photoz[:, 1])

    return blend_parameter1, blend_parameter2


def make_2d_fig(
    set_name,
    model_name,
    values,
    blend_parameter_name,
    blend_parameter_desc,
    out_dir,
    xticks,
    yticks,
    nb_bins,
):
    """Makes the corresponding ml model and hscpipe blend detection accuracy vs 2d blend configuration parameter figure.

    Args:
        - set_name (string): name of the set.
        - model_name (string): name of the model.
        - values (np.ndarray): ready to plot values.
        - blend_parameter_name (string): 2d blend parameter name to make 2d plots for.
        - blend_parameter_desc (list): 2d blend parameter description to make 2d plots for.
        - out_dir (string): output directory.
        - xticks (np.ndarray): values of x ticks.
        - yticks (np.ndarray): values of y ticks
        - nb_bins (int): number of sampling bins.
    """

    # main plot
    fig, ax = plt.subplots()
    values = np.flipud(values)
    im = ax.imshow(values)
    cbar = ax.figure.colorbar(im, ax=ax)

    # adding tick labels
    new_xticks, new_yticks = [], []
    for b in range(0, nb_bins, nb_bins // 5):
        new_xticks.append(xticks[b])
        new_yticks.append(yticks[b])
    xtick_names = []
    ytick_names = []
    for xtick in new_xticks:
        xtick_names.append(f"{xtick:.1f}")
    for ytick in new_yticks:
        ytick_names.append(f"{ytick:.1f}")
    ax.set_xticks(np.arange(0, nb_bins, nb_bins // 5))
    ax.set_xticklabels(xtick_names)
    ax.set_yticks(np.arange(0, nb_bins, nb_bins // 5))
    ax.set_yticklabels(ytick_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # axis labels
    ax.set_xlabel(blend_parameter_desc[0], fontsize=14)
    ax.set_ylabel(blend_parameter_desc[1], fontsize=14)

    # plot and save
    plt.tight_layout()
    fig_path = os.path.join(
        out_dir, f"{blend_parameter_name}_{model_name}{set_name}.png"
    )
    plt.savefig(fig_path)
    plt.gcf().clear()
    plt.close(fig)


def make_som_analysis(
    set_name,
    ml_model_name,
    blend_sample_data,
    som_parameters,
    out_dir,
    som_size=25,
    learning_rate=1.0,
    nb_epochs=1,
):
    """Makes SOM analysis and SOM plots.

    Args:
        - set_name (string): name of the set.
        - ml_model_name (string): name of the ml model.
        - blend_sample_data (dict): dictionary containing blend sample data (configuration and predictions).
        - som_parameters (dict): blend configuration parameters to use to make the SOM analysis.
        - out_dir (string): output directory.
        - som_size (int): size of the SOMs to train.
        - learning_rate (float): learning rate to train the SOMs.
        - nb_epochs (int): number of epochs to train the SOMs.
    """

    # get the ml and hscpipe predictions
    ml_preds = blend_sample_data["ml_pred"]
    hscpipe_preds = blend_sample_data["hscpipe_pred"]

    # get the blend configuration parameters to use for the analysis
    som_params = dict()
    for param_name in som_parameters.keys():
        som_params[param_name] = get_1d_blend_parameter(blend_sample_data, param_name)

    # make an array to feed the SOM and record order
    nb_samples = len(ml_preds)
    nb_som_params = len(list(som_params.keys()))
    som_params_array = np.zeros([nb_samples, nb_som_params])
    som_params_idx = dict()
    param_idx = 0
    for param_name in som_parameters.keys():
        som_params_array[:, param_idx] = normalize(som_params[param_name])
        som_params_idx[param_name] = param_idx
        param_idx += 1

    # train SOMs model
    som_model = SOM(
        m=som_size, n=som_size, dim=nb_som_params, lr=learning_rate, random_state=0
    )
    som_model.fit(som_params_array, epochs=nb_epochs)

    # get the ml, hscpipe and blend configuration parameter maps
    ml_acc_som, hscpipe_acc_som, param_soms = get_ml_hscpipe_and_param_soms(
        som_model, som_params_array, som_params_idx, ml_preds, hscpipe_preds, som_size
    )

    # compute correlations between accuracy maps and parameter maps
    correlations = compute_correlations(
        ml_acc_som, hscpipe_acc_som, param_soms, som_params_idx, som_size
    )
    print(correlations["ml_corr"])
    print(correlations["hscpipe_corr"])
    # make som plots
    make_som_plot(set_name, f"{ml_model_name}_accuracy", ml_acc_som, out_dir)
    make_som_plot(set_name, "hscpipe_accuracy", hscpipe_acc_som, out_dir)
    for param_name, param_idx in som_params_idx.items():
        make_som_plot(set_name, param_name, param_soms[param_name], out_dir)


def normalize(array):
    """Normalizes the values in the array.

    Args:
        - array (np.ndarray): values to normalize.
    Returns:
        - norm_array (np.ndarray): normalized values.
    """

    norm_array = (array - np.mean(array)) / np.std(array)

    return norm_array


def get_ml_hscpipe_and_param_soms(
    som_model, som_params_array, som_params_idx, ml_preds, hscpipe_preds, som_size
):
    """Gets the soms of ml accuracy, hscpipe accuracy and soms parameters.
    The soms contain in each cell the mean of the accuracy or the mean of the given paramter of the configuration samples affected to this cell.

    Args:
        - som_model (sklearn_som.som.SOM): trained som model.
        - som_params_array (np.ndarray): soms parameters.
        - som_params_idx (dict): soms parameters order (keys: string name, values: idx in array).
        - ml_preds (np.ndarray): ml predictions.
        - hscpipe_preds (np.ndarray): hscpipe predictions.
        - som_size (int): som size.
    Returns:
        - ml_acc_som (np.ndarray): ml accuracy som.
        - hscpipe_acc_som (np.ndarray): hscpipe accuracy som
        - param_soms (dict): dictionary containing the som for each blend configuration parameter.
    """

    # get the predicted som cell of every blend configuration sample
    som_params_preds = som_model.predict(som_params_array)

    # build the ml and hscpipe accuracy soms and each parameter som
    ml_acc_som = np.zeros([som_size, som_size], dtype=np.float)
    hscpipe_acc_som = np.zeros([som_size, som_size], dtype=np.float)
    param_soms = dict()

    # init param soms
    for param_name in som_params_idx.keys():
        param_soms[param_name] = np.zeros([som_size, som_size], dtype=np.float)

    # for each som cell
    for y in range(som_size):
        for x in range(som_size):
            som_cell_nb = y * som_size + x

            # get the blend configuration samples affected to this cell
            idx = np.where(som_params_preds == som_cell_nb)

            # fill the ml and hscpipe accuracy soms, and the parameter soms
            if len(idx[0]):
                ml_acc_som[y, x] = np.sum(ml_preds[idx[0]]) / len(idx[0])
                hscpipe_acc_som[y, x] = np.sum(hscpipe_preds[idx[0]]) / len(idx[0])
                for param_name, param_idx in som_params_idx.items():
                    param_soms[param_name][y, x] = np.mean(
                        som_params_array[idx[0], param_idx]
                    )
            else:
                ml_acc_som[y, x] = -1
                hscpipe_acc_som[y, x] = -1
                for param_name in som_params_idx.keys():
                    param_soms[param_name][y, x] = -1

    return ml_acc_som, hscpipe_acc_som, param_soms


def compute_correlations(
    ml_acc_som, hscpipe_acc_som, param_soms, som_params_idx, som_size
):
    """Computes correlation coefficient between the ml and hscpipe soms and the parameter soms.

    Args:
        - ml_acc_som (np.ndarray): ml accuracy som.
        - hscpipe_acc_som (np.ndarray): hscpipe accuracy som.
        - param_soms (dict): blend configuration parameter soms.
        - som_params_idx (dict): soms parameters order (keys: string name, values: idx in array).
        - som_size (int): som size.
    Returns:
        - correlations (dict): dict of ml and hscpipe accuracy som correlation with blend configuration parameter soms.
    """

    nb_som_params = len(list(param_soms.keys()))

    correlations = {
        "ml_corr": np.zeros([nb_som_params]),
        "hscpipe_corr": np.zeros([nb_som_params]),
    }

    for param_name, param_idx in som_params_idx.items():
        correlations["ml_corr"][param_idx] = pearsonr(
            ml_acc_som.flatten(), param_soms[param_name].flatten()
        )[0]
        correlations["hscpipe_corr"][param_idx] = pearsonr(
            hscpipe_acc_som.flatten(), param_soms[param_name].flatten()
        )[0]

    return correlations


def make_som_plot(set_name, som_name, som, out_dir):
    """Makes som plot.

    Args:
        - set_name (string): name of the set.
        - som_name (string): name to use for file naming.
        - som (np.ndarray): som values.
        - out_dir (string): output directory.
    """

    fig, ax = plt.subplots()
    im = ax.imshow(som)
    cbar = ax.figure.colorbar(im, ax=ax)
    fig_path = os.path.join(out_dir, f"{som_name}{set_name}.png")
    fig.tight_layout()
    plt.savefig(fig_path)
    plt.gcf().clear()
    plt.close(fig)


def main():

    # data directories
    data_dir = "../../data"
    blend_dir = os.path.join(data_dir, "blending")
    set_name = "_ms8"
    suffix = "_release_run"
    ml_model_name = "torch_sky_sigma"

    # set ms8 is used for training and testing so we only use the last 50000 samples to test
    # set ms4 is only used for testing so we use the 200000 samples to test
    if set_name == "_ms4":
        nb_train = 0
        nb_test = 200000
    elif set_name == "_ms8":
        nb_train = 150000
        nb_test = 50000

    # get hscpipe predictions and ground truth labels
    pickle_file_path = os.path.join(
        blend_dir, f"set_with_hst{set_name}{suffix}_final.pickle"
    )
    with open(pickle_file_path, "rb") as pf:
        final_results_dict = pickle.load(pf)
    hscpipe_preds, gt_labels = get_hscpipe_predictions_and_labels(
        nb_train, nb_test, final_results_dict
    )

    # get ml predictions
    ml_preds_path = os.path.join(
        blend_dir, f"{ml_model_name}_inference_set_with_hst{set_name}{suffix}.npy"
    )
    ml_preds = np.load(ml_preds_path)
    ml_preds = ml_preds[-nb_test:]

    # output directories
    out_dir = "all_plots"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # make basic plots
    make_basic_plots(
        set_name, gt_labels, hscpipe_preds, ml_preds, ml_model_name, out_dir
    )

    # get blend sample data in a handy format to make blend detection accuracy plots
    nb_blend_samples = np.sum(gt_labels)
    blend_sample_data = get_blend_sample_data(
        nb_blend_samples, nb_train, nb_test, final_results_dict, hscpipe_preds, ml_preds
    )

    # make 1d plots, i.e. blend detection accuracy w.r.t. to a single blend configuration parameter
    # note that the corresponding functions must implement the corresponding format
    blend_parameters = {
        "distance": "Distance (pixels)",
        "norm_distance": "Distance normalized by sizes",
        "i_mag_diff": "I-band magnitude difference",
        "size_ratio_moment": "Moment size ratio",
        "size_ratio_kron": "Kron size ratio",
        "photoz_diff": "Photo-z difference",
    }
    cur_out_dir = os.path.join(out_dir, "1d_plots")
    if not os.path.isdir(cur_out_dir):
        os.mkdir(cur_out_dir)
    make_all_1d_plots(
        set_name, ml_model_name, blend_sample_data, blend_parameters, cur_out_dir
    )

    # make 2d plots, i.e. blend detection accuracy w.r.t. to two blend configuration parameters
    # note that the corresponding functions must implement the corresponding format
    blend_parameters = {
        "min_i_mag_max_i_mag": ["Minimum i-band magnitude", "Maximum i-band magnitude"],
        "min_size_max_size_moment": ["Minimum moment size", "Maximum moment size"],
        "min_size_max_size_kron": ["Minimum kron size", "Maximum kron size"],
        "distance_photoz_diff": ["Distance (pixels)", "Photo-z difference"],
    }
    cur_out_dir = os.path.join(out_dir, "2d_plots")
    if not os.path.isdir(cur_out_dir):
        os.mkdir(cur_out_dir)
    make_all_2d_plots(
        set_name, ml_model_name, blend_sample_data, blend_parameters, cur_out_dir
    )

    # make som analysis
    # note that the corresponding functions must implement the corresponding format
    som_parameters = {
        "distance": "Distance (pixels)",
        "min_size_moment": "Minimal moment size",
        "size_ratio_moment": "Moment size ratio",
        "min_i_mag": "Minimal i-band magnitude",
        "photoz_diff": "Photo-z difference",
    }
    cur_out_dir = os.path.join(out_dir, "som")
    if not os.path.isdir(cur_out_dir):
        os.mkdir(cur_out_dir)
    make_som_analysis(
        set_name, ml_model_name, blend_sample_data, som_parameters, cur_out_dir
    )


if __name__ == "__main__":
    main()
