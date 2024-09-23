import os
import pickle
import random as rd

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from astropy.io import fits
from matplotlib import gridspec


def make_figure(stamp_dict, stamp_idx, stamp_dir, bands, out_dir):

    # get hsc image
    hsc_path = os.path.join(stamp_dir, f"{stamp_idx}_stamp.fits")
    hsc_im = fits.getdata(hsc_path)[bands]
    for ch in range(hsc_im.shape[0]):
        mini = np.min(hsc_im[ch])
        maxi = np.max(hsc_im[ch])
        hsc_im[ch] = 255 * (np.flipud(hsc_im[ch]) - mini) / (maxi - mini)

    # get hst image
    hst_path = os.path.join(stamp_dir, f"{stamp_idx}_hst_stamp.fits")
    hst_im = fits.getdata(hst_path)[0]
    hst_im = 255 * (hst_im - np.min(hst_im)) / (np.max(hst_im) - np.min(hst_im))

    # figure layout
    row_size = 15
    fig = plt.figure(figsize=(row_size, row_size))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 1])
    ax1, ax2 = fig.add_subplot(spec[0]), fig.add_subplot(spec[1])
    ax1.axis("off")
    ax2.axis("off")

    # draw hsc and hst images
    ax1.imshow(np.transpose(hsc_im, (1, 2, 0)).astype(np.uint8))
    ax2.imshow(np.flipud(hst_im).astype(np.uint8))

    # add position overlays on hsc image
    im_size = hsc_im.shape[1]
    gt_coords = stamp_dict["i"]["gt_coords"]
    k = 1
    for gt_coord in gt_coords:
        ax1.plot(gt_coord[0], im_size - gt_coord[1], "+", color="black")
        ax1.annotate(
            str(k),
            xy=(gt_coord[0] - 4, im_size - gt_coord[1]),
            color="black",
            weight="bold",
        )
        k += 1
    k = 1
    pred_coords = stamp_dict["i"]["pred_coords"]
    if len(pred_coords):  # avoid failures when there are not predicted objects
        for pred_coord in pred_coords:
            if not isinstance(
                pred_coord, int
            ):  # avoid failures when coords are not valid
                ax1.plot(pred_coord[0], im_size - pred_coord[1], "x", color="red")
                ax1.annotate(
                    str(k),
                    xy=(pred_coord[0] + 2, im_size - pred_coord[1]),
                    color="red",
                    weight="bold",
                )
            k += 1

    # save figure
    out_path = os.path.join(out_dir, f"{stamp_idx}_fig.png")
    fig.tight_layout()
    plt.savefig(out_path)
    plt.gcf().clear()


def main():

    # data directory
    data_dir = "../../data"

    # blending directories
    suffix = "_release_run"
    set_name = "_ms8"
    blend_dir = os.path.join(data_dir, "blending")
    stamp_dir = os.path.join(blend_dir, f"set_with_hst{set_name}{suffix}")

    # get the main dictionary from pickle file
    pickle_file_path = os.path.join(
        blend_dir, f"set_with_hst{set_name}{suffix}_final.pickle"
    )
    with open(pickle_file_path, "rb") as pickfile:
        main_dict = pickle.load(pickfile)

    # bands to use to draw hsc image
    bands = [0, 2, 4]  # for g i z

    # output directory
    out_dir = f"./figures/set{set_name}/with_hst"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # pick random samples to make figures
    all_stamp_indexes = list(main_dict.keys())
    picked_stamp_indexes = rd.choices(all_stamp_indexes, k=10)
    for stamp_idx in tqdm.tqdm(picked_stamp_indexes, desc="Figures"):
        make_figure(main_dict[stamp_idx], stamp_idx, stamp_dir, bands, out_dir)


if __name__ == "__main__":
    main()
