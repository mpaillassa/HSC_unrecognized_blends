import math
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def main():

    # data directories
    data_dir = "../data"
    galsim_cosmos_data_dir = os.path.join(data_dir, "galsim_cosmos_dataset")

    # catalogs paths
    parent_cat_path = os.path.join(
        galsim_cosmos_data_dir, "hst_cut26_el_cosmos_lensing14_match.fits"
    )
    parent_cat = fits.getdata(parent_cat_path, 1)
    sample_cat_path = os.path.join(
        data_dir,
        "galsim_cosmos_dataset/full_release_run/real_galaxy_catalog_26_extension_fits.fits",
    )
    sample_cat = fits.getdata(sample_cat_path)

    # get flux radius
    parent_flux_radius = parent_cat["FLUX_RADIUS_1"][
        :, 1
    ]  # 0, 1, 2, 3 are for 25%, 50%, 75%, 100% flux radius
    sample_flux_radius = sample_cat["flux_radius"]

    # plot
    plt.hist(
        parent_flux_radius,
        bins=100,
        density=True,
        range=[0, 30],
        label="Normalized parent distribution",
        alpha=0.25,
    )
    plt.hist(
        sample_flux_radius,
        bins=100,
        density=True,
        range=[0, 30],
        label="Normalized sampled distribution",
        alpha=0.25,
    )
    plt.xlabel("Flux radius", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig("flux_radius_distributions.png")


if __name__ == "__main__":
    main()
