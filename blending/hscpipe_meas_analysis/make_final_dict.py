import math
import os
import pickle

import tqdm
from astropy.io import fits


def main():

    # data directory
    data_dir = "../../data"

    # blending directories
    suffix = "_release_run"
    set_name = "_ms4"
    blend_dir = os.path.join(data_dir, "blending")
    stamp_dir = os.path.join(blend_dir, f"set_with_hst{set_name}{suffix}")

    # bands
    bands = ["g", "r", "i", "z", "y"]

    # get dictionary of preprocessed results from preprocess_hscpipe_meas.py
    pp_results_pickle_path = os.path.join(
        blend_dir, f"set_with_hst{set_name}{suffix}_pp_analysis.pickle"
    )
    with open(pp_results_pickle_path, "rb") as pf:
        pp_results = pickle.load(pf)

    # get dictionaries of classified results from classify_hscpipe_meas.py
    single_cl_pickle_path = os.path.join(
        blend_dir, f"set_with_hst{set_name}{suffix}_single_classification.pickle"
    )
    with open(single_cl_pickle_path, "rb") as pf:
        single_cl = pickle.load(pf)
    blend_cl_pickle_path = os.path.join(
        blend_dir, f"set_with_hst{set_name}{suffix}_blend_classification.pickle"
    )
    with open(blend_cl_pickle_path, "rb") as pf:
        blend_cl = pickle.load(pf)

    # validation catalog to retrieve isolated source measurements
    validation_cat_path = os.path.join(
        data_dir,
        f"galsim_cosmos_dataset/validation_process/validation_process{suffix}.fits",
    )
    validation_cat = fits.getdata(validation_cat_path)

    # make a final dictionary combining all the result analysis for further studies
    # the final dict final_dict has the following nested structure:
    #  keys: stamp indices
    #  values: dict with keys: "blend_flag"
    #                          "pipe_flag"
    #                          "ml_blend"
    #                          "elcosmos_photoz"
    #                          "F814W_flux_radius"
    #                          "i_size_kron"
    #                          "i_size_moments"
    #                          "pix_distance"
    #                          filter strings ("g", "r", "i", "z", "y")
    #                    values: whether the stamp is a blend (1) or a single object (0)
    #                            hscpipe classification analysis
    #                            ml model probability of being a blend
    #                            ground truth photoz from EL-COSMOS
    #                            flux radius from HST F814W catalog measurement
    #                            kron size from hscpipe isolated validation measurement
    #                            moment size from hscpipe isolated validation measurement
    #                            distance between sources in pixels
    #                            dict with keys: gt_coords
    #                                            gt_mags
    #                                            iso_mags
    #                                            iso_flux_err
    #                                            pred_coords
    #                                            pred_mags
    #                                            pred_flux_err
    #                                            det_matrix
    #                                      values: ground truth coordinates
    #                                              ground truth EL-COSMOS magnitudes
    #                                              hscpipe isolated magnitude measurements
    #                                              hscpipe isolated flux error measurements
    #                                              hscpipe predicted coordinates
    #                                              hscpipe predicted magnitudes
    #                                              hscpipe predicted flux errors
    #                                              hscpipe detection matrix
    final_dict = dict()

    # add all the single stamps
    for k in tqdm.tqdm(single_cl.keys(), desc="Single classification cases"):

        print(k, len(single_cl[k]))

        for stamp_idx in tqdm.tqdm(single_cl[k], desc=f"Stamps of case {k}"):

            stamp_dict = dict()

            ### general flags

            # blend or single object
            stamp_dict["blend_flag"] = 0

            # hscpipe result analysis classification
            stamp_dict["pipe_flag"] = k

            # ml proba to be blend or single # temporary to 0.5
            stamp_dict["ml_blend"] = 0.5

            ### ground truth general quantities

            # read stamp header to gather photoz and size
            stamp_path = os.path.join(stamp_dir, f"{stamp_idx}_stamp.fits")
            hd = fits.getheader(stamp_path, 1)
            id1 = hd["IDENT_1_0"]
            validation_row = validation_cat[validation_cat["IDENT"] == id1]

            # photoz
            photoz = [hd["ZPHOT_0"]]
            stamp_dict["elcosmos_photoz"] = photoz

            # flux radius size
            flux_radius = [hd["flux_radius_0"]]
            stamp_dict["F814W_flux_radius"] = flux_radius

            # kron size
            kron_size = validation_row["i_kron_radius"][0]
            stamp_dict["i_kron_size"] = kron_size

            # moment size
            s1 = (
                validation_row["i_shape_xx"] * validation_row["i_shape_yy"]
                - validation_row["i_shape_xy"] ** 2
            )
            if s1 > 0:
                moment_size = math.pow(s1, 0.25)
            else:
                moment_size = -1
            stamp_dict["i_moment_size"] = moment_size

            # distance
            stamp_dict["pix_distance"] = -1

            ### ground truth and hscpipe predicted quantities per band

            pred_ids = pp_results[stamp_idx]["pred_ids"]
            for band in bands:
                band_dict = dict()

                # gt coord
                band_dict["gt_coords"] = pp_results[stamp_idx][band]["gt_coords"]

                # gt mag
                band_dict["gt_mags"] = pp_results[stamp_idx][band]["gt_mags"]

                # iso mags
                band_dict["iso_mags"] = pp_results[stamp_idx][band]["iso_mags"]

                # iso flux err
                band_dict["iso_flux_err"] = pp_results[stamp_idx][band]["iso_flux_err"]

                # hscpipe measurements
                det_matrix = pp_results[stamp_idx][band]["det_matrix"]

                pred_coords = []
                pred_mags = []
                pred_flux_err = []
                for pred_id in pred_ids:
                    pred_coords.append(
                        pp_results[stamp_idx][band][pred_id]["pred_coords"]
                    )
                    pred_mags.append(pp_results[stamp_idx][band][pred_id]["pred_mag"])
                    pred_flux_err.append(
                        pp_results[stamp_idx][band][pred_id]["pred_flux_err"]
                    )

                band_dict["pred_coords"] = pred_coords
                band_dict["pred_mags"] = pred_mags
                band_dict["pred_flux_err"] = pred_flux_err
                band_dict["det_matrix"] = det_matrix

                stamp_dict[band] = band_dict

            final_dict[stamp_idx] = stamp_dict

    # add all the blend stamps
    for k in tqdm.tqdm(blend_cl.keys(), desc="Blend classification cases"):

        print(k, len(blend_cl[k]))

        for stamp_idx in tqdm.tqdm(blend_cl[k], desc=f"Stamps of case {k}"):

            stamp_dict = dict()

            ### general flags

            # blend or single object
            stamp_dict["blend_flag"] = 1

            # hscpipe result analysis classification
            stamp_dict["pipe_flag"] = k

            # ml proba to be blend or single # temporary to 0.5
            stamp_dict["ml_blend"] = 0.5

            ### ground truth general quantities

            # read stamp header to gather photoz and size
            stamp_path = os.path.join(stamp_dir, f"{stamp_idx}_stamp.fits")
            hd = fits.getheader(stamp_path, 1)
            id1 = hd["IDENT_1_0"]
            validation_row1 = validation_cat[validation_cat["IDENT"] == id1]
            id2 = hd["IDENT_1_1"]
            validation_row2 = validation_cat[validation_cat["IDENT"] == id2]

            # photoz
            photoz = [hd["ZPHOT_0"], hd["ZPHOT_1"]]
            stamp_dict["elcosmos_photoz"] = photoz

            # flux radius size
            flux_radius = [hd["flux_radius_0"], hd["flux_radius_1"]]
            stamp_dict["F814W_flux_radius"] = flux_radius

            # kron size
            kron_size = [
                validation_row1["i_kron_radius"][0],
                validation_row2["i_kron_radius"][0],
            ]
            stamp_dict["i_kron_size"] = kron_size

            # moment size
            s1 = (
                validation_row1["i_shape_xx"] * validation_row1["i_shape_yy"]
                - validation_row1["i_shape_xy"] ** 2
            )
            if s1 > 0:
                moment_size1 = math.pow(s1, 0.25)
            else:
                moment_size1 = -1
            s2 = (
                validation_row2["i_shape_xx"] * validation_row2["i_shape_yy"]
                - validation_row2["i_shape_xy"] ** 2
            )
            if s2 > 0:
                moment_size2 = math.pow(s2, 0.25)
            else:
                moment_size2 = -1
            stamp_dict["i_moment_size"] = [moment_size1, moment_size2]

            # distance
            posx1, posy1 = hd["x_peak_0"], hd["y_peak_0"]
            posx2, posy2 = hd["x_peak_1"], hd["y_peak_1"]
            distance = math.sqrt((posx1 - posx2) ** 2 + (posy1 - posy2) ** 2)
            stamp_dict["pix_distance"] = distance

            ### ground truth and hscpipe predicted quantities per band

            pred_ids = pp_results[stamp_idx]["pred_ids"]
            for band in bands:
                band_dict = dict()

                # gt coord
                band_dict["gt_coords"] = pp_results[stamp_idx][band]["gt_coords"]

                # gt mag
                band_dict["gt_mags"] = pp_results[stamp_idx][band]["gt_mags"]

                # iso mags
                band_dict["iso_mags"] = pp_results[stamp_idx][band]["iso_mags"]

                # iso flux err
                band_dict["iso_flux_err"] = pp_results[stamp_idx][band]["iso_flux_err"]

                # hscpipe measurements
                det_matrix = pp_results[stamp_idx][band]["det_matrix"]

                pred_coords = []
                pred_mags = []
                pred_flux_err = []
                for pred_id in pred_ids:
                    pred_coords.append(
                        pp_results[stamp_idx][band][pred_id]["pred_coords"]
                    )
                    pred_mags.append(pp_results[stamp_idx][band][pred_id]["pred_mag"])
                    pred_flux_err.append(
                        pp_results[stamp_idx][band][pred_id]["pred_flux_err"]
                    )

                band_dict["pred_coords"] = pred_coords
                band_dict["pred_mags"] = pred_mags
                band_dict["pred_flux_err"] = pred_flux_err
                band_dict["det_matrix"] = det_matrix

                stamp_dict[band] = band_dict

            final_dict[stamp_idx] = stamp_dict

    final_pickle_path = os.path.join(
        blend_dir, f"set_with_hst{set_name}{suffix}_final.pickle"
    )
    with open(final_pickle_path, "wb") as pf:
        pickle.dump(final_dict, pf)


if __name__ == "__main__":
    main()
