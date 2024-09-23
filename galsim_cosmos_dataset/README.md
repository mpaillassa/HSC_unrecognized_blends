# galsim_cosmos_dataset

The extended GalSim COSMOS data set is done with _make_data_set.py_. In order to generate it, the following data is needed:
* hst_cut26_el_cosmos_lensing14_match.fits: the HST catalog from [Leauthaud et al., 2007](https://ui.adsabs.harvard.edu/abs/2007ApJS..172..219L/abstract), cut with the _make_hst_cut_ function (same cuts as the original GalSim COSMOS data set, except the F814W magnitude cut at 26 instead of 25.2), crossmatched with the [EL-COSMOS catalog](https://ui.adsabs.harvard.edu/abs/2020MNRAS.494..199S/abstract) to get HSC multiband magnitudes and lensing14.fits (provided by Rachel Mandelbaum to get the columns providing the focus information needed to get the appropriate HST PSF for each galaxy). Expected to be in </path_to_repo/data/galsim_cosmos_dataset>.
* HST mosaic files: can be downloaded with _download_hst.py_. Expected to be in <path_to_repo/data/HST_mosaic>.
* HST PSF models: can be downloaded from [here](http://www.astro.dur.ac.uk/~rjm/acs/PSF/). Expected to be in <path_to_repo/data/galsim_cosmos_dataset/HST_PSF>.


## validation_process

This directory contains code to make the validation process of the data by using the HSC pipelines. It consists of several subdirectories.

### detection_and_measurements
* _make_fake_coadds.py_: generate Blending ToolKit stamps of each galaxy as isolated and arrange them into coadds to be processed by the HSC pipelines. This needs to be run in the BTK environment. Coadds also include stars so that the pipeline can use them to estimate PSF and ApCorr.
* _process_fake_coadds.py_: run the HSC pipelines on the fake Blending ToolKit stamp HSC coadds. This needs to be run in an HSC pipelines environment, with the emulateHscCoadd task setup (see below) and consists of several HSC pipelines command line tasks:
    *  Emulate HSC coadds with the emulateHscCoadd task from https://github.com/jcoupon/importExtData, slightly modified to run with HscPipe 8.
    *  Detecting sources with the detectCoaddSources task.
    *  Running tasks to get forced measurements through the multibandDriver.
* _make_validation_plots.py_: analyze the detection/measurement results and makes validation plots (magnitude histograms of detected galaxies and true versus measured magnitude plots).

### mag_vs_snr
* _download_hsc_data.py_: download all necessary pdr2_wide HSC data of COSMOS field (9813 tract, all 81 patches, 9812 tract, 0,X patches) to inject sources in those coadds.
* _prepare_injection.py_: prepare fake catalogs and fake source images to be added in pdr2_wide HSC data with the insertFakes task.
* _run_injection.py_: run the insertFakes task to inject fake sources in pdr2_wide HSC data. This needs to be run with LSST pipelines > 23.0.0 because injecting sources in real data from images is not yet in HscPipe 8.
* _run_inj_meas.py_: run pipeline detection and measurements on the pdr2_wide HSC data containing fake sources.
* _compare_meas_noise.py_: compare measurements of injected sources within HSC pdr2_wide data with their respective Blending ToolKit stamp measurements to assess if the noise in Blending ToolKit stamps is similar to that of real data.
