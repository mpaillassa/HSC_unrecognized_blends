# blending

Simulate training and testing samples to train the blend identifier:
* _make_btk_stamps.py_: simulate samples of isolated galaxies and blends of galaxies with the Blending ToolKit. This needs to be run in the Blending ToolKit environement.

Make fake HSC coadds from the samples and process them with the HSC pipelines:
* _make_fake_coadds.py_: make fake HSC coadds from the samples to run the HSC pipelines on them.
* _process_fake_coadds.py_: run the HSC pipelines on the fake coadds. This needs to be run in an HSC pipelines environment, with the emulateHscCoadd task setup.


## hscpipe_meas_analysis

Analyze the HSC pipelines detections on the samples:
* _preprocess_hscpipe_meas.py_: get the results of HSC pipelines detections on the samples.
* _classify_and_plot_hscpipe_meas.py_: get more precise results for isolated galaxies and blends of galaxies separately. Also contains code to make figures of given samples showing images and HSC pipelines results.
* _make_final_dict.py_: rearrange the results into a final dict more convenient to be used for the comparison analysis with the blend identifier model.
* _hsc_with_hst_figures_.py: make figures of HSC stamps along with the corresponding (simulated) HST stamps.

## blend_detection_analysis

Analyze and compare the blend identifier and HSC pipelines detection results. This needs the blend identifier trained to get its results.
* _blend_detection_analysis.py_: make confusion matrices, blend detection accuracy plots vs configuration parameters, SOM analysis.
