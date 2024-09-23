# MLDeblender

This repository contains scripts to reproduce the data products and results of blend (and unrecognized blend) identification for HSC.
As it involves many different environments (HSC pipelines, LSST pipelines, Blending ToolKit, GPU machine), it consists of different directories containing scripts to run separately.
The directories and their corresponding use case are listed below and more details are contained in additional READMEs in each directory.

## galsim_cosmos_dataset

This directory contains code to build the extended galsim COSMOS data set and its validation process.

## blending

This directory contains code to:
* simulate training and testing samples to train the blend identifier.
* make fake HSC coadds from the samples and analyze the HSC pipelines detections on those samples.
* analyze and compare the blend identifier and HSC pipelines detection results.

## training

This directory contains code to train the blend identifier.
