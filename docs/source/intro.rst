=======================
EHT2018-IDFE-pipeline
=======================

This pipeline is used for performing image domain analysis of EHT 2018 data, focussing on the M87 science case.
The first part of the pipeline performs image domain feature extraction (IDFE) using the Ring Extractor module (REx, Chael 2019) from eht-imaging and Variational Image Domain Analysis
(VIDA, Tiede et al. 2022) from VIDA.jl. The second part of the pipeline applies the metronization (Christian et al. 2022) algorithm to perform Topographical Data Analysis for identifying
ring-like structures in the input images, and consolidates the results of all the previous steps.

This documentation explains the steps necessary for installing and running the pipeline.
