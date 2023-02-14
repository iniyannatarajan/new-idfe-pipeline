================
Data description
================

In the 2018 observing campaign of the EHT, M87 was observed over multiple days and frequency bands. Two calibration pipelines, HOPS and CASA, were used to calibrate these data.
The output of each calibration pipeline was processed by 5 imaging tools. To validate the performance of the imaging tools, 15 synthetic datasets (of which 5 are loosely classified as
"validation" datasets), from 15 different source models, were generated for each day+band combination. More details on the outputs and conventions followed by the imaging tools are given below. 
Customarily, the results of the various imaging tools are copied to eht-cloud.

Days : 3644 (April 21) and 3647 (April 25).

Bands: b1, b2, b3, b4; some imaging tools also process combined band data i.e., b1+2, b3+4, b1+2+3+4

Comrade
^^^^^^^

**Comrade/netcal** - hosts all the synthetic and M87 datasets.

All the synthetic datasets are named **<model>_<day>_<band>**. All the M87 datasets are named **<casa/hops>_<day>_<band>**.

Comrade does not process band-combined datasets.

SMILI
^^^^^

**smili** - hosts all the synthetic and M87 datasets. 

All the synthetic datasets are named **<model>_<day>_<band>**. All the M87 datasets are named **<casa/hops>_<day>_<band>**.

SMILI also includes band-combined datasets labelled b1+2, b3+4, and b1+2+3+4.

difmap
^^^^^^

**difmap** - hosts all the synthetic and M87 datasets. Also hosts an entirely parallel set of datasets suffixed **_geofit**.

All the synthetic datasets are named **<model>_<day>_<band>**. All the M87 datasets are named **<casa/hops>_<day>_<band>**.

difmap does not process band-combined datasets.

THEMIS
^^^^^^

**THEMIS/synthetic_unblurred/netcal** - hosts all synthetic datasets; on Cannon cluster, the path is slightly changed to **THEMIS/netcal**.
**THEMIS/M87real** - hosts all M87 datasets, subdivided further under **<casa/hops>_raster+LSG_unblurred>**.

All the synthetic datasets are named **<model>_<day>_<band>**. All the M87 datasets are named **<casa/hops>_<day>_<band>**.

THEMIS also includes band-combined datasets labelled b1b2, b3b4, and b1b2b3b4, except for **hops_3644** datasets, which are labelled b12, b34, and b1234.

ehtim
^^^^^
