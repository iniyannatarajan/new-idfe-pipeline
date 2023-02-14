=======
Running
=======

The driver scripts *eht2018_idfe_driver_realdata.py* and *eht2018_idfe_driver_syndata.py* set values for variables necessary to locate
topsets/posteror samples and run REx and VIDA on them. The output of the driver scripts are a set of REx and VIDA output files in HDF5 and CSV formats respectively.

These scripts use no comand-line arguments and all the values are set within the script to keep a record of what was run. The user must check these values at the beginning of every run.

Copy the required IDFE driver script to the output directory, modify the input variables, and run with python 3.6 or above.
To see the available command-line arguments run with the -h option.

.. code-block:: bash

    python <idfe-driver-script> -h

