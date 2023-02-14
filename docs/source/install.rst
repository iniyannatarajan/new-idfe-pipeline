============
Installation
============

We recommend installing the pipeline inside a virtual environment running Python 3.10 or above.
On a new Debian-based system, one may need to install, among other things, *python3-venv* via *apt-get*.
As always, upgrade the following packages in a new virtual environment:

.. code-block:: bash

    pip install --upgrade pip setuptools wheel

Prerequisites
-------------

If installing on a Debian-based system, ensure that *cmake* and *libboost-python-dev* are installed.

Clone the dev branch of *eht-imaging* and install using *pip*.

.. code-block:: bash

    git clone -b dev https://github.com/achael/eht-imaging.git
    cd eht-imaging
    pip install .

Ensure that numpy version 1.23.x is installed (uninstall any version that is higher) for eht-imaging to work properly. For example::

    conda install -c conda-forge numpy=1.23.5=py310h53a5b5f_0

Install other necessary packages::

    conda install -c conda-forge seaborn joypy termcolor

Clone *ehtplot* and install using *pip*.

.. code-block:: bash

    git clone https://github.com/liamedeiros/ehtplot.git
    cd ehtplot
    pip install .

Install *metronization* from GitHub using *pip* (Metronization installation does not work as of now!!!).

.. code-block:: bash

    pip install -e git+git://github.com/focisrc/metronization.git#egg=metronization

Installing the pipeline
-----------------------

Clone the IDFE pipeline and install using *pip*.

.. code-block:: bash

    git clone https://github.com/iniyannatarajan/eht2018-idfe-pipeline.git
    cd eht2018-idfe-pipeline
    pip install -e .
