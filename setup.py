from setuptools import setup, find_packages
from idfe import __version__

with open("README.rst") as tmp:
    readme = tmp.read()

setup(
    author='Iniyan Natarajan',
    author_email='iniyan.natarajan@cfa.harvard.edu',
    name='idfe',
    version=__version__,
    description='Image Domain Feature Extraction for the Event Horizon Telescope',
    long_description=readme,
    long_description_content_type="text/x-rst",
    url='https://github.com/iniyannatarajan/eht2018-idfe-pipeline/',
    license='GNU GPL v2',
    packages=find_packages(include=['idfe','idfe.*']),
    package_data={
        'idfe': ['data/*'],
        },
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'pandas',
        'joypy',
        'h5py',
        'astropy',
        'ephem',
        'future',
        'requests',
        'tables',
        'termcolor',
        'ipython',
        ],
    keywords='idfe',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.9',
        ],
)
