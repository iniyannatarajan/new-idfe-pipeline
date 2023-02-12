#!/usr/bin/env python

# Script to compute mean image of all images in a given directory

import os
import numpy as np
from astropy.io import fits

import ehtplot
#%matplotlib inline
import matplotlib.pyplot as plt

# define some variables
DEG2UAS = 3600e6 # convert degrees to uas
FoV = 128 # this value chosen since both smili and ehtim image this FoV; the others will be clipped to this FoV

parentdir = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE'

imagerlist = ['Comrade', 'smili', 'difmap', 'THEMIS', 'ehtim']
netcal = 'netcal' # only netcal data always
modellist = ['ring', 'cres000', 'cres090', 'cres180', 'cres270', 'dblsrc', 'disk', 'edisk', 'point+disk', 'point+edisk', 'grmhd', 'ecres000', 'ecres045', 'ecres090', 'ecres315']
daylist = ['3644', '3647']
caliblist = ['hops', 'casa']
bandlist = ['b1', 'b2', 'b3', 'b4']
smilibandlist = ['b1+2', 'b3+4', 'b1+2+3+4']
themisbandlist = ['b1b2', 'b3b4', 'b1b2b3b4']

# get image parameters for each imager from a representative image
repimgdict = {'Comrade': os.path.join(parentdir, 'Comrade', netcal, 'ring_3644_b1', 'ring_3644_b1_image_0001.fits'),
                'smili': os.path.join(parentdir, 'smili', 'ring_3644_b1', 'ring_3644_b1_000001.fits'),
                'difmap': os.path.join(parentdir, 'difmap', 'ring_3644_b1', 'ring_3644_b1_000001.fits'),
                'THEMIS': os.path.join(parentdir, 'THEMIS', netcal, 'ring_3644_b1', 'ring_3644_b1_001.fits'),
                'ehtim': os.path.join(parentdir, 'ehtim', 'ring_3644_hops-b1', 'ring_3644_hops-b1_0000001.fits')}

repimgdims = {}

for key in repimgdict.keys():
    hdul = fits.open(repimgdict[key])
    repimgdims[key] = hdul[0].header['NAXIS1']
    hdul.close()

for imager in imagerlist:
    for model in modellist:
        for day in daylist:
            for calib in caliblist:
                if imager == 'smili': bands = bandlist + smilibandlist
                elif imager == 'THEMIS': bands = bandlist + themisbandlist
                else: bands = bandlist
                for band in bands:
                    if imager == 'smili' and model == 'ring' and day == '3644' and calib == 'hops' and band == 'b1':
                        inputdir = os.path.join(parentdir, imager, f'{model}_{day}_{band}')
                        label = f'{imager}_{model}_{day}_{calib}_{band}'
                        print(f'Processing {label}...')
             
                        # create input file list
                        filelist = f'{label}.filelist'
                        cmd = f"readlink -f {inputdir}/*.fits > {filelist}"
                        os.system(cmd)

                        fnames = np.genfromtxt(f'{filelist}', dtype='str')
                        #os.system(f'rm {filelist}')

                        # read images one by one and add to imgsum
                        imgsum = np.zeros((repimgdims[imager], repimgdims[imager]))
                        ntotal = len(fnames)
                        for fname in fnames:
                            hdul = fits.open(fname)
                            if imager in ['Comrade', 'ehtim', 'THEMIS']:
                                img = hdul[0].data
                            elif imager in ['difmap', 'smili']:
                                img = hdul[0].data[0,0]
                            hdul.close()
 
                            imgsum += img
 
                        imgmean = imgsum/ntotal
 
                        # create new FITS file
                        # copy header from the first file in fnames
                        hdultmp = fits.open(fnames[0])
                        cphdr = hdultmp[0].header
                        hdultmp.close()
 
                        hdumean = fits.PrimaryHDU(data=imgmean, header=cphdr)
                        hdulmean = fits.HDUList([hdumean])
                        hdulmean.writeto(f'{label}_mean.fits')
 
                        # write as png image
 
                        # CLEAN and Themis FITS files have a larger FoV. Cut them to 150 uas for easier visualisation
                        if imager in ['CLEAN', 'Themis']:
                            # read in values from the header
                            naxis1 = hdumean.header['NAXIS1']
                            naxis2 = hdumean.header['NAXIS2']
                            crpix1 = hdumean.header['CRPIX1']
                            crpix2 = hdumean.header['CRPIX2']
                            cdelt1 = hdumean.header['CDELT1']
                            cdelt2 = hdumean.header['CDELT2']
 
                            # Determine indices for slicing
                            ylower = int(np.floor(crpix2-(FoV/(cdelt2*DEG2UAS)/2)))
                            yupper = int(np.floor(crpix2+(FoV/(cdelt2*DEG2UAS)/2)))
                            xlower = int(np.floor(crpix1+(FoV/(cdelt1*DEG2UAS)/2)))
                            xupper = int(np.floor(crpix1-(FoV/(cdelt1*DEG2UAS)/2)))
 
                            plt.imsave(f'{label}_mean_FoV{FoV}uas.png', imgmean[ylower:yupper, xlower:xupper], origin='lower', cmap='afmhot_10us')
                        else:
                            plt.imsave(f'{label}_mean_FoV{FoV}uas.png', imgmean, origin='lower', cmap='afmhot_10us')
 
                        print(f'{label} images created.')
