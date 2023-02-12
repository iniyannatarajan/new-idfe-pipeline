#!/usr/bin/env python

# Script to create blurred FITS images using AIPS CC components
# Used to create a new set of topset images from the original DIFMAP images for further IDFE+TDA processing

import os
import numpy as np
import pandas as pd
import ehtim as eh
from ehtim.const_def import *
from idfe.utils import *
#from functools import partial
#from multiprocessing import Pool

# define some variables
imagerlist =  ['difmap', 'difmap_geofit']
modellist = ['ring', 'cres000', 'cres090', 'cres180', 'cres270', 'dblsrc', 'disk', 'edisk', 'point+disk', 'point+edisk', 'grmhd', 'ecres000', 'ecres045', 'ecres090', 'ecres315', 'casa', 'hops']
daylist = ['3644'] # '3647' later
bandlist = ['b1', 'b2', 'b3', 'b4']

inparentdir = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/difmap'
outparentdir = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/difmap_blurredcc_r2'
topsetdir = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/topset'

proc = 48 # number of processes; must not exceed the number of physical cores available

fov = 200*RADPERUAS # uas to rad
npix = 256
interp = 'linear'
beamsize = 20*RADPERUAS # uas to rad
frac = 1.0

for imager in imagerlist:
    for model in modellist:
        for day in daylist:
            for band in bandlist:
                # get topset ids
                if imager == 'difmap':
                    if model == 'casa': topsetfile = os.path.join(topsetdir, 'CLEAN', f'topset_casa_{day}_{band}.csv')
                    else: topsetfile = os.path.join(topsetdir, 'CLEAN', f'topset_hops_{day}_{band}.csv')
                elif imager == 'difmap_geofit':
                    if model == 'casa': topsetfile = os.path.join(topsetdir, 'CLEAN_geofit', f'topset_casa_{day}_{band}.csv')
                    else: topsetfile = os.path.join(topsetdir, 'CLEAN_geofit', f'topset_hops_{day}_{band}.csv')
                
                if os.path.isfile(topsetfile):
                    df = pd.read_csv(topsetfile)
                    topsetids = np.array(df['id'])
                else:
                    topsetids = np.array([])

                # navigate to the appropriate directory
                if imager == 'difmap': inputdir = os.path.join(inparentdir, f'{model}_{day}_{band}')
                elif imager == 'difmap_geofit': inputdir = os.path.join(inparentdir, f'{model}_{day}_{band}_geofit')

                if os.path.isdir(inputdir):
                    if topsetids.shape[0] == 0:
                        warn(f"No topset info available for {inputdir}! Skipping...")
                    else:
                        # create filelist
                        filelist = []
                        for idval in topsetids:
                            if imager == 'difmap': filelist.append(os.path.join(inputdir, f'{model}_{day}_{band}_{idval:06}.fits') + '\n')
                            elif imager == 'difmap_geofit': filelist.append(os.path.join(inputdir, f'{model}_{day}_{band}_geofit_{idval:06}.fits') + '\n')

                        if imager == 'difmap': filelistname = f"{model}_{day}_{band}.filelist"
                        elif imager == 'difmap_geofit': filelistname = f"{model}_{day}_{band}_geofit.filelist"
                        with open(filelistname, 'w') as f:
                            f.writelines(filelist)

                        if imager == 'difmap': outpath = os.path.join(outparentdir, f'{model}_{day}_{band}')
                        elif imager == 'difmap_geofit': outpath = os.path.join(outparentdir, f'{model}_{day}_{band}_geofit')
                        os.mkdir(outpath)

                        # load into ehtim
                        fnames = np.genfromtxt(f'{filelistname}', dtype='str')

                        '''pool = Pool(proc)
                        imglist = list(pool.imap(partial(eh.image.load_fits, aipscc=True), fnames))
                        pool.close()'''

                        imglist = []
                        for fname in fnames:
                            imglist.append(eh.image.load_fits(fname, aipscc=True))
                
                        for img, idval in zip(imglist, topsetids):
                            img = img.regrid_image(targetfov=fov, npix=npix, interp=interp) # regrid image to restrict the FoV of CLEAN images
                            img = img.blur_gauss(beamparams=[beamsize,beamsize,0], frac=frac)
                            if imager == 'difmap': img.save_fits(os.path.join(outpath, f'{model}_{day}_{band}_{idval:06}.fits'))
                            elif imager == 'difmap_geofit': img.save_fits(os.path.join(outpath, f'{model}_{day}_{band}_geofit_{idval:06}.fits'))

                else:
                    warn(f'{inputdir} does not exist! Skipping...')
