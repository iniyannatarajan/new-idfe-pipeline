# -*- coding: utf-8 -*-

###############################################################################
# EHT Image Domain Feature Extraction Consolidation Script
# Compatible with Python version 3.6 and above
###############################################################################

###############################################
# imports

import os
import re
import sys
import copy
import glob
import json
import argparse
import ehtim as eh
import numpy as np
import pandas as pd
from scipy import ndimagei
from astropy.io import fits
from multiprocessing import Pool
from scipy.stats import circmean, circstd

import metronization as mt
from idfe.utils import *
#from idfe.clustering import *

###############################################
# define some inputs

# parameters to extract from the results
inparlist = ['D', 'PAori', 'W', 'fc', 'A'] # Input parameter names -- radius (but labelled D), position angle, width, frac. central brightness, brightness asymmetry
outparlist = ['R', 'PAori', 'W', 'fc', 'A'] # Output parameter names -- radius, position angle, width, frac. central brightness, brightness asymmetry
parunitlist = ['uas', 'deg', 'uas', 'unitless', 'unitless']
parlabellist = [r'Radius ($\mu$as)', r'Position angle ($^{\circ}$)', r'FWHM width ($\mu$as)', r'Frac. Central Brightness', r'Asymmetry']

'''inparlist = ['D', 'PAori', 'W', 'A'] # Input parameter names -- radius (but labelled D), position angle, width, brightness asymmetry
outparlist = ['R', 'PAori', 'W', 'A'] # Output parameter names -- radius, position angle, width, brightness asymmetry
parunitlist = ['uas', 'deg', 'uas', 'unitless']
parlabellist = [r'Radius ($\mu$as)', r'Position angle ($^{\circ}$)', r'FWHM width ($\mu$as)', r'Asymmetry']'''

'''inparlist = ['D', 'W', 'A'] # Input parameter names -- radius (but labelled D), width, brightness asymmetry
outparlist = ['R', 'W', 'A'] # Output parameter names -- radius, width, brightness asymmetry
parunitlist = ['uas', 'uas', 'unitless']
parlabellist = [r'Radius ($\mu$as)', r'FWHM width ($\mu$as)', r'Asymmetry']'''

npars = len(outparlist)

imagerlist = ['CLEAN', 'ehtim', 'SMILI', 'Themis']
scatterlist = ['deblur', 'nodeblur']
caliblist = ['hops', 'casa']
daylist = ['3598', '3599']
idfelist = ['REx', 'VIDA']

parentdir = '/DATA/EHT/20211024_SgrA_datasets_PaperIV/results/topset_results'
outputdir = 'Nov27_SGRA_R1'

varg1 = 1 # template parameter N
varg2 = 4 # template parameter M

stretch = False # always False unless IDFE is repeated with a new VIDA script

#####################################
# metronization settings

metronization_parentdir = '/DATA/EHT/20211024_SgrA_datasets_PaperIV/topset_images'

nproc = 96
metronmodelist = ['1nometron', '2expmode_permissive', '3expmode_moderate', '4expmode_strict']

# dimensions of Sgr A* images
sgra_dims = {'CLEAN': 128,
                'ehtim': 80,
                'SMILI': 75,
                'Themis': 128}

# rough values rounded to 3 decimal points (in uas)
sgra_cdelts = {'CLEAN': 2,
                'ehtim': 1.875,
                'SMILI': 2,
                'Themis': 1.575} # but the rescaling factor in read_fits will be 1.578125, so that the FoV is 202 uas, not 201.6 uas

# FoVs of Sgr A* images (product of the above two quantities)
sgra_fovs = {'CLEAN': 256,
                'ehtim': 150,
                'SMILI': 150,
                'Themis': 202} # for Themis, this is actually ~201.6, but we use this convention for ease of computation

target_dims = 150 # Npixels
target_fov = 150 # in uas

# load reference image -- this is constant throughout
refname = '/DATA/EHT/idfe-scripts/CLEAN_deblur_hops_3599_lo+hi_mean.fits' # 128x128 (256 uas FoV)
hdul = fits.open(refname)
ref = hdul[0].data[0,0]
hdul.close()

# maintain one reference image per imager whose images need to be aligned before cropping
ref_clean = ndimage.zoom(ref, 2) # 128p -> 256p with 256 uas FoV i.e. 1 uas resolution
ref_themis = ref_clean[sgra_fovs['CLEAN']//2-sgra_fovs['Themis']//2:sgra_fovs['CLEAN']//2+sgra_fovs['Themis']//2, \
        sgra_fovs['CLEAN']//2-sgra_fovs['Themis']//2:sgra_fovs['CLEAN']//2+sgra_fovs['Themis']//2] # crop to 150p (150 uas FoV)

def align(ref, img):
    vis1 = np.fft.fftn(ref)
    vis2 = np.fft.fftn(img)
    c    = np.real(np.fft.ifftn(vis1 * np.conj(vis2)))
    i, j = np.unravel_index(np.argmax(c), c.shape)
    return np.roll(np.roll(img, i, axis=0), j, axis=1)


def read_fits(fname, isehtim, imager='CLEAN'):
    '''
    Read a FITS file and return the image array. Used as wrapper for parallelization
    '''
    hdul = fits.open(fname)
    if isehtim:
        img = hdul[0].data
    else:
        img = hdul[0].data[0,0]
    hdul.close()

    if imager == 'CLEAN':
        img_rescaled = ndimage.zoom(img, 2) # 128p -> 256p (256 uas FoV)
        img_aligned = align(ref_clean, img_rescaled)
        img_final = img_aligned[sgra_fovs['CLEAN']//2-target_fov//2:sgra_fovs['CLEAN']//2+target_fov//2, \
                sgra_fovs['CLEAN']//2-target_fov//2:sgra_fovs['CLEAN']//2+target_fov//2] # crop to 150p (150 uas FoV)
    elif imager == 'ehtim':
        img_final = ndimage.zoom(img, 1.875) # 80p -> 150p (150 FoV)
    elif imager == 'SMILI':
        img_final = ndimage.zoom(img, 2) # 75p -> 150p (150 FoV)
    elif imager == 'Themis':
        img_rescaled = ndimage.zoom(img, 1.578125) # 128p -> 202p (202 FoV)
        img_aligned = align(ref_themis, img_rescaled)
        img_final = img_aligned[sgra_fovs['Themis']//2-target_fov//2:sgra_fovs['Themis']//2+target_fov//2, \
                sgra_fovs['Themis']//2-target_fov//2:sgra_fovs['Themis']//2+target_fov//2] # crop to 150p (150 uas FoV)

    return img_final


def wrap_toposign(img, ind, codeval, ngrid, hlower, hupper, pspan, hspan, threshold, hbirth, pbirth, hdepth):
    '''
    Wrap toposign for parallelization
    '''
    # add surveyid to the dictionary
    nthres = 20 # increase levels to 20
    thres_arr = np.arange(nthres) / nthres
    try:
        topores = mt.toposign(img, ngrid, threshold=thres_arr, hlower=hlower, hupper=hupper, pspan=pspan, hspan=hspan, hbirth=hbirth, pbirth=pbirth)
        topores['valid'] = True
    except IndexError:
        info(f'Image {ind}: {codeval} returned an IndexError. Moving on...')
        topores = {}
        topores['valid'] = False
    topores['id'] = codeval

    # Jan30 v5: further improve how pieces and holes are calculated
    pieces = 0
    holes = 0
    h_prev = 0
    holes_just_updated = 0
    for tind in range(nthres):
        #print('---------------------------------------------')
        #print(f'START FOR LOOP: At tind={tind} and threshold={thres_arr[tind]}, pieces={pieces}, holes={holes}, h_prev={h_prev}')
        if thres_arr[tind] >= threshold:
            p_curr, h_curr = topores[thres_arr[tind]]
            #print(f'p_curr={p_curr}, h_curr={h_curr} at threshold {thres_arr[tind]}')
            if h_curr > h_prev: # at least 1 new hole found; now check whether it persists
                #print(f'h_curr={h_curr} > h_prev={h_prev}; now check for persistence')
                # adjust hdepth according to the number of remaining elements in thres_arr
                if tind+hdepth < nthres:
                    tind_next = tind+hdepth-1
                else:
                    tind_next = nthres-1
                #print(f'Given hdepth={hdepth} and tind={tind}, tind_next={tind_next}')
                h_next = topores[thres_arr[tind_next]][1]
                
                #print(f'before: holes_just_updated = {holes_just_updated}')
                if h_next == h_curr:
                    if holes_just_updated:
                        holes_to_add = h_curr - holes_just_updated
                        #print(f'h_next={h_next} = h_curr={h_curr}; {holes} becomes {holes+holes_to_add} holes')
                        holes = holes + holes_to_add
                        holes_just_updated = holes_to_add
                    else:
                        #print(f'h_next={h_next} = h_curr={h_curr}; {holes} becomes {holes+h_curr} holes')
                        holes = holes + h_curr
                        holes_just_updated = h_curr                    
                elif h_next > h_curr:
                    holes_to_add = h_next - h_curr
                    #print(f'h_next={h_next} > h_curr={h_curr}; {holes} becomes {holes+holes_to_add} holes')
                    holes = holes + holes_to_add # add the difference to holes
                    holes_just_updated = holes_to_add
                elif h_next < h_curr:
                    #print(f'h_next={h_next} < h_curr={h_curr}; {holes} becomes {holes+h_next} holes')
                    holes = holes + h_next
                    holes_just_updated = h_next

                #print(f'after: holes_just_updated = {holes_just_updated}')
                h_prev = h_curr # set h_prev up for the next round

                if p_curr > pieces:
                    pieces = p_curr
            else:
                holes_just_updated = 0 # if holes was not updated in the previous thres, set to 0

        else:
            p_curr = topores[thres_arr[tind]][0]
            if p_curr > pieces:
                pieces = p_curr
  
    if holes == 0:
        topores['type'] = 'noring'
    elif holes == 1:
        if pieces == 1: topores['type'] = 'ring'
        elif pieces > 1: topores['type'] = 'noisyring'
    elif holes > 1:
        topores['type'] = 'multiring'

    return topores


def run_metronization(filelist, ngrid=32, hlower=None, hupper=None, pspan=2, hspan=0.7, threshold=0.2, hbirth=np.sqrt(2), pbirth=None, proc=8, isehtim=False, hdepth=2, imager='CLEAN'):
    '''
    Metronize a list of images
    '''
    fnames = np.genfromtxt(f'{filelist}', dtype='str')
    topsetcode = [int(re.findall('\d+',x)[-1]) for x in fnames]
    nimages = len(topsetcode)

    info(f'Metronizing files in {filelist}...')

    pool = Pool(proc)

    # Read all FITS images into a list of arrays
    argslist = [(fname, isehtim, imager) for fname in fnames]
    imglist = pool.starmap(read_fits, argslist)

    # Metronize all images
    argslist = [(img, ind, codeval, ngrid, hlower, hupper, pspan, hspan, threshold, hbirth, pbirth, hdepth) for img,ind,codeval in zip(imglist, range(nimages), topsetcode)]
    metronized = pool.starmap(wrap_toposign, argslist)

    pool.close()
   
    return metronized

#####################################

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

# create empty consolidated HDF5 file
consolidated_colnames = ['id']
consolidated_colnames = ['metronmode']
consolidated_colnames = ['metrontype']
consolidated_colnames = ['label']
consolidated_colnames = ['imager']
consolidated_colnames = ['scatter']
consolidated_colnames = ['calib']
consolidated_colnames = ['day']
consolidated_colnames = ['band']
for idfemethod in idfelist:
    for outpar,parunit in zip(outparlist, parunitlist):
        consolidated_colnames.append(f'{idfemethod}_{outpar}_{parunit}')

df_consolidated = pd.DataFrame(columns=consolidated_colnames)

# declare index for consolidated dataframe
index = 0

for imager in imagerlist:
    for scatter in scatterlist:
        for calib in caliblist:
            for day in daylist:

                # do a Themis check
                if imager == 'Themis':
                    if scatter != 'deblur':
                        continue
                    if calib == 'hops':
                        if day == '3598':
                            continue
                        else:
                            band = 'lo'
                    else:
                        band = 'lo+hi'
                else:
                    band = 'lo+hi'

                # pick filenames and labels
                rex_filename = f'{parentdir}/{imager}_{scatter}_{calib}_{day}_{band}_REx.h5'
                rex_h5colname_suffix = f'{imager}_{scatter}_{calib}_{day}_{band}'
                vida_filename = f'{parentdir}/{imager}_{scatter}_{calib}_{day}_{band}_VIDA_template14_fc.csv'

                # set output hdf5 colname
                dataset_label = f'{imager}_{scatter}_{calib}_{day}_{band}'

                # Read in the fitted param values for rex and vida for each FITS file in each dataset and mode
                df_rex = pd.read_hdf(rex_filename, 'parameters')
                #df_rex.sort_values(by=['id'], inplace=True) # sort columns by topset id
                topsetidrex_arr = np.array(df_rex['id'])
                nimages = topsetidrex_arr.shape[0]
            
                df_output_pars = pd.DataFrame([])
                df_output_pars['id'] = topsetidrex_arr
                # read in all the relevant parameters from REx HDF5 output file
                for inpar,outpar,parunit in zip(inparlist, outparlist, parunitlist):
                  if inpar == 'D':
                    df_output_pars[f'REx_{outpar}_{parunit}'] = np.array(df_rex[f'{inpar}_{rex_h5colname_suffix}'])/2. # convert diameter to radius
                  elif inpar != 'PAori':
                    df_output_pars[f'REx_{outpar}_{parunit}'] = np.array(df_rex[f'{inpar}_{rex_h5colname_suffix}'])
                  elif inpar == 'PAori':
                    # adjust the position angle range output from REx
                    tmparr = np.array(df_rex[f'{inpar}_{rex_h5colname_suffix}'])
                    tmparr[np.where(tmparr>180)] = tmparr[np.where(tmparr>180)]%360-360
                    tmparr[np.where(tmparr<=-180)] = tmparr[np.where(tmparr<=-180)]%360
                    df_output_pars[f'REx_{outpar}_{parunit}'] = tmparr

                # read in the VIDA output file
                vida_allpars = np.genfromtxt(vida_filename, dtype=None, delimiter=',', encoding=None)

                #############################################################
                # read relevant VIDA parameters and convert them to REx equivalent values
                for outpar,parunit in zip(outparlist,parunitlist):
                    if outpar == 'R':
                        df_output_pars[f'VIDA_{outpar}_{parunit}'] = np.array(vida_allpars[1:,0].astype(float)) # radius
                    elif outpar == 'W':
                        df_output_pars[f'VIDA_{outpar}_{parunit}'] = 2*np.sqrt(2*np.log(2))*vida_allpars[1:,1].astype(float) # convert sigma to ring width
                    elif outpar == 'PAori':
                        if varg1 == 0:
                            df_output_pars[f'VIDA_{outpar}_{parunit}'] = np.rad2deg(np.pi/2 - np.array(vida_allpars[1:,6].astype(float))) # position angle
                        elif varg1 == 1:
                            df_output_pars[f'VIDA_{outpar}_{parunit}'] = np.rad2deg(np.pi/2 - np.array(vida_allpars[1:,8].astype(float))) # position angle

                        # adjust position angle values to be b/w -180 to 180 degrees
                        tmparr = df_output_pars[f'VIDA_{outpar}_{parunit}'].to_numpy()
                        tmparr[np.where(tmparr>180)] = tmparr[np.where(tmparr>180)]%360-360
                        tmparr[np.where(tmparr<=-180)] = tmparr[np.where(tmparr<=-180)]%360
                        df_output_pars[f'VIDA_{outpar}_{parunit}'] = tmparr
                    elif outpar == 'fc':
                        if varg1 == 0:
                            if stretch:
                                df_output_pars[f'VIDA_{outpar}_{parunit}'] = vida_allpars[1:,10].astype(float) # floor; NOT converted to fc
                            else:
                                df_output_pars[f'VIDA_{outpar}_{parunit}'] = vida_allpars[1:,16].astype(float) # floor; NOT converted to fc
                        elif varg1 == 1:
                            if stretch:
                                df_output_pars[f'VIDA_{outpar}_{parunit}'] = vida_allpars[1:,12].astype(float) # floor; NOT converted to fc
                            else:
                                df_output_pars[f'VIDA_{outpar}_{parunit}'] = vida_allpars[1:,18].astype(float) # floor; NOT converted to fc
                    elif outpar == 'A':
                        if varg1 == 0:
                            df_output_pars[f'VIDA_{outpar}_{parunit}'] = vida_allpars[1:,2].astype(float)/2. # brightness asymmetry (divide by 2 to match REx convention)
                        elif varg1 == 1:
                            df_output_pars[f'VIDA_{outpar}_{parunit}'] = vida_allpars[1:,4].astype(float)/2. # brightness asymmetry
                #############################################################

                # replace infinities with NaN
                df_output_pars.replace([np.inf, -np.inf], np.nan, inplace=True)

                ### df_output_pars.sort_values(by=['id'], inplace=True) # sort columns by topset id ::: COLUMNS SHOULD ALREADY BE SORTED
                df_output_pars.to_hdf(f'{outputdir}/IDFE_{dataset_label}_results.h5', 'parameters', mode='w', complevel=9, format='table')
                info(f'Relevant parameters saved to {outputdir}/IDFE_{dataset_label}_results.h5')
            
                ################################################################
                # set parameters for 3 metronization modes
                for metronmode in metronmodelist:
                    #df_metronized = copy.deepcopy(df_output_pars)
                    # set optimal metronization parameters
                    if metronmode == '2permissive':
                        ngrid_dict = {'CLEAN': 32,
                                      'ehtim': 20,
                                      'SMILI': 15,
                                      'Themis': 64}
                        pspan_dict = {'CLEAN': 2,
                                      'ehtim': 2,
                                      'SMILI': 2,
                                      'Themis': 2}
                        hspan_dict = {'CLEAN': 0.7,
                                      'ehtim': 0.7,
                                      'SMILI': 0.7,
                                      'Themis': 0.7}
                        thres_dict = {'CLEAN': 0.2,
                                     'ehtim': 0.2,
                                     'SMILI': 0.2,
                                     'Themis': 0.2}
                    elif metronmode == '3moderate':
                        ngrid_dict = {'CLEAN': 32,
                                      'ehtim': 20,
                                      'SMILI': 15,
                                      'Themis': 64}
                        pspan_dict = {'CLEAN': 2,
                                      'ehtim': 2,
                                      'SMILI': 2,
                                      'Themis': 2}
                        hspan_dict = {'CLEAN': 0.8,
                                      'ehtim': 0.8,
                                      'SMILI': 0.8,
                                      'Themis': 0.8}
                        thres_dict = {'CLEAN': 0.3,
                                     'ehtim': 0.3,
                                     'SMILI': 0.3,
                                     'Themis': 0.3}
                    elif metronmode == '4strict':
                        ngrid_dict = {'CLEAN': 32,
                                      'ehtim': 20,
                                      'SMILI': 15,
                                      'Themis': 64}
                        pspan_dict = {'CLEAN': 2,
                                      'ehtim': 2,
                                      'SMILI': 2,
                                      'Themis': 2}
                        hspan_dict = {'CLEAN': 0.8,
                                      'ehtim': 0.8,
                                      'SMILI': 0.9,
                                      'Themis': 0.8}
                        thres_dict = {'CLEAN': 0.4,
                                     'ehtim': 0.4,
                                     'SMILI': 0.3,
                                     'Themis': 0.4}

                    # metronize and remove rows depending on the mode
                    if metronmode != '1nometron':
                        
                        # metronize
                        metronization_inputdir = os.path.join(metronization_parentdir, f'images_{scatter}_{imager}', f'{calib}_{day}')

                        metronization_dirlabel = f'{scatter}_{imager}_{calib}_{day}_{band}' 

                        # create input file list
                        #info(f'Creating list of input FITS images...')
                        filelist = f'{metronization_dirlabel}.filelist'
                        createlistcmd = f"readlink -f {metronization_inputdir}/*.fits >{filelist}"
                        #info(createlistcmd)
                        os.system(createlistcmd)

                        # account for differing FITS dimensions of ehtim and other imagers
                        # set isehtim=True for both ehtim and Themis, since the corresponding FITS images have similar Ndims
                        if imager in ['ehtim', 'Themis']:
                            isehtim = True
                        elif imager in ['CLEAN', 'SMILI']:
                            isehtim = False

                        # metronize the images
                        info(f'Metronizing in mode "{metronmode}: ngrid={ngrid_dict[imager]}, pspan={pspan_dict[imager]}, hspan={hspan_dict[imager]}, threshold={thres_dict[imager]}"...')
                        toporeslist = run_metronization(filelist, ngrid=ngrid_dict[imager], hlower=None, hupper=None, pspan=pspan_dict[imager], \
                                hspan=hspan_dict[imager], threshold=thres_dict[imager], proc=nproc, isehtim=isehtim, imager=imager)

                        with open(f'{outputdir}/{dataset_label}_metronmode_{metronmode}.json', 'w') as jsonfile:
                            json.dump(toporeslist, jsonfile)

                        # remove filelist file to avoid clutter
                        os.system(f'rm {filelist}')
                        
                        # use toposign results to collect list of topset ids to exclude
                        #exclude_ids = []
                        topores_types = []
                        for toporesdict in toporeslist:
                            topores_types.append(toporesdict['type'])
                            #if toporesdict['type'] == 'noring':
                            #    exclude_ids.append(toporesdict['id'])

                        #metronize_exclude_frac = len(exclude_ids)/len(toporeslist)
                        nimages = len(toporeslist) #- len(exclude_ids)

                        # remove the above ids from df_metronized
                        #df_metronized = df_metronized[~df_metronized.id.isin(exclude_ids)]

                    ################################################################
                    # compute the statistical quantities necessary for the consolidated output

                    # create temporary dataframe and populate it
                    tmpdf = pd.DataFrame()
                    tmpdf['id'] = df_output_pars['id'].astype(int)
                    tmpdf['label'] = [f'{dataset_label}_{metronmode}']*nimages
                    tmpdf['metronmode'] = [metronmode]*nimages
                    if metronmode == '1nometron':
                        tmpdf['metrontype'] = ['ring']*nimages
                    else:
                        tmpdf['metrontype'] = topores_types
                    tmpdf['imager'] = [imager]*nimages
                    tmpdf['calib'] = [calib]*nimages
                    tmpdf['scatter'] = [scatter]*nimages
                    tmpdf['day'] = [day]*nimages
                    tmpdf['band'] = [band]*nimages
                    for idfemethod in idfelist:
                        for outpar,parunit in zip(outparlist, parunitlist):
                            tmpdf[f'{idfemethod}_{outpar}_{parunit}'] = df_output_pars[f'{idfemethod}_{outpar}_{parunit}']

                    # append to consolidated dataframe
                    df_consolidated = df_consolidated.append(tmpdf)

# set index for dataframe
df_consolidated = df_consolidated.set_index(np.arange(df_consolidated.shape[0]))

# save the consolidated dataframe
df_consolidated.to_hdf(f'{outputdir}/IDFE_SgrA_consolidated.h5', 'parameters', mode='w', complevel=9, format='table')
info(f'Consolidated DataFrame saved to {outputdir}/IDFE_SgrA_consolidated.h5')
