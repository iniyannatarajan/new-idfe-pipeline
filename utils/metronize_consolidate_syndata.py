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
from scipy import ndimage
from astropy.io import fits
from multiprocessing import Pool
from scipy.stats import circmean, circstd

import metronization as mt
from idfe.utils import *

###############################################
# define some inputs

# parameters to extract from the results
rexparlist = ['D', 'PAori', 'W', 'fc', 'A'] # diameter, position angle, width, frac. central brightness, brightness asymmetry
vidaparlist = ['r0', u'\u03bes', u'\u03c3', 'floor', 's'] # radius, position angle, sigma for width, floor, brightness asymmetry
parunitlist = ['uas', 'deg', 'uas', 'NA', 'NA']
parlabellist = [r'Diameter ($\mu$as)', r'Position angle ($^{\circ}$)', r'FWHM width ($\mu$as)', r'Frac. cen. brightness', r'Brightness asymmetry']

npars = len(rexparlist)

# to loop through datasets
imagerlist = ['Comrade', 'smili', 'difmap', 'THEMIS', 'ehtim']
netcal = 'netcal' # only netcal data always
modellist = ['ring', 'cres000', 'cres090', 'cres180', 'cres270', 'dblsrc', 'disk', 'edisk', 'point+disk', 'point+edisk', 'grmhd', 'ecres000', 'ecres045', 'ecres090', 'ecres315']
daylist = ['3644', '3647']
caliblist = ['hops', 'casa']
bandlist = ['b1', 'b2', 'b3', 'b4']
smilibandlist = ['b1+2', 'b3+4', 'b1+2+3+4']
themisbandlist = ['b1b2', 'b3b4', 'b1b2b3b4']
idfelist = ['REx', 'VIDA']

parentdir = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE'
outputdir = 'metronized'

# metronization settings
#metronization_parentdir = '/DATA/EHT/20211405_MoD_datasets_PaperIV'
nproc = 48
metronmodelist = ['1nometron', '2expmode_permissive', '3expmode_moderate', '4expmode_strict']
hdepth = 2

# get image parameters for each imager from a representative image
repimgdict = {'Comrade': os.path.join(parentdir, 'Comrade', netcal, 'ring_3644_b1', 'ring_3644_b1_image_0001.fits'),
                'smili': os.path.join(parentdir, 'smili', 'ring_3644_b1', 'ring_3644_b1_000001.fits'),
                'difmap': os.path.join(parentdir, 'difmap', 'ring_3644_b1', 'ring_3644_b1_000001.fits'),
                'THEMIS': os.path.join(parentdir, 'THEMIS', netcal, 'ring_3644_b1', 'ring_3644_b1_001.fits'),
                'ehtim': os.path.join(parentdir, 'ehtim', 'ring_3644_hops-b1', 'ring_3644_hops-b1_0000001.fits')}

repimgdims = {}
repimgcdelts = {}
repimgfovs = {}

for key in repimgdict.keys():
    hdul = fits.open(repimgdict[key])
    repimgdims[key] = hdul[0].header['NAXIS1']
    repimgcdelts[key] = np.abs(hdul[0].header['CDELT1']*3600*1e6) # convert from degrees to uas and get abs value in case cdelt is -ve
    repimgfovs[key] = np.ceil(repimgdims[key]*repimgcdelts[key])
    hdul.close()

'''# dimensions of Sgr A* images
sgra_dims = {'CLEAN': 128,
                'ehtim': 80,
                'SMILI': 75,
                'Themis': 256}

# NB: THESE FACTORS ARE USED TO RESCALE AN IMAGE
sgra_cdelts = {'CLEAN': 2,
                'ehtim': 1.875,
                'SMILI': 2,
                'Themis': 0.7843137254903879}

# FoVs of Sgr A* images (product of the above two quantities)
sgra_fovs = {'CLEAN': 256,
                'ehtim': 150,
                'SMILI': 150,
                'Themis': 201} #200.7843137255393} rounding up'''

target_dims = 128 # Npixels
target_fov = 128 # in uas

print(repimgdict)
print(repimgdims)
print(repimgcdelts)
print(repimgfovs)

sys.exit(0)

# load reference image -- this is constant throughout
refimgname = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/results/consolidate/smili_ring_3644_hops_b1_mean.fits' # [64x64, FoV: 128 uas; native smili resolution]
hdul = fits.open(refimgname)
refimg = ndimage.zoom(hdul[0].data, 2) # zoom the reference image to a pixel resolution of 1 uas i.e. 64p -> 128p
hdul.close()


def align(ref, img):
    vis1 = np.fft.fftn(ref)
    vis2 = np.fft.fftn(img)
    c    = np.real(np.fft.ifftn(vis1 * np.conj(vis2)))
    i, j = np.unravel_index(np.argmax(c), c.shape)
    return np.roll(np.roll(img, i, axis=0), j, axis=1)


def read_fits(fname, has2dims, imager='smili'):
    '''
    Read a FITS file and return the image array. Used as wrapper for parallelization
    '''
    hdul = fits.open(fname)
    if has2dims:
        img = hdul[0].data
    else:
        img = hdul[0].data[0,0]
    hdul.close()

    if imager == 'Comrade':
        img_rescaled = ndimage.zoom(img, repimgfovs[imager])
    elif imager == 'difmap':
        img_rescaled = ndimage.zoom(img, repimgfovs['difmap']) # 128p -> 256p (256 uas FoV)
        img_aligned = align(ref_clean, img_rescaled)
        img_final = img_aligned[repimgfovs['difmap']//2-target_fov//2:repimgfovs['difmap']//2+target_fov//2, \
                repimgfovs['difmap']//2-target_fov//2:repimgfovs['difmap']//2+target_fov//2] # crop to 150p (150 uas FoV)
    elif imager == 'ehtim':
        img_final = ndimage.zoom(img, repimgcdelts['ehtim']) # 80p -> 150p (150 FoV)
    elif imager == 'SMILI':
        img_final = ndimage.zoom(img, repimgcdelts['SMILI']) # 75p -> 150p (150 FoV)
    elif imager == 'Themis':
        #img_rescaled = ndimage.zoom(img, sgra_cdelts['Themis']) # 128p -> 202p (202 FoV)
        #img_aligned = align(ref_themis, img_rescaled)
        img_aligned = align(ref_clean, img)
        img_final = img_aligned[sgra_fovs['CLEAN']//2-target_fov//2:sgra_fovs['CLEAN']//2+target_fov//2, \
                sgra_fovs['CLEAN']//2-target_fov//2:sgra_fovs['CLEAN']//2+target_fov//2] # crop to 150p (150 uas FoV)

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


def run_metronization(filelist, ngrid=32, hlower=None, hupper=None, pspan=2, hspan=0.7, threshold=0.2, hbirth=np.sqrt(2), pbirth=None, proc=8, has2dims=False, hdepth=2, imager='CLEAN'):
    '''
    Metronize a list of images
    '''
    fnames = np.genfromtxt(f'{filelist}', dtype='str')
    topsetcode = [int(re.findall('\d+',x)[-1]) for x in fnames]
    nimages = len(topsetcode)

    #info(f'Metronizing files in {filelist}...')

    pool = Pool(proc)

    # Read all FITS images into a list of arrays
    argslist = [(fname, has2dims, imager) for fname in fnames]
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
consolidated_colnames = ['calib']
consolidated_colnames = ['scatter']
consolidated_colnames = ['day']
consolidated_colnames = ['grmhd']
consolidated_colnames = ['spin']
consolidated_colnames = ['Rh']
consolidated_colnames = ['inc']
consolidated_colnames = ['band']
for idfemethod in idfelist:
    for outpar,parunit in zip(outparlist, parunitlist):
        consolidated_colnames.append(f'{idfemethod}_{outpar}_{parunit}')

df_consolidated = pd.DataFrame(columns=consolidated_colnames)

# declare index for consolidated dataframe
index = 0

for imager in imagerlist:
    for scatter in scatterlist:
        for day in daylist:
            for grmhd in grmhdlist:
                for spin in spinlist:
                    for Rh in Rhlist:
                        for inc in inclist:

                            info(f'imager={imager}, scatter={scatter}, day={day}, grmhd={grmhd}, spin={spin}, Rh={Rh}, inc={inc}')
                            # do only the combined day analysis for Themis
                            if imager == 'Themis': 
                                if day != '3598+3599':
                                    info(f'imager={imager} but day={day}. Skipping!')
                                    continue
                                if grmhd != 'SANE':
                                    info(f'imager={imager} but grmhd={grmhd}. Skipping!')
                                    continue
                                if scatter != 'deblur':
                                    info(f'imager={imager} but scatter={scatter}. Skipping!')
                                    continue
                            # do the combined day anaylsis only for Themis
                            if imager in ['CLEAN', 'ehtim', 'SMILI']:
                                if day == '3598+3599':
                                    info(f'day={day} but imager={imager}. Skipping!')
                                    continue

                            # set imagerdir
                            imagerdir = os.path.join(parentdir, f'{imager}')

                            if imager == 'CLEAN':
                                rex_filename = f'{imagerdir}/{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO+HI_REx.h5'
                                rex_h5colname_suffix = f'{imager}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO'
                                vida_filename = f'{imagerdir}/{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO+HI_VIDA_template14_fc.csv'
                            elif imager == 'ehtim':
                                rex_filename = f'{imagerdir}/{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO_REx.h5'
                                rex_h5colname_suffix = f'{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO'
                                vida_filename = f'{imagerdir}/{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO_VIDA_template14_fc.csv'
                            elif imager == 'SMILI':
                                rex_filename = f'{imagerdir}/{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO+HI_REx.h5'
                                # account for the interruption and change in H5 colname starting from 3598/deblur/SANE/a+0.94/Rh160/i90
                                if day == '3598' and scatter == 'deblur':
                                    if grmhd == 'SANE' and spin == 'a+0.94' and Rh == 'Rh160' and inc == 'i90':
                                        rex_h5colname_suffix = f'{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO+HI'
                                    else:
                                        rex_h5colname_suffix = f'{imager}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO+HI'
                                else:
                                    rex_h5colname_suffix = f'{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO+HI'
                                vida_filename = f'{imagerdir}/{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO+HI_VIDA_template14_fc.csv'
                            elif imager == 'Themis':
                                rex_filename = f'{imagerdir}/{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_lo+hi_REx.h5'
                                rex_h5colname_suffix = f'{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_lo+hi'
                                vida_filename = f'{imagerdir}/{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_lo+hi_VIDA_template14_fc.csv'
            
                            # set output hdf5 colname
                            dataset_label = f'{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_{band}'
            
                            # Read in the fitted param values for rex and vida for each FITS file in each dataset and mode
                            df_rex = pd.read_hdf(rex_filename, 'parameters')
                            #df_rex.sort_values(by=['id'], inplace=True) # sort columns by topset id
                            topsetidrex_arr = np.array(df_rex['id'], dtype=int)
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
                                if metronmode == '2expmode_permissive':
                                    ngrid = 25
                                    pspan = 2
                                    hspan = 1.6
                                    thres = 0.1
                                    hbirth = 2
                                elif metronmode == '3expmode_moderate':
                                    ngrid = 25
                                    pspan = 2
                                    hspan = 1.6
                                    thres = 0.2
                                    hbirth = 2
                                elif metronmode == '4expmode_strict':
                                    ngrid = 25
                                    pspan = 2
                                    hspan = 1.6
                                    thres = 0.3
                                    hbirth = 2
            
                                # metronize and remove rows depending on the mode
                                if metronmode != '1nometron':
                                    
                                    # metronize
                                    metronization_inputdir = os.path.join(metronization_parentdir, imager)
            
                                    if imager in ['CLEAN', 'ehtim']:
                                        metronization_dirlabel = f'{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_LO'
                                    elif imager == 'SMILI':
                                        metronization_dirlabel = f'{grmhd}_{spin}_{Rh}_{inc}_{day}_LO+HI'
                                    elif imager == 'Themis':
                                        metronization_dirlabel = f'{imager}_{scatter}_{calib}_{day}_{grmhd}_{spin}_{Rh}_{inc}_lo+hi'

                                    # assign subdir
                                    if imager != 'Themis':
                                        subdir = os.path.join(metronization_inputdir, metronization_dirlabel)
                                    elif imager == 'Themis':
                                        subdir = os.path.join(metronization_inputdir, 'deblur_3598+3599_hops/images', metronization_dirlabel)

                                    # create input file list
                                    #info(f'Creating list of input FITS images...')
                                    filelist = f'{metronization_dirlabel}.filelist'
                                    if imager == 'CLEAN':
                                        if scatter == 'deblur':
                                            createlistcmd = f"readlink -f {subdir}/*_dsct_im.fits >{filelist}"
                                        elif scatter == 'nodeblur':
                                            createlistcmd = f"readlink -f {subdir}/*_sct_im.fits >{filelist}"
                                    if imager in ['ehtim', 'SMILI']:
                                        if scatter == 'deblur':
                                            createlistcmd = f"readlink -f {subdir}/*_dsct.fits >{filelist}"
                                        elif scatter == 'nodeblur':
                                            createlistcmd = f"readlink -f {subdir}/*_sct.fits >{filelist}"
                                    if imager == 'Themis':
                                        createlistcmd = f"readlink -f {subdir}/*.fits >{filelist}"                                  
                                    #info(createlistcmd)
                                    os.system(createlistcmd)
            
                                    # account for differing FITS dimensions of ehtim and other imagers
                                    # set has2dims=True for both ehtim and Themis, since the corresponding FITS images have similar Ndims
                                    if imager in ['ehtim', 'Themis']:
                                        has2dims = True
                                    elif imager in ['CLEAN', 'SMILI']:
                                        has2dims = False
            
                                    # metronize the images
                                    info(f'Metronizing in mode "{metronmode}: ngrid={ngrid}, pspan={pspan}, hspan={hspan}, threshold={thres}, hbirth={hbirth}"...')
                                    toporeslist = run_metronization(filelist, ngrid=ngrid, hlower=None, hupper=None, pspan=pspan, \
                                            hspan=hspan, threshold=thres, hbirth=hbirth, proc=nproc, has2dims=has2dims, hdepth=hdepth, imager=imager)
            
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
                                tmpdf['grmhd'] = [grmhd]*nimages
                                tmpdf['spin'] = [spin]*nimages
                                tmpdf['Rh'] = [Rh]*nimages
                                tmpdf['inc'] = [inc]*nimages                               
                                tmpdf['band'] = [band]*nimages
                                for idfemethod in idfelist:
                                    for outpar,parunit in zip(outparlist, parunitlist):
                                        tmpdf[f'{idfemethod}_{outpar}_{parunit}'] = df_output_pars[f'{idfemethod}_{outpar}_{parunit}']
            
                                # append to consolidated dataframe
                                df_consolidated = df_consolidated.append(tmpdf)

# set index for dataframe
df_consolidated = df_consolidated.set_index(np.arange(df_consolidated.shape[0]))

# save the consolidated dataframe
df_consolidated.to_hdf(f'{outputdir}/IDFE_MoD_consolidated_CLEAN.h5', 'parameters', mode='w', complevel=9, format='table')
info(f'Consolidated DataFrame saved to {outputdir}/IDFE_MoD_consolidated_CLEAN.h5')
