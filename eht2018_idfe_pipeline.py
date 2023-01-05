# -*- coding: utf-8 -*-

###############################################################################
# EHT Image Domain Feature Extraction Pipeline
# Compatible with Python version 3.6 and above
# To be run in the directory where the output files are required to be located 
###############################################################################

###############################################
# imports

import os
import sys
import glob
import json
import argparse
import ehtim as eh
import pandas as pd

from idfe.clustering import *
from idfe.idfealg import *
from idfe.plotting import *

###############################################
# define some inputs not present in argparse

# parameters to extract from the results and plot
rexparlist = ['D', 'PAori', 'W', 'fc', 'A'] # REx parameters -- diameter, position angle, width, frac. central brightness, brightness asymmetry
outparlist = ['R', 'PAori', 'W', 'fc', 'A'] # Output parameters -- radius, position angle, width, frac. central brightness, brightness asymmetry
parlabellist = [r'Radius ($\mu$as)', r'Position angle ($^{\circ}$)', r'FWHM width ($\mu$as)', r'Frac. Central Brightness', r'Asymmetry']
npars = len(outparlist)

###################################################################################

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-p', '--proc', type=int, default=2, help='Number of processors to use')
    p.add_argument('-i', '--inputdir', type=str, required=True, help='Input directory containing FITS images')
    p.add_argument('-d', '--dataset', type=str, required=True, help='Dataset name for suffixing column names from the HDF5 REx output')
    p.add_argument('--isehtim', action='store_true', help='Specify if the image is from eht-imaging')
    p.add_argument('-ng', '--ngrid', type=int, default=16, help='Number of grids to divide the image into during metronization')
    p.add_argument('-pl', '--plength', type=float, default=2, help='Robustness parameter for metronization (tolerance length for disconnected pieces)')
    p.add_argument('-hl', '--hlength', type=float, default=4, help='Robustness parameter for metronization (tolerance length for holes)')
    p.add_argument('--isclean', action='store_true', help='Specify if the image is from CLEAN')
    p.add_argument('-b', '--beaminuas', type=float, default=20, help='Beamsize for blurring CLEAN images in uas')
    p.add_argument('-v', '--vida', type=str, required=True, help='VIDA script to be used')
    p.add_argument('-t', '--template', nargs='+', type=int, required=True, help='VIDA template parameters')
    p.add_argument('-s', '--stride', type=int, default=200, help='Number of images after which to checkpoint')
    p.add_argument('-m', '--model', type=str, default='FLOOR', choices=['FLOOR', 'GFLOOR'], help='VIDA floor model')
    p.add_argument('--stretch', action='store_true', help='Turn ellipticity on; without this the ring will be assumed circular')
    p.add_argument('--restart', action='store_true', help='Restart from the last checkpoint contained in the CSV file specified with --out')
    p.add_argument('-e', '--execmode', type=str, default='all', choices=['all', 'metron', 'idfe', 'plot'], help='Execution mode in which to run the pipeline')
    return p

################################ Start the pipeline ###############################
def main(args):

  # define some REx and VIDA related variables
  varg1 = args.template[0]
  varg2 = args.template[1]
  rex_outfile = f'{args.dataset}_REx.h5'
  vida_outfile = f'{args.dataset}_VIDA_template{varg1}{varg2}.csv'

  if args.execmode in ['all', 'metron', 'idfe']:

    ##### Read in FITS images #####
   
    dirname = args.inputdir.split('/')[-1]
    filelist = dirname+'.filelist'

    info(f'Creating list of input FITS images...')
    createlistcmd = f"readlink -f {args.inputdir}/*.fits >{filelist}"
    info(createlistcmd)
    os.system(createlistcmd)

    ######################################
    # Clustering (metronization etc.)
    
    '''metron_res = run_metronization(filelist, ngrid=args.ngrid, plength=args.plength, hlength=args.hlength, proc=args.proc, isehtim=args.isehtim)
    with open('metron.json', 'w') as jsonfile:
        json.dump(metron_res, jsonfile)

    info(f'Metronization output saved to metron.json')'''

    ## TODO:: if execmode is all or idfe, create a new filelist as replacement for args.dataset for the calls to REx and VIDA.
    
  ##########################################################
  # Image domain feature extraction using REx and VIDA
  if args.execmode in ['all', 'idfe']:
        
    info(f'Performing IDFE using REx...')
    runrex(filelist, args.dataset, rex_outfile, args.isclean, proc=args.proc, beaminuas=args.beaminuas)
    
    info(f'Performing IDFE using VIDA...')
    runvida(args.vida, filelist, vida_outfile, proc=args.proc, arg1=varg1, arg2=varg2, stride=args.stride, stretch=args.stretch, restart=args.restart, model=args.model)

  #############################################################
  # Aggregate results, save output arrays and generate plots
  if args.execmode in ['all', 'plot']:

    info(f'Extract relevant parameters from REx and VIDA results...')
    # Read in the fitted param values for rex and vida for each FITS file in each dataset and mode
    df_rex = pd.read_hdf(rex_outfile, 'parameters')
    #df_rex.sort_values(by=['id'], inplace=True) # sort columns by topsetcode
    topsetcoderex_arr = np.array(df_rex['id'])
    nimages = topsetcoderex_arr.shape[0]
    
    df_output_pars = pd.DataFrame([])
    df_output_pars['id'] = topsetcoderex_arr
    # read in all the relevant parameters from REx HDF5 output file
    for rexpar,outpar in zip(rexparlist, outparlist):
      if rexpar == 'D':
        df_output_pars[f'REx_{outpar}_{args.dataset}'] = np.array(df_rex[f'{rexpar}_{args.dataset}'])/2. # convert diameter to radius
      elif rexpar != 'PAori':
        df_output_pars[f'REx_{outpar}_{args.dataset}'] = np.array(df_rex[f'{rexpar}_{args.dataset}'])
      else:
        # adjust the position angle range output from REx
        tmparr = np.array(df_rex[f'{rexpar}_{args.dataset}'])
        tmparr[np.where(tmparr>180)] = tmparr[np.where(tmparr>180)]%360-360
        tmparr[np.where(tmparr<=-180)] = tmparr[np.where(tmparr<=-180)]%360
        df_output_pars[f'REx_{outpar}_{args.dataset}'] = tmparr

    # read in the necessary parameters from VIDA output
    if args.stretch:
        vida_allpars = np.genfromtxt(vida_outfile, dtype=None, delimiter=',', encoding=None)
    else:
        vida_allpars = np.genfromtxt(vida_outfile.replace('.csv', '_fc.csv'), dtype=None, delimiter=',', encoding=None)

    vida_pars = np.zeros((nimages, npars))

    ##################################################
    # read each parameter of interest from VIDA; this should match the parameter list at the beginning of the script
    vida_pars[:,0] = np.array(vida_allpars[1:,0].astype(float)) # radius
    vida_pars[:,2] = 2*np.sqrt(2*np.log(2))*vida_allpars[1:,1].astype(float) # convert sigma to ring width

    if varg1 == 0:
        vida_pars[:,1] = np.rad2deg(np.pi/2 - np.array(vida_allpars[1:,6].astype(float))) # position angle
        if args.stretch:
            vida_pars[:,3] = vida_allpars[1:,10].astype(float) # floor; NOT converted to fractional central brightness
        else:
            vida_pars[:,3] = vida_allpars[1:,16].astype(float) # fractional central brightness
        vida_pars[:,4] = vida_allpars[1:,2].astype(float)/2. # brightness asymmetry (divide by 2 to match REx convention)

    elif varg1 == 1:
        vida_pars[:,1] = np.rad2deg(np.pi/2 - np.array(vida_allpars[1:,8].astype(float))) # position angle
        if args.stretch:
            vida_pars[:,3] = vida_allpars[1:,12].astype(float) # floor; NOT converted to fractional central brightness
        else:
            vida_pars[:,3] = vida_allpars[1:,18].astype(float) # fractional central brightness
        vida_pars[:,4] = vida_allpars[1:,4].astype(float)/2. # brightness asymmetry

    # adjust position angle values to be b/w -180 to 180 degrees
    tmparr = vida_pars[:,1]
    tmparr[np.where(tmparr>180)] = tmparr[np.where(tmparr>180)]%360-360
    tmparr[np.where(tmparr<=-180)] = tmparr[np.where(tmparr<=-180)]%360
    vida_pars[:,1] = tmparr

    ##################################################
   
    # comment the following for now, since the same FITS images will be passed through both REx and VIDA; no NaNs necessary
    '''# extract topset codes from the input fits file names in the VIDA output CSV file
    topsetcodevida_arr = np.array([int(re.findall('\d+',x)[-1]) for x in vida_allpars[1:,-1]])
    # compare vida_allpars[1:,-1] with topsetcoderex_arr and insert np.nan where necessary
    for ii in np.arange(nimages):
      if topsetcoderex_arr[ii] not in topsetcodevida_arr:
          vida_pars = np.insert(vida_pars, ii, np.nan, axis=0)'''
    
    # add columns to the output DataFrame
    for ii in np.arange(npars):
        df_output_pars[f'VIDA_{outparlist[ii]}_{args.dataset}'] = vida_pars[:,ii]

    #df_output_pars.sort_values(by=['id'], inplace=True) # sort columns by topsetcode
    df_output_pars.to_hdf(f'IDFE_{args.dataset}_results.h5', 'parameters', mode='w', complevel=9, format='table')
    info(f'Relevant parameters saved to IDFE_{args.dataset}_results.h5')

    ################################################################
    # Generate plots
    
    info('Generating plots...')
    plot_hist_pars(df_output_pars, outparlist, parlabellist, args.dataset, bins=10, fontsize=24)
    plot_images_vs_pars(df_output_pars, outparlist, parlabellist, args.dataset, fontsize=24)
    plot_scatter_hists(df_output_pars, outparlist, parlabellist, args.dataset, bins=10, fontsize=24)

if __name__ == '__main__':
  args = create_parser().parse_args()
  ret = main(args)
  sys.exit(ret)
