import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from idfe.utils import *
from idfe.idfealg import *

# define some variables
imagerlist =  ['Comrade', 'smili', 'difmap'] # 'THEMIS' and 'ehtim' not finalised yet; 'difmap_geofit' low priority
netcallist = ['netcal']
modellist = ['cres000', 'cres090', 'cres180', 'cres270', 'dblsrc', 'disk', 'edisk', 'point+disk', 'point+edisk', 'ring']
daylist = ['3644'] # '3647' later
calib = 'hops' # only hops for the synthetic data
bandlist = ['b1', 'b2', 'b3', 'b4']
smilibandlist = ['b1+2', 'b3+4'] # 'b1+2+3+4' low priority
themisbandlist = ['b1b2', 'b3b4'] # 'b1b2b3b4' low priority

parentdir = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE'
topsetparent = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/topset'
vidascript = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/software/eht2018-idfe-pipeline/idfe/vida_LS_general.jl' # vida script to run

execmode = 'both' # perform idfe and plotting
beaminuas = 20 # beamsize for CLEAN blurring in uas

proc = 48 # number of processes; must not exceed the number of physical cores available
# vida template dict
template = {'dblsrc':'gauss_2', 'disk_stretch': 'stretchdisk_1', 'disk_nostretch': 'disk_1', 'others_stretch': 'stretchmring_1_4', 'others_nostretch': 'mring_1_4'} 
stride = 200 # checkpointing interval for VIDA
stretch = True # NB: must be always set to True for M87!!!
restart = False

def execute(filelist, dataset_label, template, execmode, imager):
    """ execute pipeline"""

    rex_outfile = f'{dataset_label}_REx.h5'
    vida_outfile = f'{dataset_label}_VIDA_{template}.csv'

    if imager in ['difmap', 'difmap_geofit']: isclean = True
    else: isclean = False

    if execmode in ['both', 'rex']:
        info('Running REx...')
        runrex(filelist, dataset_label, rex_outfile, isclean, proc=proc, beaminuas=beaminuas)

    if execmode in ['both', 'vida']:
        # TODO: blur CLEAN images for VIDA? isclean not input to runvida()
        info('Running VIDA...')
        if imager == 'difmap':
            runvida(vidascript, filelist, vida_outfile, proc=proc, template=template, stride=stride, stretch=stretch, restart=restart, blur=beaminuas)
        else:
            runvida(vidascript, filelist, vida_outfile, proc=proc, template=template, stride=stride, stretch=stretch, restart=restart)

    return

# loop through dirs and perform IDFE
for imager in imagerlist:
    if imager in ['Comrade', 'THEMIS']:
        for netcal in netcallist:
            for model in modellist:
                for day in daylist:
                    if imager == 'Comrade': bands = bandlist
                    elif imager == 'THEMIS': bands = bandlist + themisbandlist
                    for band in bands:
                        inputdir = os.path.join(parentdir, imager, netcal, f'{model}_{day}_{band}')
                        if os.path.isdir(inputdir):
                            dataset_label = f'{imager}_{netcal}_{model}_{calib}_{day}_{band}'

                            # create filelist and pass to pipeline
                            filelist = f"{dataset_label}.filelist"
                            createlistcmd = f"readlink -f {inputdir}/*.fits >{filelist}"
                            info(createlistcmd)
                            os.system(createlistcmd)
                           
                            # execute pipeline
                            if model == 'dblsrc':
                                execmode = 'vida'
                                # run 2 Guassian model
                                execute(filelist, dataset_label, template['dblsrc'], execmode, imager)
                            elif 'disk' in model:
                                execmode = 'vida'
                                if stretch:
                                    execute(filelist, dataset_label, template['disk_stretch'], execmode, imager)
                                else:
                                    execute(filelist, dataset_label, template['disk_nostretch'], execmode, imager)
                            else:
                                execmode = 'both'
                                if stretch:
                                    execute(filelist, dataset_label, template['others_stretch'], execmode, imager)
                                else:
                                    execute(filelist, dataset_label, template['others_nostretch'], execmode, imager)
                        else:
                            warn(f'{inputdir} does not exist! Skipping...')                            

    elif imager in ['smili', 'ehtim', 'difmap', 'difmap_geofit']:
        netcal = 'netcal' # TODO: set netcal status; check with image evaluation team
        for model in modellist:
            for day in daylist:
                if imager == 'smili': bands = bandlist + smilibandlist
                else: bands = bandlist
                for band in bands:
                    # TODO: choose topset images -- create a filelist with topset images and pass its name to the pipeline
                    if imager == 'smili':
                        topsetfile = os.path.join(topsetparent, imager.upper(), f'topset_{calib}_{day}_{band}.csv')
                    elif imager == 'difmap':
                        topsetfile = os.path.join(topsetparent, 'CLEAN', f'topset_{calib}_{day}_{band}.csv')
                    elif imager == 'difmap_geofit':
                        topsetfile = os.path.join(topsetparent, 'CLEAN_geofit', f'topset_{calib}_{day}_{band}.csv')
                    elif imager == 'ehtim':
                        topsetfile = os.path.join(topsetparent, 'ehtim_new_202301', f'topset_{calib}_{day}_{band}.csv')

                    if os.path.isfile(topsetfile):
                        df = pd.read_csv(topsetfile)
                        topsetids = np.array(df['id'])
                    else:
                        topsetids = np.array([])

                    # deduce dataset path and pass on to the pipeline                    
                    if imager == 'ehtim': inputdir = os.path.join(parentdir, imager, f'{model}_{day}_{calib}-{band}')
                    elif imager == 'difmap_geofit': inputdir = os.path.join(parentdir, 'difmap', f'{model}_{day}_{band}_geofit')
                    else: inputdir = os.path.join(parentdir, imager, f'{model}_{day}_{band}')

                    # if image evaluation has been done for this particular dataset, proceed with execution; otherwise skip directory
                    if os.path.isdir(inputdir):
                        dataset_label = f'{imager}_{netcal}_{model}_{calib}_{day}_{band}'
                        if topsetids.shape[0] == 0:
                            warn(f"No topset info available for {inputdir}! Skipping...")
                        else:
                            # create filelist and pass to pipeline
                            filelist = []
                            for idval in topsetids:
                                if imager == 'ehtim':
                                    filelist.append(os.path.join(inputdir, f'{model}_{day}_hops-{band}_{idval:07}.fits') + '\n')
                                elif imager == 'difmap_geofit':
                                    filelist.append(os.path.join(inputdir, f'{model}_{day}_{band}_geofit_{idval:06}.fits') + '\n')
                                else:
                                    filelist.append(os.path.join(inputdir, f'{model}_{day}_{band}_{idval:06}.fits') + '\n')

                            filelistname = f"{dataset_label}.filelist"
                            with open(filelistname, 'w') as f:
                                f.writelines(filelist)

                            # execute pipeline
                            if model == 'dblsrc':
                                execmode = 'vida'
                                # run 2 Guassian model
                                execute(filelistname, dataset_label, template['dblsrc'], execmode, imager)
                            elif 'disk' in model:
                                execmode = 'vida'
                                if stretch:
                                    execute(filelistname, dataset_label, template['disk_stretch'], execmode, imager)
                                else:
                                    execute(filelistname, dataset_label, template['disk_nostretch'], execmode, imager)
                            else:
                                execmode = 'both'
                                if stretch:
                                    execute(filelistname, dataset_label, template['others_stretch'], execmode, imager)
                                else:
                                    execute(filelistname, dataset_label, template['others_nostretch'], execmode, imager)

                    else:
                        warn(f'{inputdir} does not exist! Skipping...')
