import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from idfe.utils import *

# define some variables
imagerlist =  ['Comrade', 'smili', 'difmap', 'THEMIS', 'ehtim']
netcallist = ['netcal']
daylist = ['3644', '3645', '3647']
caliblist = ['hops', 'casa'] # TODO: set calib status; check with image evaluation team
bandlist = ['b1', 'b2', 'b3', 'b4']
smilibandlist = ['b1+2', 'b3+4', 'b1+2+3+4'] # combined band list for smili
themisbandlist = ['b1b2', 'b3b4', 'b1b2b3b4'] # combined band list for themis

parentdir = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE'
topsetparent = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/topset'
pipeline = 'eht2018_idfe_pipeline.py' # name of pipeline script in the current directory
vidascript = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/software/eht2018-idfe-pipeline/idfe/vida_LS_general.jl' # vida script to run

execmode = 'idfe' # perform idfe and plotting
beaminuas = 20 # beamise for CLEAN blurring in uas

nproc = 48 # number of processes; must not exceed the number of physical cores available
template = {'stretch': 'stretchmring_1_4', 'nostretch': 'mring_1_4'}
stride = 200 # checkpointing interval for VIDA
stretch = True # NB: must be always set to True for M87!!!
restart = False

def execute(pipeline, nproc, filelist, dataset_label, beaminuas, vidascript, template, execmode, stride, stretch, restart, imager):
    """ execute pipeline"""

    command = f"python {pipeline} -p {nproc} -i {filelist} -d {dataset_label} -b {beaminuas} -v {vidascript} -t {template} -e {execmode} -s {stride} "
    if stretch: # for vida
        command += '--stretch '
    if restart:
        command += '--restart '
    if imager == 'difmap':
        command += '--isclean '
    if imager == 'ehtim':
        command += '--isehtim '
    info(command)
    os.system(command)
    #os.system(f'mv {os.path.basename(inputdir)}.filelist {dataset_label}.filelist') # TODO: this is how the pipeline script names filelist; change it there!!!

    return

# loop through dirs and perform IDFE
for imager in imagerlist:
    if imager in ['Comrade', 'THEMIS']:
        for netcal in netcallist:
            for calib in caliblist:
                for day in daylist:
                    if imager == 'Comrade': bands = bandlist
                    elif imager == 'THEMIS': bands = bandlist + themisbandlist
                    for band in bands:
                        inputdir = os.path.join(parentdir, imager, netcal, f'{calib}_{day}_{band}')
                        if os.path.isdir(inputdir):
                            dataset_label = f'{imager}_{netcal}_{calib}_{day}_{band}'

                            # create filelist and pass to pipeline
                            filelist = f"{dataset_label}.filelist"
                            createlistcmd = f"readlink -f {inputdir}/*.fits >{filelist}"
                            info(createlistcmd)
                            os.system(createlistcmd)
                           
                            # execute pipeline
                            if stretch:
                                execute(pipeline, nproc, filelist, dataset_label, beaminuas, vidascript, template['stretch'], execmode, stride, stretch, restart, imager)
                            else:
                                execute(pipeline, nproc, filelist, dataset_label, beaminuas, vidascript, template['nostretch'], execmode, stride, stretch, restart, imager)
                        else:
                            warn(f'{inputdir} does not exist! Skipping...')                            

    elif imager in ['smili', 'ehtim', 'difmap']:
        netcal = 'netcal' # TODO: set netcal status; check with image evaluation team
        for calib in caliblist:
            for day in daylist:
                if imager == 'smili': bands = bandlist + smilibandlist
                else: bands = bandlist
                for band in bands:
                    # TODO: choose topset images -- create a filelist with topset images and pass its name to the pipeline
                    if imager == 'smili':
                        topsetfile = os.path.join(topsetparent, imager.upper(), f'topset_{calib}_{day}_{band}.csv')
                    elif imager == 'difmap':
                        topsetfile = os.path.join(topsetparent, 'CLEAN', f'topset_{calib}_{day}_{band}.csv')
                    else:
                        topsetfile = os.path.join(topsetparent, imager, f'topset_{calib}_{day}_{band}.csv')
                    if os.path.isfile(topsetfile):
                        df = pd.read_csv(topsetfile)
                        topsetids = np.array(df['id'])
                    else:
                        topsetids = np.array([])

                    # deduce dataset path and pass on to the pipeline                    
                    inputdir = os.path.join(parentdir, imager, f'{calib}_{day}_{band}')

                    # if image evaluation has been done for this particular dataset, proceed with execution; otherwise skip directory (TODO: verify with others if skipping is the right thing!)
                    if os.path.isdir(inputdir):
                        dataset_label = f'{imager}_{netcal}_{calib}_{day}_{band}'
                        if topsetids.shape[0] == 0:
                            warn(f"No topset info available for {inputdir}! Skipping...")
                        else:
                            # create filelist and pass to pipeline
                            filelist = []
                            for idval in topsetids:
                                if imager == 'ehtim':
                                    filelist.append(os.path.join(inputdir, f'{calib}_{day}_{band}_{idval:07}.fits') + '\n')
                                else:
                                    filelist.append(os.path.join(inputdir, f'{calib}_{day}_{band}_{idval:06}.fits') + '\n')

                            filelistname = f"{dataset_label}.filelist"
                            with open(filelistname, 'w') as f:
                                f.writelines(filelist)

                            # execute pipeline
                            if stretch:
                                execute(pipeline, nproc, filelistname, dataset_label, beaminuas, vidascript, template['stretch'], execmode, stride, stretch, restart, imager)
                            else:
                                execute(pipeline, nproc, filelistname, dataset_label, beaminuas, vidascript, template['nostretch'], execmode, stride, stretch, restart, imager)
                    else:
                        warn(f'{inputdir} does not exist! Skipping...')
