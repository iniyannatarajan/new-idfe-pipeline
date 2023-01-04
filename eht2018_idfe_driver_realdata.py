import os
import matplotlib.pyplot as plt

from idfe.utils import *

# define some variables
imagerlist = ['Comrade', 'smili', 'difmap', 'THEMIS', 'ehtim']
netcallist = ['netcal', 'nonnetcal']
sourcelist = ['M87', '3C279']
daylist = ['3644', '3645', '3647']
caliblist = ['hops', 'casa'] # TODO: set calib status; check with image evaluation team
bandlist = ['b1', 'b2', 'b3', 'b4']
smilibandlist = ['b1+2', 'b3+4', 'b1+2+3+4'] # combined band list for smili
themisbandlist = ['b1b2', 'b3b4', 'b1b2b3b4'] # combined band list for themis

parentdir = '/repository/2018-april/img/m87/paramsurvey_2211'

pipeline = 'eht2018_idfe_pipeline.py' # name of pipeline script in the current directory
vidascript='/home/iniyan/projects/eht2018-idfe/eht2018-idfe-pipeline/idfe/vida_LS_stretched_mring.jl' # vida script to run

execmode = 'idfe' # perform idfe and plotting
beaminuas = 20 # beamise for CLEAN blurring in uas

nproc = 96 # number of processes; must not exceed the number of physical cores available
varg1 = 1 # template parameter N
varg2 = 4 # template parameter M
stride = 200 # checkpointing interval for VIDA
stretch = True # NB: must be always set to True for M87!!!
restart = False

def execute(pipeline, nproc, inputdir, dataset_label, beaminuas, vidascript, varg1, varg2, execmode, stride, stretch, restart, imager):
    command = f"python {pipeline} -p {nproc} -i {inputdir} -d {dataset_label} -b {beaminuas} -v {vidascript} -t {varg1} {varg2} -e {execmode} -s {stride} "
    if stretch: # for vida
        command += '--stretch '
    if restart:
        command += '--restart '
    if imager == 'difmap':
        command += '--isdifmap '
    if imager == 'ehtim':
        command += '--isehtim '
    info(command)
    os.system(command)
    os.system(f'mv {os.path.basename(inputdir)}.filelist {dataset_label}.filelist') # TODO: this is how the pipeline script names filelist; change it there!!!

    return

# loop through dirs and perform IDFE
for imager in imagerlist:
    if imager == 'Comrade':
        netcal = 'netcal' # TODO: set netcal status; check with image evaluation team
        inputdir = os.path.join(parentdir, imager, netcal) # set partial inputdir value
        for source in modellist:
            for calib in caliblist:
                for day in daylist:
                    for band in bandlist:
                        if source == 'M87': inputdir = os.path.join(inputdir, f'{calib}_{day}_{band}')
                        elif source == '3C279': inputdir = os.path.join(inputdir, f'{source}{calib}_{day}_{band}')
                        if os.path.isdir(inputdir):
                            dataset_label = f'{imager}_{netcal}_{source}_{calib}_{day}_{band}'
                            execute(pipeline, nproc, inputdir, dataset_label, beaminuas, vidascript, varg1, varg2, execmode, stride, stretch, restart, imager)
                        else:
                            warn(f'{inputdir} does not exist. Skipping...')                            

    elif imager in ['smili', 'ehtim', 'difmap']:
        inputdir = os.path.join(parentdir, imager) # set partial inputdir value
        netcal = 'netcal' # TODO: set netcal status; check with image evaluation team
        source = 'M87'
        for day in daylist:
            if imager == 'smili': bands = bandlist + smilibandlist
            else: bands = bandlist
            for band in bands:
                if imager == 'ehtim': inputdir = os.path.join(inputdir, f'{calib}_{day}_{band}')
                else: inputdir = os.path.join(inputdir, f'{model}_{day}_{band}')
                if os.path.isdir(inputdir):
                    dataset_label = f'{imager}_{netcal}_{model}_{calib}_{day}_{band}'
                    execute(pipeline, nproc, inputdir, dataset_label, beaminuas, vidascript, varg1, varg2, execmode, stride, stretch, restart, imager)
                else:
                    warn(f'{inputdir} does not exist. Skipping...')

    elif imager == 'THEMIS':
        netcal = 'netcal'
        source = 'M87'
        inputdir = os.path.join(parentdir, imager, f'{source}real') # set partial inputdir value
        # TODO: Check with Avery about the dir structure

