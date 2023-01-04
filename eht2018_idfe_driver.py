import os
import matplotlib.pyplot as plt

from idfe.utils import *

# define some variables
imagerlist = ['Comrade', 'smili', 'difmap', 'THEMIS', 'ehtim']
#scatterlist = ['deblur', 'nodeblur']
netcallist = ['netcal', 'nonnetcal']
modellist = ['cres000', 'cres090', 'cres180', 'cres270', 'dblsrc', 'disk', 'ecres000', 'ecres045', 'ecres090', 'ecres315', 'edisk', 'grmhd', 'point+disk', 'point_edisk', 'ring', '', '3C279']
daylist = ['3644', '3645', '3647']
caliblist = ['hops', 'casa']
bandlist = ['b1', 'b2', 'b3', 'b4', 'b1+2', 'b3+4', 'b1+2+3+4']

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
        for netcal in netcallist:
            inputdir = os.path.join(parentdir, imager, netcal) # set partial inputdir value
            for day in daylist:
                for band in bandlist:
                    for model in modellist:
                        if model == '3C279' or not model:
                            for calib in caliblist:
                                inputdir = os.path.join(inputdir, f'{model}{calib}_{day}_{band}')
                                if os.path.isdir(inputdir):
                                    dataset_label = f'{imager}_{netcal}_{model}_{calib}_{day}_{band}'
                                    execute(pipeline, nproc, inputdir, dataset_label, beaminuas, vidascript, varg1, varg2, execmode, stride, stretch, restart, imager)
                                else:
                                    warn(f'{inputdir} does not exist. Skipping...')
                        else:
                            calib = 'hops' # TODO: set calib status; check with image evaluation team
                            inputdir = os.path.join(inputdir, f'{model}_{day}_{band}')
                            if os.path.isdir(inputdir):
                                dataset_label = f'{imager}_{netcal}_{model}_{calib}_{day}_{band}'
                                execute(pipeline, nproc, inputdir, dataset_label, beaminuas, vidascript, varg1, varg2, execmode, stride, stretch, restart, imager)
                            else:
                                warn(f'{inputdir} does not exist. Skipping...')                            

    elif imager == 'smili':
        inputdir = os.path.join(parentdir, imager) # set partial inputdir value
        netcal = 'netcal' # TODO: set netcal status; check with image evaluation team
        for day in daylist:
            for band in bandlist:
                for model in modellist:
                    if model == '3C279' or not model:
                        for calib in caliblist:
                            inputdir = os.path.join(inputdir, f'{model}{calib}_{day}_{band}')
                            if os.path.isdir(inputdir):
                                dataset_label = f'{imager}_{netcal}_{model}_{calib}_{day}_{band}'
                                execute(pipeline, nproc, inputdir, dataset_label, beaminuas, vidascript, varg1, varg2, execmode, stride, stretch, restart, imager)
                            else:
                                warn(f'{inputdir} does not exist. Skipping...')
                    else:
                        calib = 'hops' # TODO: check with image evaluation team
                        inputdir = os.path.join(inputdir, f'{model}_{day}_{band}')
                        if os.path.isdir(inputdir):
                            dataset_label = f'{imager}_{netcal}_{model}_{calib}_{day}_{band}'
                            execute(pipeline, nproc, inputdir, dataset_label, beaminuas, vidascript, varg1, varg2, execmode, stride, stretch, restart, imager)
                        else:
                            warn(f'{inputdir} does not exist. Skipping...')                    

    elif imager == 'difmap':
        # set up dir structure
    elif imager == 'THEMIS':
        # set up dir structure
    elif imager == 'ehtim':
        # set up dir structure

    for scatter in scatterlist:
        dirname = os.path.join(parentdir, f'images_{scatter}_{imager}')
        for calib in caliblist:
            for day in daylist:
                inputdir = os.path.join(dirname, f'{calib}_{day}')
                dataset_label = f'{imager}_{scatter}_{calib}_{day}_{bandlist[0]}'
                command = f"python {pipeline} -p {nproc} -i {inputdir} -d {dataset_label} -b {beaminuas} -v {vidascript} -t {varg1} {varg2} -e {execmode} -s {stride} "
                if stretch:
                    command += '--stretch '
                if restart:
                    command += '--restart '
                if imager == 'CLEAN':
                    command += '--isclean '
                    #command += f"-ng {ngrid_dict['CLEAN']} -pl {plength_dict['CLEAN']} -hl {hlength_dict['CLEAN']} "
                if imager in ['ehtim', 'Themis']:
                    command += '--isehtim '
                    #command += f"-ng {ngrid_dict['ehtim']} -pl {plength_dict['ehtim']} -hl {hlength_dict['ehtim']} "

                info(command)
                os.system(command)
                os.system(f'mv {os.path.basename(inputdir)}.filelist {dataset_label}.filelist') # this is how the pipeline script names the .filelist file
