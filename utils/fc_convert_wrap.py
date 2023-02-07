import os
import sys
import subprocess

from idfe.utils import *

fcscript = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/software/eht2018-idfe-pipeline/idfe/fc_convert_2018.jl'
vidapath = os.path.split(fcscript)[0]
parentdir = '/n/holylfs05/LABS/bhi/Lab/doeleman_lab/inatarajan/EHT2018_M87_IDFE/results'
dirlist = ['real', 'synthetic', 'validation']
syntheticmodellist = ['cres000', 'cres090', 'cres180', 'cres270', 'dblsrc', 'disk', 'edisk', 'point+disk', 'point+edisk', 'ring']
validationmodellist = ['ecres000', 'ecres045', 'ecres090', 'ecres315', 'grmhd']
flist = 'filelist.txt'

for indir in dirlist:
    fulldirpath = os.path.join(parentdir, indir)

    # add full path for each csv file to a filelist
    if indir == 'real':
        cmd = f'readlink -f {fulldirpath}/*VIDA*.csv > {flist}'
        info(cmd)
        subprocess.run(cmd, shell=True, universal_newlines=True)
    elif indir == 'synthetic':
        for smodel in syntheticmodellist:
            cmd = f'readlink -f {fulldirpath}/{smodel}/*VIDA*.csv >> {flist}'
            info(cmd)
            subprocess.run(cmd, shell=True, universal_newlines=True)
    elif indir == 'validation':
        for vmodel in validationmodellist:
            cmd = f'readlink -f {fulldirpath}/{vmodel}/*VIDA*.csv >> {flist}'
            info(cmd)
            subprocess.run(cmd, shell=True, universal_newlines=True)
            
cmd = f'julia --project={vidapath} {fcscript} {flist}'
info(cmd)
subprocess.run(cmd, shell=True, universal_newlines=True)
