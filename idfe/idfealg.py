import re
import os
import sys
import copy
import ehtim as eh
import pandas as pd
import subprocess
from multiprocessing import Pool
from scipy import interpolate, stats
import astropy.units as u
from astropy.constants import k_B,c
from ehtim.const_def import *

from idfe.utils import *


# Define REx-related ring characteristics functions from Yutaro
def extract_ring_quantities(image, xc=None, yc=None, rmin_search=5, rmax_search=50):
    Npa=360
    Nr=2*rmax_search

    # zero negative values -- affects only Themis images
    image.imvec[image.imvec<0] = 0

    if xc==None or yc==None:
    # Compute the image center -----------------------------------------------------------------------------------------
        xc,yc = fit_ring(image,rmin_search=2*rmin_search,rmax_search=2*rmax_search)
    # Gridding and interpolation ---------------------------------------------------------------------------------------
    x= np.arange(image.xdim)*image.psize/RADPERUAS
    y= np.flip(np.arange(image.ydim)*image.psize/RADPERUAS)
    z = image.imarr()
    f_image = interpolate.interp2d(x,y,z,kind="cubic") # init interpolator

    # Create a mesh grid in polar coordinates
    radial_imarr = np.zeros([Nr,Npa])

    pa = np.linspace(0,360,Npa)
    pa_rad = np.deg2rad(pa)
    radial = np.linspace(0,rmax_search,Nr)
    dr = radial[-1]-radial[-2]

    Rmesh, PAradmesh = np.meshgrid(radial, pa_rad)
    x = Rmesh*np.sin(PAradmesh) + xc
    y = Rmesh*np.cos(PAradmesh) + yc
    for r in range(Nr):
        z = [f_image(x[i][r],y[i][r]) for i in range(len(pa))]
        radial_imarr[r,:] = z[:]
    radial_imarr = np.fliplr(radial_imarr)
    # Calculate the r_pk at each PA and average -> using radius  --------------------------------------------------------
    # Caluculating the ring width from rmin and rmax
    peakpos = np.unravel_index(np.argmax(radial_imarr), shape=radial_imarr.shape)

    Rpeak=[]
    Rmin=[]
    Rmax=[]
    ridx_rmax= np.argmin(np.abs(radial - rmax_search))
    I_floor = radial_imarr[ridx_rmax,:].mean()
    for ipa in range(len(pa)):
        tmpIr = copy.copy(radial_imarr[:,ipa])
        tmpIr[np.where(radial < rmin_search)]=0
        ridx_pk = np.argmax(tmpIr)
        rpeak = radial[ridx_pk]
        if ridx_pk > 0 and ridx_pk < Nr-1:
            val_list= tmpIr[ridx_pk-1:ridx_pk+2]
            rpeak = quad_interp_radius(rpeak, dr, val_list)[0]
        Rpeak.append(rpeak)

        tmpIr = copy.copy(radial_imarr[:,ipa])-I_floor
        # if tmpIr < 0, make rmin & rmax nan
        rmin,rmax = calc_width(tmpIr,radial,rpeak)
        # append
        Rmin.append(rmin)
        Rmax.append(rmax)
    paprofile = pd.DataFrame()
    paprofile["PA"] = pa
    paprofile["rpeak"] = Rpeak
    paprofile["rhalf_max"]=Rmax
    paprofile["rhalf_min"]=Rmin

    D = np.mean(paprofile["rpeak"]) * 2
    Derr = paprofile["rpeak"].std() * 2
    Rhalf_max = np.mean(paprofile["rhalf_max"])
    Rhalf_min = np.mean(paprofile["rhalf_min"])
    W = np.mean(paprofile["rhalf_max"] - paprofile["rhalf_min"])
    Werr =  (paprofile["rhalf_max"] - paprofile["rhalf_min"]).std()

    # Calculate the orientation angle, contrast, and asymmetry
    rin  = D/2.-W/2.
    rout  = D/2.+W/2.
    if rin <= 0.:
        rin  = 0.

    exptheta =np.exp(1j*pa_rad)

    pa_ori_r=[]
    amp_r = []
    ridx1 = np.argmin(np.abs(radial - rin))
    ridx2 = np.argmin(np.abs(radial - rout))
    for r in range(ridx1, ridx2+1, 1):
        amp =  (radial_imarr[r,:]*exptheta).sum()/(radial_imarr[r,:]).sum()
        amp_r.append(amp)
        pa_ori = np.angle(amp, deg=True)
        pa_ori_r.append(pa_ori)
    pa_ori_r=np.array(pa_ori_r)
    amp_r = np.array(amp_r)
    PAori = stats.circmean(pa_ori_r,high=360,low=0)
    PAerr = stats.circstd(pa_ori_r,high=360,low=0)
    A = np.mean(np.abs(amp_r))
    Aerr = np.std(np.abs(amp_r))

    ridx_r5= np.argmin(np.abs(radial - 5))
    ridx_pk = np.argmin(np.abs(radial - D/2))
    fc = radial_imarr[0:ridx_r5,:].mean()/radial_imarr[ridx_pk,:].mean()

    # source size from 2nd moment
    fwhm_maj,fwhm_min,theta = image.fit_gauss()
    fwhm_maj /= RADPERUAS
    fwhm_min /= RADPERUAS

    # calculate flux ratio
    Nxc = int(xc/image.psize*RADPERUAS)
    Nyc = int(yc/image.psize*RADPERUAS)
    hole = extract_hole(image,Nxc,Nyc,r=rin)
    ring = extract_ring(image,Nxc,Nyc,rin=rin, rout=rout)
    outer = extract_outer(image,Nxc,Nyc,r=rout)
    hole_flux = hole.total_flux()
    outer_flux = outer.total_flux()
    ring_flux = ring.total_flux()

    Shole  = np.pi*rin**2
    Souter = (2.*rout)**2.-np.pi*rout**2
    Sring = np.pi*rout**2-np.pi*rin**2

    # convert uas^2 to rad^2
    Shole = Shole*RADPERUAS**2
    Souter = Souter*RADPERUAS**2
    Sring = Sring*RADPERUAS**2

    #unit K brightness temperature
    freq = image.rf*u.Hz
    hole_dflux  = hole_flux/Shole*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    outer_dflux = outer_flux/Souter*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value
    ring_dflux = ring_flux/Sring*(c**2/2/k_B/freq**2).to(u.K/u.Jansky).value

    # output dictionary
    outputs = dict(
        radial_imarr=radial_imarr,
        peak_idx=peakpos,
        rpeak=radial[peakpos[0]],
        papeak=pa[peakpos[1]],
        paprofile=paprofile,
        xc=xc,
        yc=yc,
        r = radial,
        PAori = PAori,
        PAerr = PAerr,
        A = A,
        Aerr = Aerr,
        fc = fc,
        D = D,
        Derr = Derr,
        W = W,
        Werr = Werr,
        fwhm_maj=fwhm_maj,
        fwhm_min=fwhm_min,
        hole_flux = hole_flux,
        outer_flux = outer_flux,
        ring_flux = ring_flux,
        totalflux = image.total_flux(),
        hole_dflux = hole_dflux,
        outer_dflux = outer_dflux,
        ring_dflux = ring_dflux,
        Rhalf_max =Rhalf_max,
        Rhalf_min = Rhalf_min
    )

    return outputs


# Clear ring structures
def extract_hole(image,Nxc,Nyc, r=30):
    outimage = copy.deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2>=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)

    return outimage


def extract_outer(image,Nxc,Nyc, r=30):
    outimage = copy.deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - r**2<=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)
    return outimage


def extract_ring(image, Nxc,Nyc,rin=30,rout=50):
    outimage = copy.deepcopy(image)
    x = (np.arange(outimage.xdim)-Nxc+1)*outimage.psize/RADPERUAS
    y =  (np.arange(outimage.ydim)-Nyc+1)*outimage.psize/RADPERUAS
    x,y = np.meshgrid(x, y)
    masked = outimage.imarr()
    masked[np.where(x**2 + y**2 - rin**2<=0)] = 0
    masked[np.where(x**2 + y**2 - rout**2>=0)] = 0
    outimage.imvec = masked.reshape(outimage.ydim*outimage.xdim)

    return outimage


def quad_interp_radius(r_max, dr, val_list):
    v_L = val_list[0]
    v_max = val_list[1]
    v_R = val_list[2]
    rpk = r_max + dr*(v_L - v_R) / (2 * (v_L + v_R - 2*v_max))
    vpk = 8*v_max*(v_L + v_R) - (v_L - v_R)**2 - 16*v_max**2
    vpk /= (8*(v_L + v_R - 2*v_max))
    return (rpk, vpk)


def calc_width(tmpIr,radial,rpeak):
    spline = interpolate.UnivariateSpline(radial, tmpIr-0.5*tmpIr.max(), s=0)
    roots = spline.roots()  # find the roots

    if len(roots) == 0:
        return(radial[0], radial[-1])

    rmin = radial[0]
    rmax = radial[-1]
    for root in np.sort(roots):
        if root < rpeak:
            rmin = root
        else:
            rmax = root
            break

    return (rmin, rmax)


def fit_ring(image,Nr=50,Npa=25,rmin_search = 10,rmax_search = 100,fov_search = 0.1,Nserch =20):
    # rmin_search,rmax_search must be diameter
    image_blur = image.blur_circ(2.0*RADPERUAS,fwhm_pol=0)
    image_mod = image_blur.threshold(cutoff=0.05)
    image_mod = image
    xc,yc = eh.features.rex.findCenter(image_mod, rmin_search=rmin_search, rmax_search=rmax_search,
                         nrays_search=Npa, nrs_search=Nr,
                         fov_search=fov_search, n_search=Nserch)
    return xc,yc


def runrex(filelist, label, rex_outfile, isclean, proc=8, beaminuas=20, frac=1.0):

    # get filenames from the filelist and define some variables
    fnames = np.genfromtxt(f'{filelist}', dtype='str')
    topsetcode = np.array([int(re.findall('\d+',x)[-1]) for x in fnames])

    allpars = ["D","Derr","W","Werr","PAori","PAerr","papeak","A","Aerr","fc","xc","yc","fwhm_maj","fwhm_min","hole_flux","outer_flux","ring_flux","totalflux","hole_dflux","outer_dflux","ring_dflux"]
    npars = len(allpars)
    nimages = topsetcode.shape[0]

    # Load FITS images into ehtim and blur if necessary
    pool = Pool(proc)
    imglist = list(pool.imap(eh.image.load_fits, fnames))
    pool.close()

    # Load FITS images into ehtim serially (for DEBUGGING)
    '''imglist = []
    for fname in fnames:
        info(f'Loading image {fname}...')
        imglist.append(eh.image.load_fits(fname))'''
    
    if isclean:
        for img in imglist:
            img = img.blur_gauss(beamparams=[beaminuas*RADPERUAS,beaminuas*RADPERUAS,0], frac=frac)
        
    # extract ring characteristics in parallel
    pool = Pool(proc)
    ring_outputs = list(pool.imap(extract_ring_quantities, imglist))
    pool.close()

    # extract ring characteristics serially (for DEBUGGING)
    '''info('Start extracting ring...')
    ring_outputs = []
    for fname, img in zip(fnames, imglist):
        info(f'Analysing image {fname}...')
        try:
            ring_outputs.append(extract_ring_quantities(img))
        except OSError:
            warn(f'Image {image} failed! Aborting.')'''

    # Read relevant parameters from ring_outputs into a numpy array
    rex_pars = np.zeros((nimages, npars))

    for ii in np.arange(nimages):
        for jj in np.arange(npars):
            rex_pars[ii,jj] = ring_outputs[ii][allpars[jj]]
    
    # Write the numpy arrays into a pandas DataFrame
    df_rex = pd.DataFrame([])
    df_rex['id'] = topsetcode

    for ii in np.arange(npars):
        df_rex[f'{allpars[ii]}_{label}'] = rex_pars[:,ii]

    df_rex.to_hdf(rex_outfile, 'parameters', mode='w', complevel=9, format='table')
    info(f'REx output saved to {rex_outfile}')


def runvida(vidascript, filelist, vida_outfile, proc=8, template='mring_1_4', stride=200, stretch=False, restart=False, model='FLOOR'):
    '''
    Perform IDFE using VIDA
    '''

    runvidacmd = f'julia -p {proc} {vidascript} {filelist} --out {vida_outfile} --template {template} --stride {stride}'

    if restart:
      runvidacmd += ' --restart'

    info(runvidacmd)
    subprocess.run(runvidacmd, shell=True, universal_newlines=True)

    # extract fractional central brightness from VIDA output and write to a new file
    if stretch:
        info(f"stretch enabled for VIDA")
    else:
        cmd = f'readlink -f {vida_outfile} >filelist'
        info(cmd)
        subprocess.run(cmd, shell=True, universal_newlines=True)

        # get path for fc_convert.jl
        vidapath = os.path.split(vidascript)[0]
        fcscript = os.path.join(vidapath, 'fc_convert.jl')

        cmd = f'julia --project={vidapath} {fcscript} filelist --model {model}'
        info(cmd)
        subprocess.run(cmd, shell=True, universal_newlines=True)

    info(f'VIDA output saved to {vida_outfile}')
