"""Image reconstruction from raw PET data"""
import logging
import os
import time
from collections import namedtuple
from collections.abc import Iterable
from numbers import Real

from pkg_resources import resource_filename
from pathlib import Path
from os import fspath, path
# resources contain isotope info
from .. import mmr_auxe, mmraux, mmrnorm, resources
from ..img import mmrimg
from ..lm.mmrhist import randoms
from ..sct import vsm
from . import petprj
from ..prj_sig import prjsig
import cuvec as cu
import numpy as np
import scipy.ndimage as ndi
from tqdm.auto import trange
from scipy.ndimage import gaussian_filter

from niftypet import nimpa
#take image as input in function as well as the amount of bins and angles
def Reconstruct(image,angles,muMaps,scanner_params,hst):
    image = np.array(image, dtype=np.float32)
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']
    #mumaps
    muh, muo = muMaps
    mus = mmrimg.convert2dev(muo+muh, Cnt)
    #removegaps from the prompt sinogram
    #psino = prompt sinogram
    #psng = mmraux.remgaps(hst['psino'],txLUT,Cnt)
    #asng = attenuation factor sinogram
    
    #forward project
    NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
    # number of sinos in span-1
    isub = np.array([-1], dtype=np.int32)
    nsinos = NRNG_c**2
    sinogramShape = (344,252,837)#(txLUT['Naw'], nsinos)#
    asng = cu.zeros(sinogramShape, dtype=np.float32)
    ims = mmrimg.convert2dev(image, Cnt)
    
    petprj.fprj(asng,cu.asarray(ims, dtype=np.float32),txLUT, axLUT, 
    isub, Cnt,False)
    #h = np.max(asng)/2
    #asngT = asng*(asng>h)
    #sino[isub, :] = asng
    print("fprj: ",asng)
    
    #randoms
    rsino, snglmap = randoms(hst, scanner_params)
    rsng = mmraux.remgaps(cu.asarray(asng, dtype=np.float32), txLUT, Cnt)
    #print("rsng: ",rsng)
    nvz = Cnt['SZ_IMZ']

    out_shape = Cnt['SZ_IMX'], Cnt['SZ_IMY'], nvz
    img = cu.asarray(np.zeros((ims.shape), dtype=np.float32))
    
    petprj.bprj(img, cu.asarray(asng, dtype=np.float32), txLUT, axLUT, 
    isub,Cnt)
    
    print("img: ",img)
    
    return img, asng, image, ims,rsino,rsng#, sino



def RemoveGaps(sinogram, scanner_params):
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    remmedSino = mmraux.remgaps(cu.asarray(sinogram, dtype=np.float32), txLUT, Cnt)
    return remmedSino


def Randoms(scanner_params,hst):
    rsino, snglmap = randoms(hst, scanner_params)
    return rsino, snglmap

def Scatter(datain,mumaps,scanner_params,hst,rsino,eim):
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']
    #emd = EMML(hst['psino'],datain,mumaps,scanner_params,hst,1)
    #eim = EMML(cu.asarray(np.ones_like(hst['psino']), dtype=np.float32), datain,
    #mumaps,scanner_params,hst,7)
    #sensitivity = Back(cu.asarray(np.ones_like(hst['psino']), dtype=np.float32),mumaps,scanner_params,hst)
    #img = emd/(sensitivity+1e-8)
    #emd = nimpa.getnii(datain['em_crr'])
    muh, muo = mumaps
    mus = mmrimg.convert2dev(muo+muh, Cnt)
    NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
    isub = np.array([-1], dtype=np.int32)
    nsinos = NRNG_c**2
    fMus = Forward(mus,mumaps,scanner_params,hst)
    
    #ncmp, _ = mmrnorm.get_components(datain, Cnt)
    #nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst, normcomp=ncmp, gpu_dim=False)
    
    ssn = vsm(
        datain,
        mumaps,
        eim,
        scanner_params,
        hst,
        rsino,
        0.1,
        
    )
    return ssn

def PutGapsIn(sino,scanner_params):
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']
    rsng = mmraux.putgaps(sino, txLUT, Cnt)
    return rsng
def Forward(image,muMaps,scanner_params,hst,attenuation=True):
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']
    sinogramShape = (344,344,837)#(txLUT['Naw'], nsinos)#
    image = np.array(image)
    try:
        ims = mmrimg.convert2dev(image, Cnt)
    except:
        ims = image
    pred = cu.ones(sinogramShape,dtype=np.float32)
    #muh, muo = muMaps
    #mus = mmrimg.convert2dev(muo+muh, Cnt)
    #NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
    isub = np.array([-1], dtype=np.int32)
    #nsinos = NRNG_c**2
    asng = cu.zeros(sinogramShape, dtype=np.float32)
    petprj.fprj(asng,cu.asarray(ims, dtype=np.float32),txLUT, axLUT, 
                    isub, Cnt,attenuation)
    rsng = mmraux.putgaps(asng, txLUT, Cnt)
    return asng, rsng

def SigForward(image, scanner_params, attenuation=True):
    Cnt, txLUT, axLUT = scanner_params['Cnt'], scanner_params['txLUT'], scanner_params['axLUT']
    nsinos = Cnt['NSN'] if Cnt['SPN'] == 2 else Cnt['NSN1']
    sinogramShape = (Cnt['NAW'], nsinos)
    ims = cu.asarray(np.array(image), dtype=np.float32)
    isub = np.array([-1], dtype=np.int32)
    asng = cu.zeros(sinogramShape, dtype=np.float32)
    prjsig.fprj(asng, ims, txLUT, axLUT, isub, Cnt, int(attenuation), sync=True)
    return asng

def convDev(projected,Cnt):
    bimg = mmrimg.convert2e7(projected, Cnt)
    return bimg

def psf_config(psf, Cnt):
    '''
    Generate separable PSF kernel (x, y, z) based on FWHM for x, y, z

    Args:
      psf:
        None: PSF reconstruction is switched off
        'measured': PSF based on measurement (line source in air)
        float: an isotropic PSF with the FWHM defined by the float or int scalar
        [x, y, z]: list or Numpy array of separate FWHM of the PSF for each direction
        ndarray: 3 x 2*RSZ_PSF_KRNL+1 Numpy array directly defining the kernel in each direction
    '''

    def _config(fwhm3, check_len=True):
        # resolution modelling by custom kernels
        if check_len:
            if len(fwhm3) != 3 or any(f < 0 for f in fwhm3):
                raise ValueError('Incorrect separable kernel FWHM definition')

        kernel = np.empty((3, 2 * Cnt['RSZ_PSF_KRNL'] + 1), dtype=np.float32)
        for i, psf in enumerate(fwhm3):
            # > FWHM -> sigma conversion for all dimensions separately
            if i == 2:
                sig = fwhm2sig(psf, voxsize=Cnt['SZ_VOXZ'] * 10)
            else:
                sig = fwhm2sig(psf, voxsize=Cnt['SZ_VOXY'] * 10)

            x = np.arange(-Cnt['RSZ_PSF_KRNL'], Cnt['RSZ_PSF_KRNL'] + 1)
            kernel[i, :] = np.exp(-0.5 * (x**2 / sig**2))
            kernel[i, :] /= np.sum(kernel[i, :])

        psfkernel = np.empty((3, 2 * Cnt['RSZ_PSF_KRNL'] + 1), dtype=np.float32)
        psfkernel[0, :] = kernel[2, :]
        psfkernel[1, :] = kernel[0, :]
        psfkernel[2, :] = kernel[1, :]

        return psfkernel

    if psf is None:
        psfkernel = _config([], False)
        # switch off PSF reconstruction by setting negative first element
        psfkernel[0, 0] = -1
    elif psf == 'measured':
        psfkernel = nimpa.psf_measured(scanner='mmr', scale=1)
    elif isinstance(psf, Real):
        psfkernel = _config([psf] * 3)
    elif isinstance(psf, Iterable):
        psf = np.asanyarray(psf)
        if psf.shape == (3, 2 * Cnt['RSZ_PSF_KRNL'] + 1):
            psfkernel = _config([], False)
            psfkernel[0, :] = psf[2, :]
            psfkernel[1, :] = psf[0, :]
            psfkernel[2, :] = psf[1, :]
        elif len(psf) == 3:
            psfkernel = _config(psf)
        else:
            raise ValueError(f"invalid PSF dimensions ({psf.shape})")
    else:
        raise ValueError(f"unrecognised PSF definition ({psf})")
    return psfkernel
def fwhm2sig(fwhm, voxsize=1.):
    return (fwhm/voxsize) / (2 * (2 * np.log(2))**.5)
def Back(sino,muMaps,scanner_params,hst):
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']
    isub = np.array([-1], dtype=np.int32)
    if Cnt['SPN'] == 1 and 'rSZ_IMZ' in Cnt:
        nvz = Cnt['rSZ_IMZ']
    else:
        nvz = Cnt['SZ_IMZ']
    sinogramShape = (Cnt['SZ_IMX'], Cnt['SZ_IMY'], Cnt['SZ_IMZ'])#(127,344,344)
    
    
    img = cu.asarray(np.zeros(sinogramShape), dtype=np.float32)
    rsng = mmraux.remgaps(cu.asarray(sino, dtype=np.float32), txLUT, Cnt)
    petprj.bprj(img, cu.asarray(rsng, dtype=np.float32), txLUT, axLUT, 
                    isub,Cnt, False)
    img = convDev(img,Cnt)
    #ims = mmrimg.convert2dev(img, Cnt)
    #vol = np.transpose(img, (2,1,0))   # xyz -> zyx
    #vol = np.rot90(vol, 1, axes=(2,1)) # rotate each slice
    
    return img#vol

def SigBack(sino, scanner_params):
    Cnt, txLUT, axLUT = scanner_params['Cnt'], scanner_params['txLUT'], scanner_params['axLUT']
    isub = np.array([-1], dtype=np.int32)
    im_shape = (Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ'])
    img = cu.zeros(im_shape, dtype=np.float32)
    sino_cu = cu.asarray(np.asarray(sino, dtype=np.float32))
    prjsig.bprj(img, sino_cu, txLUT, axLUT, isub, Cnt, sync=True)
    return img
#tmpsens = cu.ones((837,344,344),dtype=np.float32)
#bksens = Back(tmpsens,)
def getNormPara(datain,Cnt):
    ncmp, rest = mmrnorm.get_components(datain, Cnt)
    return ncmp,rest
def getNormParameters(datain,Cnt):
    ncmp, rest = mmrnorm.get_components(datain, Cnt)
    return ncmp,rest, resources.riLUT
def getNorm(datain,scanner_params,hst,Cnt):
    ncmp, _ = mmrnorm.get_components(datain, Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst, normcomp=ncmp, gpu_dim=False)
    return nsng

def EMML(sinog, datain, muMaps, scanner_params, hst, iterations=1):
    
    # Derive shape from actual data - never hardcode
    sinogramShape = sinog.shape  # (837, 252, 344)
    print("Using sinogram shape:", sinogramShape)
    
    Cnt = scanner_params['Cnt']
    
    pred = cu.ones((
        Cnt['SO_IMZ'],
        Cnt['SO_IMY'],
        Cnt['SO_IMX']
    ), dtype=np.float32)

    muh, muo = muMaps
    muhs = mmrimg.convert2dev(muh, Cnt)
    mus  = mmrimg.convert2dev(muo+muh, Cnt)

    fMus = Forward(mus, muMaps, scanner_params, hst,True)
    acf_sino = fMus[1]  # will also be (837, 252, 344)

    ncmp, _ = mmrnorm.get_components(datain, Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst, normcomp=ncmp, gpu_dim=False)

    # Sensitivity: Back(ones) using CORRECT shape
    # sensitivity = Back(
    #     cu.asarray(np.ones(sinogramShape, dtype=np.float32)),
    #     muMaps, scanner_params, hst
    # )
    sensitivity = Back(
        cu.asarray(nsng*acf_sino, dtype=np.float32),
        muMaps, scanner_params, hst
    )
    sensitivity = np.maximum(sensitivity, 1e-8)

    print("sensitivity min/max:", float(sensitivity.min()), float(sensitivity.max()))

    for i in range(iterations):
        fwd = Forward(pred, muMaps, scanner_params, hst,False)[1]
        
        expected = acf_sino * fwd *nsng
        expected = np.maximum(expected, 1e-10)

        ratio = (sinog / (expected + 1e-8)) * acf_sino * nsng
        
        correction = Back(
            cu.asarray(ratio, dtype=np.float32),
            muMaps, scanner_params, hst
        )

        pred *= (correction / sensitivity)
        pred  = np.maximum(pred, 0.0)

        print(f"Iter {i+1} | pred min/max/mean: "
              f"{float(pred.min()):.4f} / {float(pred.max()):.4f} / {float(pred.mean()):.4f}")

    return pred

from scipy.ndimage import convolve1d

def apply_psf(psf, Cnt):
    """
    Apply PSF blurring to a sinogram via separable 1D convolutions.
    
    Args:
        sino:       np.ndarray, shape (837, 252, 344)
                    (segments x angles x radial_bins)
        psf_kernel: np.ndarray, shape (3, 17)
                    One 1D kernel per sinogram axis
    Returns:
        blurred sinogram of same shape
    """
    def _config(fwhm3, check_len=True):
        # resolution modelling by custom kernels
        if check_len:
            if len(fwhm3) != 3 or any(f < 0 for f in fwhm3):
                raise ValueError('Incorrect separable kernel FWHM definition')

        kernel = np.empty((3, 2 * Cnt['RSZ_PSF_KRNL'] + 1), dtype=np.float32)
        for i, psf in enumerate(fwhm3):
            # > FWHM -> sigma conversion for all dimensions separately
            if i == 2:
                sig = fwhm2sig(psf, voxsize=Cnt['SZ_VOXZ'] * 10)
            else:
                sig = fwhm2sig(psf, voxsize=Cnt['SZ_VOXY'] * 10)

            x = np.arange(-Cnt['RSZ_PSF_KRNL'], Cnt['RSZ_PSF_KRNL'] + 1)
            kernel[i, :] = np.exp(-0.5 * (x**2 / sig**2))
            kernel[i, :] /= np.sum(kernel[i, :])

        psfkernel = np.empty((3, 2 * Cnt['RSZ_PSF_KRNL'] + 1), dtype=np.float32)
        psfkernel[0, :] = kernel[2, :]
        psfkernel[1, :] = kernel[0, :]
        psfkernel[2, :] = kernel[1, :]

        return psfkernel
#def psf(x, Cnt):
#    if Cnt['SIGMA_RM']:
#        x = ndi.gaussian_filter(x, sigma=Cnt['SIGMA_RM'], mode='constant', output=None)
#    else:
#        sig = fwhm2sig(2.5, voxsize=Cnt['SZ_VOXY'] * 10)
#        x = ndi.gaussian_filter(x, sigma=sig, mode='constant', output=None)
#    return x
    
def Scatter_EMML(sinog,datain,muMaps,scanner_params,hst,
    iterations=4,randomsinp=[],scatterinp=[],pred = None):
    #Cnt['SIGMA_RM'] = mmrrec.fwhm2sig(fwhm_rm, voxsize=Cnt['SZ_VOXZ'] * 10) if fwhm_rm else 0
    
    sinogramShape = sinog.shape  # (837, 252, 344)
    print("Using sinogram shape:", sinogramShape)
    
    Cnt = scanner_params['Cnt']
    if pred is None:
        pred = cu.ones((
            Cnt['SO_IMZ'],
            Cnt['SO_IMY'],
            Cnt['SO_IMX']
        ), dtype=np.float32)
    else:
        pred = pred
    if len(randomsinp) == 0:
        randoms = cu.zeros(sinog.shape)
    else:
        randoms = cu.asarray(randomsinp)
    if len(scatterinp) == 0:
        scatter = cu.zeros(sinog.shape)
    else:
        scatter = cu.asarray(scatterinp)
    muh, muo = muMaps

    ncmp, _ = getNormPara(datain,Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst, normcomp=ncmp, gpu_dim=False)
    
    muh, muo = muMaps
    mus = mmrimg.convert2dev(muo + muh, Cnt)
    _, fwd_mu_gaps = Forward(mus, muMaps, scanner_params, hst,True)
    fwd_mu_gaps = np.array(fwd_mu_gaps, dtype=np.float32)
    #fwd_mu_gaps = np.exp(-np.array(fwd_mu_gaps, dtype=np.float32))
    import functools
    fwhm = 2.5
    SIGMA2FWHMmm = (8 * np.log(2))**0.5 * np.array([scanner_params['Cnt']['SO_VX' + i] for i in 'ZYX']) * 10
    psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)
    sensitivity = Back(
        cu.asarray(nsng*fwd_mu_gaps, dtype=np.float32),
        muMaps, scanner_params, hst
    )
    
    randoms = div_nzer(randoms,nsng*fwd_mu_gaps)
    
    sensitivity = div_nzer(1,psf(sensitivity))
    msk = Mask(Cnt)
    
    #sensitivity = np.maximum(sensitivity, 1e-3)
    sensitivity[msk] = 0.0
    
    print("sensitivity min/max:", float(sensitivity.min()), float(sensitivity.max()))
    
    tref = hst['t0']
    lmbd = np.log(2) / resources.riLUT[Cnt['ISOTOPE']]['thalf']
    dcycrr = np.exp(lmbd * tref) * lmbd * hst['dur'] / (1 - np.exp(-lmbd * hst['dur']))
    # apply quantitative correction to the image
    qf = ncmp['qf'] / resources.riLUT[Cnt['ISOTOPE']]['BF'] / float(hst['dur'])
    qf_loc = ncmp['qf_loc']
    additive = cu.asarray(randoms, dtype=np.float32) + cu.asarray(scatter, dtype=np.float32)
    
    for i in range(iterations):
        

        
        fwd = Forward(psf(pred), muMaps, scanner_params, hst,False)[1]
        
        fwd += additive
        expected = fwd
        #expected = np.maximum(expected, 0.0)
        

        ratio = div_nzer(sinog,(expected))
        correction = Back(
            cu.asarray(ratio, dtype=np.float32),
            muMaps, scanner_params, hst
        )
        
        pred = ((pred) * sensitivity) * psf(correction)
        
        #pred  = np.maximum(pred, 0.0)

        #pred *= dcycrr*qf*qf_loc
        #pred = (pred * dcycrr * qf * qf_loc).astype(np.float32)
        
        if len(scatterinp) is not 0:
            
            scatter = Scatter(datain,muMaps,scanner_params,hst,randoms,pred)
        


        print(f"Iter {i+1} | pred min/max/mean: "
              f"{float(pred.min()):.4f} / {float(pred.max()):.4f} / {float(pred.mean()):.4f}")
        #pred[~msk] = 0.0  
    return pred#*dcycrr*qf*qf_loc

def Scatter_OSEM(sinog, datain, muMaps, scanner_params, hst,
    iterations=4, n_subsets=4, randomsinp=[], scatterinp=[], pred=None):

    sinogramShape = sinog.shape  # (837, 252, 344)
    print("Using sinogram shape:", sinogramShape)
    Cnt = scanner_params['Cnt']

    if pred is None:
        pred = cu.ones((
            Cnt['SO_IMZ'],
            Cnt['SO_IMY'],
            Cnt['SO_IMX']
        ), dtype=np.float32)

    if len(randomsinp) == 0:
        randoms = cu.zeros(sinog.shape)
    else:
        randoms = cu.asarray(randomsinp)

    if len(scatterinp) == 0:
        scatter = cu.zeros(sinog.shape)
    else:
        scatter = cu.asarray(scatterinp)

    muh, muo = muMaps
    ncmp, _ = getNormPara(datain, Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst, normcomp=ncmp, gpu_dim=False)

    mus = mmrimg.convert2dev(muo + muh, Cnt)
    _, fwd_mu_gaps = Forward(mus, muMaps, scanner_params, hst, True)
    fwd_mu_gaps = np.array(fwd_mu_gaps, dtype=np.float32)

    import functools
    fwhm = 2.5
    SIGMA2FWHMmm = (8 * np.log(2))**0.5 * np.array(
        [Cnt['SO_VX' + i] for i in 'ZYX']) * 10
    psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)

    msk = Mask(Cnt)
    nsng_fwd = nsng * fwd_mu_gaps

    # Interleaved subsets on axis 1 (angular) — zero angles dropped
    n_angles = sinog.shape[1]

    # Build interleaved order then split evenly — np.array_split handles remainder
    # by distributing one extra angle to the first (remainder) subsets
    interleaved = np.array([
        a for s in range(n_subsets) for a in range(s, n_angles, n_subsets)
    ])
    # Sort each subset's indices so the mask is contiguous-friendly for numpy
    subset_indices = [np.sort(chunk) for chunk in np.array_split(interleaved, n_subsets)]

    for s, idx in enumerate(subset_indices):
        print(f"  Subset {s}: {len(idx)} angles, "
              f"range [{idx.min()}, {idx.max()}]")

    # Pre-compute per-subset sensitivities
    subset_sensitivities = []
    for idx in subset_indices:
        masked_nsng = np.zeros_like(nsng_fwd)
        masked_nsng[:, idx, :] = nsng_fwd[:, idx, :]
        sens_sub = Back(
            cu.asarray(masked_nsng, dtype=np.float32),
            muMaps, scanner_params, hst
        )
        sens_sub = div_nzer(1, psf(sens_sub))
        sens_sub[msk] = 0.0
        subset_sensitivities.append(sens_sub)

    randoms = div_nzer(randoms, nsng_fwd)
    additive = cu.asarray(randoms, dtype=np.float32) + cu.asarray(scatter, dtype=np.float32)

    for i in range(iterations):
        for s, idx in enumerate(subset_indices):

            fwd = Forward(psf(pred), muMaps, scanner_params, hst, False)[1]
            fwd += additive

            ratio = div_nzer(sinog, fwd)

            masked_ratio = np.zeros_like(ratio)
            masked_ratio[:, idx, :] = ratio[:, idx, :]

            correction = Back(
                cu.asarray(masked_ratio, dtype=np.float32),
                muMaps, scanner_params, hst
            )

            pred = pred * subset_sensitivities[s] * psf(correction)
            #slap a DDIP here
        if len(scatterinp) != 0:
            scatter = Scatter(datain, muMaps, scanner_params, hst, randoms, pred)

        print(f"Iter {i+1} | pred min/max/mean: "
              f"{float(pred.min()):.4f} / {float(pred.max()):.4f} / {float(pred.mean()):.4f}")

    return pred


def gradient(x: np.ndarray,gamma=2.0,beta=0.1) -> np.ndarray:
    
    x64 = np.asarray(x, dtype=np.float64)   # plain ndarray, full precision
    g   = np.zeros_like(x64)

    for axis in range(x64.ndim):
        for shift in (-1, +1):
            xk = np.roll(x64, shift=shift, axis=axis)

            # Zero-out the wrap-around boundary (→ zero BC, not periodic)
            sl        = [slice(None)] * x64.ndim
            sl[axis]  = 0 if shift == 1 else -1
            xk[tuple(sl)] = 0.0

            diff   = x64 - xk
            denom  = x64 + xk + gamma * np.abs(diff)
            # Guard divide-by-zero (only occurs when both voxels are 0)
            sdenom = np.where(denom > 0, denom, 1.0)

            # Analytic derivative of (x_j - x_k)^2 / denom  wrt x_j
            numer = (2.0 * diff * sdenom
                        - diff**2 * (1.0 + gamma * np.sign(diff)))
            g += numer / sdenom**2

    # R = -beta * Σ(...), so grad R = -beta * g
    # Cast back to input dtype (float32) for consistency with pred
    return (-beta * g).astype(x.dtype)


def _alpha(global_subit: int, alpha_0: float, alpha_decay: float) -> float:
    """
    Harmonic decay:  alpha_0 / (1 + global_subit / alpha_decay).

    Guaranteed to satisfy Σ alpha_n = ∞  and  Σ alpha_n^2 < ∞  when
    alpha_decay > 0, which is the standard sufficient condition for BSREM
    convergence (Ahn & Fessler 2003).
    """
    return alpha_0 / (1.0 + global_subit / alpha_decay)


import functools
def Scatter_BSREM(
    sinog,
    datain,
    muMaps,
    scanner_params,
    hst,
    # ── iteration control (same names/defaults as Scatter_OSEM) ─────────────
    iterations: int   = 100,
    n_subsets:  int   = 10,
    randomsinp        = [],
    scatterinp        = [],
    pred              = None,
    # ── BSREM-specific (new parameters) ─────────────────────────────────────
    beta:        float = 0.3,
    gamma:       float = 2.0,
    alpha_0:     float = 1.0,
    alpha_decay: float = None,  # defaults to n_subsets * iterations / 3
    delta_frac:  float = 1e-4,
):
    """
    BSREM-RDP PET reconstruction.  Drop-in replacement for Scatter_OSEM.
 
    All parameters up to `pred` are identical to Scatter_OSEM.
    New parameters (beta, gamma, alpha_0, alpha_decay, delta_frac) have
    sensible defaults and are described in the module docstring.
 
    Returns
    -------
    pred : CuVec array  (Z, Y, X), float32
    """
 
    # ── 0. Setup — identical to Scatter_OSEM ────────────────────────────────
    sinogramShape = sinog.shape        # (radial_bins, angles, axial)
    print("Using sinogram shape:", sinogramShape)
 
    Cnt = scanner_params['Cnt']
    
    if pred is None:
        pred = cu.ones((
            Cnt['SO_IMZ'],
            Cnt['SO_IMY'],
            Cnt['SO_IMX'],
        ), dtype=np.float32)
 
    if len(randomsinp) == 0:
        randoms = cu.zeros(sinog.shape)
    else:
        randoms = cu.asarray(randomsinp)
 
    if len(scatterinp) == 0:
        scatter = cu.zeros(sinog.shape)
    else:
        scatter = cu.asarray(scatterinp)
 
    muh, muo  = muMaps
    ncmp, _   = getNormPara(datain, Cnt)
    nsng      = mmrnorm.get_norm_sino(datain, scanner_params, hst,
                                      normcomp=ncmp, gpu_dim=False)
 
    mus             = mmrimg.convert2dev(muo + muh, Cnt)
    _, fwd_mu_gaps  = Forward(mus, muMaps, scanner_params, hst, True)
    fwd_mu_gaps     = np.array(fwd_mu_gaps, dtype=np.float32)
 
    fwhm         = 2.5
    SIGMA2FWHMmm = (8 * np.log(2))**0.5 * np.array(
        [Cnt['SO_VX' + i] for i in 'ZYX']) * 10
    psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)
 
    msk      = Mask(Cnt)
    nsng_fwd = nsng * fwd_mu_gaps
    pred[msk] = 0.0
    # ── 1. Interleaved subsets — identical to Scatter_OSEM ──────────────────
    n_angles    = sinog.shape[1]
    interleaved = np.array([
        a for s in range(n_subsets) for a in range(s, n_angles, n_subsets)
    ])
    subset_indices = [
        np.sort(chunk) for chunk in np.array_split(interleaved, n_subsets)
    ]
    for s, idx in enumerate(subset_indices):
        print(f"  Subset {s}: {len(idx)} angles, "
              f"range [{idx.min()}, {idx.max()}]")
 
    # ── 2. Pre-compute per-subset sensitivity images ─────────────────────────
    #
    #  OSEM stored 1/psf(sens) so it could multiply directly.
    #  BSREM needs  D(x) = (x + delta) / sens,  so we store the raw (non-
    #  inverted) blurred sensitivity and divide at update time.  This lets us
    #  incorporate delta, which must be recomputed each sub-iteration.
    #
    subset_sensitivities = []
    for idx in subset_indices:
        masked_nsng             = np.zeros_like(nsng_fwd)
        masked_nsng[:, idx, :] = nsng_fwd[:, idx, :]
        sens_sub = Back(
            cu.asarray(masked_nsng, dtype=np.float32),
            muMaps, scanner_params, hst
        )
        sens_sub        = (sens_sub/n_subsets)#psf    # PSF-blurred, matches OSEM convention
        sens_sub[msk]   = 0.0
        subset_sensitivities.append(sens_sub)
 
    # ── 3. Additive correction — identical to Scatter_OSEM ──────────────────
    randoms  = div_nzer(randoms, nsng_fwd)
    additive = (cu.asarray(randoms, dtype=np.float32)
                + cu.asarray(scatter, dtype=np.float32))
 
    # ── 4. BSREM-specific setup ──────────────────────────────────────────────
    if alpha_decay is None:
        alpha_decay = float(n_subsets * iterations) / 3.0
 
    #prior        = RelativeDifferencePrior(beta=beta, gamma=gamma)
    global_subit = 0   # sub-iteration counter across all outer iterations
 
    print(f"\nBSREM: {iterations} iters × {n_subsets} subsets"
          f" | beta={beta}, gamma={gamma}"
          f" | alpha_0={alpha_0}, alpha_decay={alpha_decay:.1f}")
 
    # ── 5. Main reconstruction loop ──────────────────────────────────────────
    for i in range(iterations):
 
        for s, idx in enumerate(subset_indices):
 
            alpha = _alpha(global_subit, alpha_0, alpha_decay)
 
            # ── 5a. Forward projection — identical to Scatter_OSEM ───────────
            fwd  = Forward((pred), muMaps, scanner_params, hst, False)[1]#psf
            fwd += additive
            # Guard log(0): CuVec is a numpy subclass so np.maximum works
            fwd  = np.maximum(fwd, 1e-15)
 
            # ── 5b. Subset likelihood gradient ───────────────────────────────
            #
            #  OSEM:   ratio = y / yhat            (multiplicative correction)
            #  BSREM:  ratio = y / yhat - 1        (gradient of log-likelihood)
            #
            #  Scaled by n_subsets so the subset gradient approximates the
            #  full-data gradient — same scaling used by the sensitivity image.
            #
            ratio_bsrem             = div_nzer(sinog, fwd) #- 1.0
            masked_ratio            = np.zeros_like(ratio_bsrem)
            masked_ratio[:, idx, :] = ratio_bsrem[:, idx, :]
 
            grad_L = Back(
                cu.asarray(masked_ratio, dtype=np.float32),
                muMaps, scanner_params, hst
            )
            grad_L = (grad_L * n_subsets)#psf   # scale to full-data gradient
 
            # ── 5c. RDP prior gradient ────────────────────────────────────────
            #
            #  pred is a CuVec (numpy subclass) so it can be passed directly.
            #  gradient() converts internally to float64 for stability, then
            #  returns a plain float32 ndarray — compatible with CuVec arithmetic.
            #
            grad_R = gradient(pred,gamma,beta)
 
            # ── 5d. EM preconditioner  D(x) = (x + delta) / sens ─────────────
            #
            #  delta is recomputed each sub-iteration from the running image max
            #  so it scales automatically with the activity level.
            #
            delta  = max(float(delta_frac * float(pred.max())), 1e-8)
            sens   = subset_sensitivities[s]          # plain ndarray, float32
            D      = (pred + delta) / np.maximum(sens, 1e-15)
            D[msk] = 0.0                              # zero outside FOV
 
            # ── 5e. BSREM update ──────────────────────────────────────────────
            #
            #  OSEM:   pred = pred * (1/sens) * psf(correction)
            #  BSREM:  pred = max(0, pred + alpha * D * (grad_L + grad_R))
            #
            pred  = pred + alpha * D * (grad_L + (grad_R))
            pred  = np.maximum((pred), 0.0)             # non-negativity projection
            pred[msk] = 0.0
            global_subit += 1
 
        # ── 6. Optional scatter re-estimation — identical to Scatter_OSEM ───
        if len(scatterinp) != 0:
            scatter  = Scatter(datain, muMaps, scanner_params, hst, randoms, pred)
            additive = (cu.asarray(randoms, dtype=np.float32)
                        + cu.asarray(scatter, dtype=np.float32))
        print("PP PP PP PP PP PP PP PP PP PP PP PP PP PP PP PP ")
        print("PP PP PP PP PP PP PP PP PP PP PP PP PP PP PP PP ")
        print("PP PP PP PP PP PP PP PP PP PP PP PP PP PP PP PP ")
        print(f"\x1b[31m\"Iter {i+1:4d} | alpha={_alpha(global_subit, alpha_0, alpha_decay):.5f}| pred min/max/mean: {float(pred.min()):.4f} / {float(pred.max()):.4f} / {float(pred.mean()):.4f}\"\x1b[0m")
 
    return pred




def Scatter_BSREM_Updated(
    sinog,
    datain,
    muMaps,
    scanner_params,
    hst,
    # Iteration control — same names/defaults as Scatter_OSEM
    iterations: int   = 100,
    n_subsets:  int   = 10,
    randomsinp        = [],
    scatterinp        = [],
    pred              = None,
    # BSREM-specific parameters
    beta:        float = 0.3,
    gamma:       float = 2.0,
    alpha_0:     float = 1.0,
    alpha_decay: float = None,   # defaults to n_subsets * iterations / 3
    delta_frac:  float = 1e-4,
):
    """
    BSREM-RDP PET reconstruction.  Drop-in replacement for Scatter_OSEM.
    Returns pred : CuVec array (Z, Y, X), float32.
    """

    # ── 0. Setup — identical to Scatter_OSEM ────────────────────────────────
    sinogramShape = sinog.shape
    print("Using sinogram shape:", sinogramShape)

    Cnt = scanner_params['Cnt']

    if pred is None:
        pred = cu.ones((
            Cnt['SO_IMZ'],
            Cnt['SO_IMY'],
            Cnt['SO_IMX'],
        ), dtype=np.float32)

    if len(randomsinp) == 0:
        randoms = cu.zeros(sinog.shape)
    else:
        randoms = cu.asarray(randomsinp)

    if len(scatterinp) == 0:
        scatter = cu.zeros(sinog.shape)
    else:
        scatter = cu.asarray(scatterinp)

    muh, muo  = muMaps
    ncmp, _   = getNormPara(datain, Cnt)
    nsng      = mmrnorm.get_norm_sino(datain, scanner_params, hst,
                                      normcomp=ncmp, gpu_dim=False)
    mus             = mmrimg.convert2dev(muo + muh, Cnt)
    _, fwd_mu_gaps  = Forward(mus, muMaps, scanner_params, hst, True)
    fwd_mu_gaps     = np.array(fwd_mu_gaps, dtype=np.float32)

    fwhm         = 2.5
    SIGMA2FWHMmm = (8 * np.log(2))**0.5 * np.array(
        [Cnt['SO_VX' + i] for i in 'ZYX']) * 10
    psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)

    msk      = Mask(Cnt)
    pred[msk] = 0.0          # zero outside FOV before first forward projection
    nsng_fwd = nsng * fwd_mu_gaps

    # ── 1. Interleaved subsets — identical to Scatter_OSEM ──────────────────
    n_angles    = sinog.shape[1]
    interleaved = np.array([
        a for s in range(n_subsets) for a in range(s, n_angles, n_subsets)
    ])
    subset_indices = [
        np.sort(chunk) for chunk in np.array_split(interleaved, n_subsets)
    ]
    for s, idx in enumerate(subset_indices):
        print(f"  Subset {s}: {len(idx)} angles, "
              f"range [{idx.min()}, {idx.max()}]")

    # ── 2. Pre-compute per-subset sensitivities ──────────────────────────────
    #
    #  s_sub_j = PSF( A_s^T[ nsng_fwd_s ] )
    #
    #  These serve two roles:
    #    a) The denominator of D_s(x) = (x+delta) / s_sub   (EM preconditioner)
    #    b) The constant subtracted from PSF(A_s^T[ratio_s]) to centre the
    #       likelihood gradient at zero, matching the OSEM correction form.
    #
    #  Using s_sub (not s_full) in D_s gives exactly the OSEM step size for
    #  the subset, which is the correct normalisation.
    #
    subset_sensitivities = []
    for idx in subset_indices:
        masked_nsng             = np.zeros_like(nsng_fwd)
        masked_nsng[:, idx, :]  = nsng_fwd[:, idx, :]
        s_sub = Back(
            cu.asarray(masked_nsng, dtype=np.float32),
            muMaps, scanner_params, hst
        )
        s_sub       = (s_sub)#psf
        s_sub[msk]  = 0.0
        subset_sensitivities.append(s_sub)

    # ── 3. Additive correction — identical to Scatter_OSEM ──────────────────
    randoms  = div_nzer(randoms, nsng_fwd)
    additive = (cu.asarray(randoms, dtype=np.float32)
                + cu.asarray(scatter, dtype=np.float32))

    # ── 4. BSREM-specific setup ──────────────────────────────────────────────
    if alpha_decay is None:
        alpha_decay = float(n_subsets * iterations) / 3.0

    #prior        = RelativeDifferencePrior(beta=beta, gamma=gamma)
    global_subit = 0

    print(f"\nBSREM: {iterations} iters x {n_subsets} subsets"
          f" | beta={beta}, gamma={gamma}"
          f" | alpha_0={alpha_0}, alpha_decay={alpha_decay:.1f}")

    # ── 5. Main reconstruction loop ──────────────────────────────────────────
    for i in range(iterations):

        for s, idx in enumerate(subset_indices):

            alpha = _alpha(global_subit, alpha_0, alpha_decay)
            s_sub = subset_sensitivities[s]

            # ── 5a. Forward projection — identical to Scatter_OSEM ───────────
            fwd  = Forward((pred), muMaps, scanner_params, hst, False)[1]#psf
            fwd += additive
            fwd  = np.maximum(fwd, 1e-15)

            # ── 5b. Subset likelihood gradient ───────────────────────────────
            #
            #  Compute PSF( A_s^T[ y_s / yhat_s ] ) — same as OSEM numerator —
            #  then subtract s_sub to centre the gradient at zero.
            #
            #  This is the additive rewrite of the OSEM correction:
            #    OSEM:   x_new = x * (1/s_sub) * bp_ratio
            #    BSREM:  x_new = x + D_s * (bp_ratio - s_sub)
            #  where D_s = (x+delta)/s_sub and bp_ratio = PSF(A_s^T[y/yhat]).
            #
            #  Do NOT backproject (ratio - 1) and scale by n_subsets.
            #  That requires A_s^T[1_s] = s_sub exactly, which fails with
            #  real non-uniform attenuation/normalisation and causes a
            #  flower artifact with as many petals as n_subsets.
            #
            ratio                   = div_nzer(sinog, fwd)        # y / yhat, >= 0
            masked_ratio            = np.zeros_like(ratio)
            masked_ratio[:, idx, :] = ratio[:, idx, :]

            bp_ratio = Back(
                cu.asarray(masked_ratio, dtype=np.float32),
                muMaps, scanner_params, hst
            )
            bp_ratio = (bp_ratio)#psf

            grad_L = bp_ratio - s_sub                             # centred correction

            # ── 5c. RDP prior gradient ────────────────────────────────────────
            #
            #  Scaled by 1/n_subsets so that after one full pass of n_subsets
            #  sub-iterations the net prior contribution equals one full
            #  prior gradient step (Ahn & Fessler 2003, eq. 14).
            #
            grad_R = gradient(pred,gamma,beta) / n_subsets

            # ── 5d. EM preconditioner  D_s(x) = (x + delta) / s_sub ──────────
            #
            #  Uses per-subset sensitivity s_sub (not s_full) to give the same
            #  step magnitude as the OSEM multiplicative update for this subset.
            #
            delta   = max(float(delta_frac * float(pred.max())), 1e-8)
            D_s     = (pred + delta) / np.maximum(s_sub, 1e-15)
            D_s[msk] = 0.0

            # ── 5e. BSREM update ──────────────────────────────────────────────
            pred  = pred + alpha * D_s * (grad_L + grad_R)
            pred  = np.maximum(pred, 0.0)    # non-negativity
            pred[msk] = 0.0                  # FOV mask

            global_subit += 1

        # ── 6. Optional scatter re-estimation — identical to Scatter_OSEM ────
        if len(scatterinp) != 0:
            scatter  = Scatter(datain, muMaps, scanner_params, hst, randoms, pred)
            additive = (cu.asarray(randoms, dtype=np.float32)
                        + cu.asarray(scatter, dtype=np.float32))
        
        print(f"\x1b[31m\"Iter {i+1:4d} | alpha={_alpha(global_subit, alpha_0, alpha_decay):.5f}| pred min/max/mean: {float(pred.min()):.4f} / {float(pred.max()):.4f} / {float(pred.mean()):.4f}\"\x1b[0m")
 

    return pred


def SigMask(Cnt, rad=27.):
    """
    Signa FOV mask, using nipet.sigaux.get_cylinder (not mmrimg.get_cylinder,
    which is mMR-specific and would silently produce a wrong-shaped/wrong-grid
    mask here). rad=27 matches the value used in your working recon script.
    """
    from niftypet import nipet as _nipet  # local import to avoid altering module-level imports
    msk = _nipet.sigaux.get_cylinder(
        Cnt, rad=rad, xo=0, yo=0, unival=1, gpu_dim=True, mask=True)
    return ~np.asarray(msk).astype(bool)   # invert: Mask() convention here is True = OUTSIDE FOV

def Signa_BSREM(
    sinog,              # measured prompts, shape (Cnt['NAW'], nsinos)
    nrmcmp,              # dict with Signa norm components, e.g. {'nrm': nrm, 'dtpu': dtpu}
    muMaps,              # (mu_hardware, mu_object), RAW loader axis order (Z,Y,X)
    scanner_params,      # dict(Cnt=Cnt, txLUT=txLUT, axLUT=axLUT) from nipet.sigaux.init_sig
    hst,
    iterations: int = 100,
    n_subsets:  int = 10,
    randomsinp        = [],
    scatterinp        = [],
    pred              = None,
    beta:        float = 0.3,
    gamma:       float = 2.0,
    alpha_0:     float = 1.0,
    alpha_decay: float = None,
    delta_frac:  float = 1e-4,
):
    """
    BSREM-RDP PET reconstruction for GE Signa data. Structural drop-in for
    Scatter_BSREM_Updated -- only how sinograms/normalisation/dead-time are
    obtained, how mu-maps are axis-corrected, and how forward/back
    projection is called has changed. BSREM-RDP update math is unchanged.
    Returns pred : CuVec array (Y, X, Z) float32 (Signa image-grid order).
    """

    # ── 0. Setup ──────────────────────────────────────────────────────────
    sinogramShape = sinog.shape
    print("Using sinogram shape:", sinogramShape)

    Cnt, txLUT, axLUT = scanner_params['Cnt'], scanner_params['txLUT'], scanner_params['axLUT']
    im_shape = (Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ'])

    if pred is None:
        pred = cu.ones(im_shape, dtype=np.float32)

    if len(randomsinp) == 0:
        randoms = cu.zeros(sinog.shape)
    else:
        randoms = cu.asarray(randomsinp)

    if len(scatterinp) == 0:
        scatter = cu.zeros(sinog.shape)
    else:
        scatter = cu.asarray(scatterinp)

    muh, muo = muMaps

    # Signa gives pre-generated norm/dead-time sinograms rather than mMR-style
    # norm components + a runtime get_norm_sino() call -- combine as your
    # read-in script does (nrm * dtpu):
    nsng = nrmcmp['nrm'] * nrmcmp['dtpu']
    nsng[:, range(1, 89, 2)] *= 2

    # ── fix mu-map axis order: loader gives (Z,Y,X), projector needs
    # (SZ_IMY, SZ_IMX, SZ_IMZ) -- confirmed necessary earlier (mu.shape was
    # (89,288,288) against expected (288,288,89)).
    def _fix_mumap_axes(mu):
        mu = np.asarray(mu)
        if mu.shape == im_shape:
            return mu
        mu_fixed = np.transpose(mu, (1, 2, 0))
        assert mu_fixed.shape == im_shape, (
            f"mu-map shape {mu.shape} doesn't match im_shape {im_shape} even "
            f"after (1,2,0) transpose -- got {mu_fixed.shape}."
        )
        return mu_fixed

    muh_fixed = _fix_mumap_axes(muh)
    muo_fixed = _fix_mumap_axes(muo)
    mus = cu.asarray(muo_fixed + muh_fixed, dtype=np.float32)

    # forward-project the (axis-corrected) mu-map to get the attenuation
    # sinogram. attenuation=True, matching the mMR original's
    # Forward(mus, muMaps, scanner_params, hst, True). The previous draft
    # forward-projected `pred` (not `mus`) with attenuation=False, which
    # produced a meaningless attenuation sinogram.
    fwd_mu = SigForward(mus, scanner_params, attenuation=True)
    fwd_mu_gaps = np.array(fwd_mu, dtype=np.float32)

    fwhm = 2.5
    SIGMA2FWHMmm = (8 * np.log(2))**0.5 * np.array(
        [Cnt['SO_VX' + i] for i in 'ZYX']) * 10
    psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)

    # Signa FOV mask -- SigMask, not Mask (Mask() uses mmrimg.get_cylinder,
    # which is mMR-specific and would be wrong here).
    msk = SigMask(Cnt)
    pred[msk] = 0.0
    nsng_fwd = nsng * fwd_mu_gaps

    # ── 1. Interleaved subsets ────────────────────────────────────────────
    # Confirmed correct: reshape(n_bins, n_angles, -1) on the flat (NAW,
    # nsinos) sinogram matches get_txLUT's iw*NSANGLES + ia indexing (bin
    # outer, angle inner) -- no further transpose needed here (a .T was
    # only ever needed for matshow display, not for indexing correctness).
    n_bins   = Cnt['NSBINS']
    n_angles = Cnt['NSANGLES']
    assert n_bins * n_angles == Cnt['NAW'], \
        "NSBINS*NSANGLES != NAW -- check reshape order for your Signa Cnt"

    def to_bab(x_naw):
        return x_naw.reshape(n_bins, n_angles, -1)

    def to_naw(x_bab):
        return x_bab.reshape(Cnt['NAW'], -1)

    interleaved = np.array([
        a for s in range(n_subsets) for a in range(s, n_angles, n_subsets)
    ])
    subset_indices = [
        np.sort(chunk) for chunk in np.array_split(interleaved, n_subsets)
    ]
    for s, idx in enumerate(subset_indices):
        print(f"  Subset {s}: {len(idx)} angles, "
              f"range [{idx.min()}, {idx.max()}]")

    # ── 2. Pre-compute per-subset sensitivities ──────────────────────────
    subset_sensitivities = []
    for idx in subset_indices:
        masked_nsng = np.zeros_like(nsng_fwd)
        masked_bab = to_bab(masked_nsng)
        masked_bab[:, idx, :] = to_bab(nsng_fwd)[:, idx, :]
        masked_nsng = to_naw(masked_bab)

        s_sub_arr = SigBack(masked_nsng, scanner_params)

        s_sub = np.array(s_sub_arr)  # psf
        s_sub[msk] = 0.0
        subset_sensitivities.append(s_sub)

    # ── 3. Additive correction ──────────────────────────────────────────
    randoms  = div_nzer(randoms, nsng_fwd)
    additive = (cu.asarray(randoms, dtype=np.float32)
                + cu.asarray(scatter, dtype=np.float32))

    # ── 4. BSREM-specific setup ───────────────────────────────────────────
    if alpha_decay is None:
        alpha_decay = float(n_subsets * iterations) / 3.0

    global_subit = 0
    print(f"\nSigna BSREM: {iterations} iters x {n_subsets} subsets"
          f" | beta={beta}, gamma={gamma}"
          f" | alpha_0={alpha_0}, alpha_decay={alpha_decay:.1f}")

    # ── 5. Main reconstruction loop ───────────────────────────────────────
    for i in range(iterations):

        for s, idx in enumerate(subset_indices):

            alpha = _alpha(global_subit, alpha_0, alpha_decay)
            s_sub = subset_sensitivities[s]

            # ── 5a. Forward projection ────────────────────────────────────
            # Fixed: previous draft's return value was discarded (fwd_arr
            # never assigned), so fwd was always zero. attenuation=False
            # here since attenuation is already folded into nsng_fwd /
            # additive / s_sub above.
            fwd_arr = SigForward(cu.asarray(pred, dtype=np.float32),
                                  scanner_params, attenuation=False)
            fwd = np.array(fwd_arr)  # psf
            fwd += additive
            fwd  = np.maximum(fwd, 1e-15)

            # ── 5b. Subset likelihood gradient ────────────────────────────
            ratio = div_nzer(sinog, fwd)
            masked_ratio = np.zeros_like(ratio)
            masked_bab = to_bab(masked_ratio)
            masked_bab[:, idx, :] = to_bab(ratio)[:, idx, :]
            masked_ratio = to_naw(masked_bab)

            bp_ratio_arr = SigBack(masked_ratio, scanner_params)
            bp_ratio = np.array(bp_ratio_arr)  # psf

            grad_L = bp_ratio - s_sub

            # ── 5c. RDP prior gradient ────────────────────────────────────
            grad_R = gradient(pred, gamma, beta) / n_subsets

            # ── 5d. EM preconditioner ────────────────────────────────────
            delta = max(float(delta_frac * float(pred.max())), 1e-8)
            D_s = (pred + delta) / np.maximum(s_sub, 1e-15)
            D_s[msk] = 0.0

            # ── 5e. BSREM update ──────────────────────────────────────────
            pred = pred + alpha * D_s * (grad_L + grad_R)
            pred = np.maximum(pred, 0.0)
            pred[msk] = 0.0

            global_subit += 1

        if len(scatterinp) != 0:
            scatter  = Scatter(datain, muMaps, scanner_params, hst, randoms, pred)
            additive = (cu.asarray(randoms, dtype=np.float32)
                        + cu.asarray(scatter, dtype=np.float32))

        print(f"\x1b[31m\"Iter {i+1:4d} | alpha={_alpha(global_subit, alpha_0, alpha_decay):.5f}"
              f"| pred min/max/mean: {float(pred.min()):.4f} / {float(pred.max()):.4f} / {float(pred.mean()):.4f}\"\x1b[0m")

    return pred



















 






















def get_neighbors(image):
    import itertools
    ndim = image.ndim
    offsets = list(itertools.product([-1,0,1], repeat=ndim))
    offsets.remove((0,)*ndim)
    neighbors = []
    for off in offsets:
        dist = np.sqrt(sum(o**2 for o in off))
        weight = 1.0 / dist
        # Use slicing instead of roll to avoid wrap-around
        shifted = np.pad(image, 1, mode='edge')  # or 'constant' with 0
        slices_src = tuple(
            slice(1 + o, s + 1 + o) for o, s in zip(off, image.shape)
        )
        neighbors.append((shifted[slices_src], weight))
    return neighbors
def rdp_gradient(image, gamma=2.0): #relative difference prior
    image = np.array(image)
    eps = 1e-10  # numerical stability floor
    grad = np.zeros_like(image, dtype=np.float64) #initialise per voxel gradient
    neighbors = get_neighbors(image) #get neighbouring voxels

    for (x_k, w_jk) in neighbors:
        diff   = image - x_k                              # x_j - x_k
        ssum   = image + x_k + gamma * np.abs(diff) + eps # denominator
        absdif = np.abs(diff)

        # Numerator derivative d/dx_j [(x_j - x_k)^2 / (x_j + x_k + γ|x_j-x_k|)]
        # Using quotient rule and chain rule for the absolute value
        #literally performing a derivative:
        sign_diff = np.sign(diff)
        num       = diff ** 2
        dnum_dxj  = 2.0 * diff
        dden_dxj  = 1.0 + gamma * sign_diff   # d/dx_j of denominator
        #w_jk = voxel j
        g = w_jk * (dnum_dxj * ssum - num * dden_dxj) / (ssum ** 2)
        grad += g
    return -grad*0.3
def bsrem_update_subset(
    image, measured, angles_deg, subset_idx,
    sensitivity, beta, gamma, step_scale,
    muMaps, scanner_params, hst, additive,
    nsng_fwd,n_subsets                               # add this
):
    image_np = image
    sens_np  = sensitivity

    Cnt = scanner_params['Cnt']
    msk = Mask(Cnt)
    import functools
    fwhm = 2.5
    SIGMA2FWHMmm = (8 * np.log(2))**0.5 * np.array(
        [Cnt['SO_VX' + i] for i in 'ZYX']) * 10
    psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)

    # Forward project
    predicted_sub = (
        Forward(cu.asarray((image_np), dtype=np.float32),
                muMaps, scanner_params, hst, True)[1]
    ) + (additive)#psf
    #predicted_sub = np.maximum(predicted_sub, 1e-10)

    # Ratio weighted by nsng_fwd — matches how sensitivity was computed
    ratio = div_nzer(measured, predicted_sub)
    masked_ratio = np.zeros_like(ratio)
    masked_ratio[:, subset_idx, :] = (ratio)[:, subset_idx, :] #* div_nzer(1.0,nsng_fwd[:, subset_idx, :])

    bp_ratio = (
        Back(cu.asarray(masked_ratio, dtype=np.float32),
             muMaps, scanner_params, hst)
    )#psf

    
    ll_gradient = n_subsets*bp_ratio - div_nzer(1.0,sens_np)

    # Overflow-safe RDP gradient
    image_np64 = image_np.astype(np.float64)
    #scale = image_np64.max()
    #if scale > 1e-10:
    image_norm = image_np64 #/ scale
    pen_gradient = rdp_gradient(image_norm, gamma=gamma) #/ scale
    #else:
    #    pen_gradient = np.zeros_like(image_np64)

    denom = sens_np#np.maximum(sens_np, 1e-10)
    delta = (image_np * denom) * (ll_gradient - beta * pen_gradient)

    # Guard against NaN/Inf before applying
    #delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

    image_new = image_np + step_scale * delta
    #image_new = np.maximum(image_new, 1e-10)
    #image_new[msk] = 0.0

    return cu.asarray(image_new, dtype=np.float32)

def qclear_reconstruct(
    sinogram,
    angles_deg,
    muMaps,
    scanner_params, 
    hst,
    datain,
    n_iterations:    int   = 20,
    beta:            float = 350.0,
    gamma:           float = 2.0,
    n_subsets:       int   = 1,
    
    background= None,
    init_image=None,
    convergence_tol: float = 1e-5,
    verbose:         bool  = True,
    randomsinp=[], scatterinp=[]
):
    """
    Q.Clear PET image reconstruction (BSREM with Relative Difference Penalty).

    Unlike OSEM, BSREM runs to practical convergence; the β parameter alone
    controls the noise–resolution trade-off.

    Parameters
    ----------
    sinogram     : (A, N) measured PET projection data (integer counts)
    angles_deg   : (A,)   projection angles in degrees
    beta         : regularisation strength — higher β = smoother image.
                   Clinical range: 50 (sharp) to 1000 (smooth).
                   GE default is β = 350.
    gamma        : edge-preservation factor in the RDP (default 2).
                   γ = 2  →  standard Q.Clear behaviour.
    n_subsets    : number of OSEM-style subsets per full pass
    n_iterations : maximum number of full passes (each pass = n_subsets sub-iters)
    background   : (A, N) sinogram of scatter + randoms; zeros if None
    init_image   : (N, N) starting image; uniform if None
    convergence_tol: relative change in image norm below which to stop early
    verbose      : print iteration progress

    Returns
    -------
    image    : (N, N) reconstructed activity image
    obj_vals : list of penalised log-likelihood values per full iteration
    """
    S, A, N = sinogram.shape
    assert (angles_deg) == A, "angles_deg length must match sinogram rows."

    if background is None:
        background = np.zeros_like(sinogram)
    Cnt = scanner_params['Cnt']
    image = cu.ones((
            Cnt['SO_IMZ'],
            Cnt['SO_IMY'],
            Cnt['SO_IMX']
        ), dtype=np.float32)

    # Pre-compute sensitivity once
    if verbose:
        print("Pre-computing sensitivity image …")
    sinogramShape = sinogram.shape  # (837, 252, 344)
    print("Using sinogram shape:", sinogramShape)

    if len(randomsinp) == 0:
        randoms = cu.zeros(sinogram.shape)
    else:
        randoms = cu.asarray(randomsinp)###
    if len(scatterinp) == 0:
        scatter = cu.zeros(sinogram.shape)
    else:
        scatter = cu.asarray(scatterinp)

    muh, muo = muMaps
    ncmp, _ = getNormPara(datain, Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst, normcomp=ncmp, gpu_dim=False)

    mus = mmrimg.convert2dev(muo + muh, Cnt)
    _, fwd_mu_gaps = Forward(mus, muMaps, scanner_params, hst, True)
    fwd_mu_gaps = np.array(fwd_mu_gaps, dtype=np.float32)

    import functools
    fwhm = 2.5
    SIGMA2FWHMmm = (8 * np.log(2))**0.5 * np.array(
        [Cnt['SO_VX' + i] for i in 'ZYX']) * 10
    psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)

    msk = Mask(Cnt)
    nsng_fwd = nsng * fwd_mu_gaps

    # Interleaved subsets on axis 1 (angular) — zero angles dropped
    n_angles = sinogram.shape[1]

    # Build interleaved order then split evenly — np.array_split handles remainder
    # by distributing one extra angle to the first (remainder) subsets
    interleaved = np.array([
        a for s in range(n_subsets) for a in range(s, n_angles, n_subsets)
    ])
    # Sort each subset's indices so the mask is contiguous-friendly for numpy
    subset_indices = [np.sort(chunk) for chunk in np.array_split(interleaved, n_subsets)]

    for s, idx in enumerate(subset_indices):
        print(f"  Subset {s}: {len(idx)} angles, "
              f"range [{idx.min()}, {idx.max()}]")

    # Pre-compute per-subset sensitivities
    subset_sensitivities = []
    for idx in subset_indices:
        masked_nsng = np.zeros_like(nsng_fwd)
        masked_nsng[:, idx, :] = nsng_fwd[:, idx, :]
        sens_sub = Back(
            cu.asarray(masked_nsng, dtype=np.float32),
            muMaps, scanner_params, hst
        )
        sens_sub = div_nzer(1,(sens_sub))#psf
        sens_sub[msk] = 0
        #sens_sub = np.maximum((sens_sub), 1e-10)
        #
        
        subset_sensitivities.append(sens_sub)
    randoms = div_nzer(randoms, nsng_fwd)
    additive = cu.asarray(randoms, dtype=np.float32) + cu.asarray(scatter, dtype=np.float32)
    # Build ordered subsets (angle indices partitioned into n_subsets groups)
    #all_indices   = np.arange(A)
    #subset_lists  = [all_indices[i::n_subsets] for i in range(n_subsets)]

    obj_vals = []
    
    for iteration in range(1, n_iterations + 1):
        image_prev = image.copy()
        alpha_n = 1.0 / (1 + iteration / 20.0)
        # --- One full pass: cycle through all subsets ---
        for s, idx in enumerate(subset_indices):
            image = bsrem_update_subset(
                image          = image,
                measured       = sinogram,
                angles_deg     = angles_deg,
                subset_idx     = idx,
                sensitivity    = subset_sensitivities[s],
                beta           = beta,
                gamma          = gamma,
                step_scale     = 1.0,
                muMaps         = muMaps,
                scanner_params = scanner_params, 
                hst            = hst,
                additive       = additive,
                nsng_fwd       = nsng_fwd,
                n_subsets      = n_subsets,
            )

            

        # --- Compute penalised log-likelihood for monitoring ---
        predicted_full = Forward(image, muMaps, scanner_params, hst, True)[1]+ additive
        predicted_full = np.maximum(predicted_full, 1e-10)
        log_lik = np.sum(sinogram * np.log(predicted_full) - predicted_full)
        pen     = beta * np.sum(np.abs(rdp_gradient(image, gamma)))   # proxy for R(x)
        obj     = log_lik - pen
        obj_vals.append(obj)

        # --- Convergence check ---
        rel_change = np.linalg.norm(image - image_prev) / (np.linalg.norm(image_prev) + 1e-12)

        if verbose:
            print(f"  Iter {iteration:3d}/{n_iterations}  |  "
                  f"Obj = {obj:.4e}  |  Rel. change = {rel_change:.4e}")

        if rel_change < convergence_tol:
            if verbose:
                print(f"  Converged at iteration {iteration}.")
            break

    return image, obj_vals



def div_nzer(x, y):
    return np.divide(x, y, out=np.zeros_like(y), where=y!=0)
def Mask(Cnt,rad=29.):
    msk = mmrimg.get_cylinder(Cnt, rad=rad, xo=0, yo=0, unival=1, gpu_dim=False) <= 0.9
    return msk
def Claude_OSEM_Test(
    measured_sino,
    datain,
    muMaps,
    scanner_params,
    hst,
    n_iterations,
    n_subsets,
    randoms=[],
    scatter=[],
    init_image=None,
    verbose=True,
):
    Cnt = scanner_params['Cnt']
    img_shape = (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX'])

    if init_image is not None:
        image = np.array(init_image, dtype=np.float32)
    else:
        image = np.ones(img_shape, dtype=np.float32)

    measured_sino = np.array(measured_sino, dtype=np.float32)

    # Norm sinogram
    ncmp, _ = mmrnorm.get_components(datain, Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst,
                                 normcomp=ncmp, gpu_dim=False)
    nsng = np.array(nsng, dtype=np.float32)

    # Attenuation correction factors
    muh, muo = muMaps
    mus = mmrimg.convert2dev(muo + muh, Cnt)
    _, fwd_mu_gaps = Forward(mus, muMaps, scanner_params, hst,True)
    fwd_mu_gaps = np.array(fwd_mu_gaps, dtype=np.float32)
    acf_sino = np.exp(-fwd_mu_gaps)

    correction = acf_sino * nsng

    # Additive terms
    additive = None
    if len(scatter) != 0 and len(randoms) != 0:
        additive = np.array(scatter, dtype=np.float32) + np.array(randoms, dtype=np.float32)

    # Mask
    msk = mmrimg.get_cylinder(Cnt, rad=29., xo=0, yo=0, unival=1, gpu_dim=False) > 0.9

    # Subset indices along angular axis (axis 1)
    n_angles = measured_sino.shape[1]
    subset_indices = [np.arange(s, n_angles, n_subsets) for s in range(n_subsets)]

    # Global sensitivity — same approach as the working original
    if verbose:
        print("Computing global sensitivity image ...")
    sens = Back(cu.asarray(correction, dtype=np.float32), muMaps, scanner_params, hst)
    sens = np.array(sens, dtype=np.float32)
    sens = np.maximum(sens, 1e-10)
    subset_sens = []
    # Pre-compute sensitivity once (exactly as in your working MLEM)
    sens = np.array(Back(cu.asarray(correction, dtype=np.float32), 
                        muMaps, scanner_params, hst), dtype=np.float32)
    sens = np.maximum(sens, 1e-10)

    for it in range(n_iterations):
        for s_idx, idx in enumerate(subset_indices):
            
            # Forward project full image
            _, fwd_gaps = Forward(image, muMaps, scanner_params, hst, False)
            fwd_gaps = np.array(fwd_gaps, dtype=np.float32)

            # Apply correction to forward projection
            fwd_corrected = fwd_gaps * correction

            # Add additive terms to full sinogram
            if additive is not None:
                fwd_corrected = fwd_corrected + additive

            fwd_corrected = np.maximum(fwd_corrected, 1e-10)

            # Build full ratio sinogram — 1.0 at non-subset angles (neutral for Back)
            ratio_sino = np.ones_like(correction)
            ratio_sino[:, idx, :] = measured_sino[:, idx, :] / fwd_corrected[:, idx, :]

            # Backproject full ratio (no zeros, no partial sinogram)
            back_ratio = np.array(
                Back(cu.asarray(ratio_sino, dtype=np.float32), muMaps, scanner_params, hst),
                dtype=np.float32
            )

            # Subset-scale the sensitivity to match the 1.0-filled non-subset angles
            # Non-subset angles contribute Back(ones * correction) to back_ratio
            # We need to remove that contribution from the denominator
            # Compute what the non-subset angles contributed to back_ratio
            ones_non_subset = np.zeros_like(correction)
            ones_non_subset[:, idx, :] = correction[:, idx, :]  # only subtract subset contribution from sens
            
            # Actually: since non-subset ratio = 1.0, back_ratio = Back(ratio_subset_angles + 1*non_subset_angles*correction)
            # This is equivalent to: back_subset_ratio + back_non_subset_correction
            # So: back_subset_ratio = back_ratio - Back(non_subset_correction)
            non_subset_mask = np.ones_like(correction)
            non_subset_mask[:, idx, :] = 0.0
            non_subset_correction = correction * non_subset_mask

            back_non_subset = np.array(
                Back(cu.asarray(non_subset_correction, dtype=np.float32), muMaps, scanner_params, hst),
                dtype=np.float32
            )

            # Per-subset sensitivity = full sens - non-subset contribution
            sens_subset = sens - back_non_subset
            sens_subset = np.maximum(sens_subset, 1e-10)

            # Update
            image = image * (back_ratio - back_non_subset) / sens_subset
            image = np.maximum(image, 0.0)
            #image[~msk] = 0.0
            

            if verbose:
                print(
                    f"Iter {it+1}/{n_iterations} | "
                    f"Subset {s_idx+1}/{n_subsets} | "
                    f"img max={image.max():.4f}"
                )
    image[~msk] = 0.0
    return image
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure
from skimage.morphology import ball, closing, opening, dilation, erosion
def create_brain_mask(
    volume: np.ndarray,
    intensity_percentile: float = 12.0,
    closing_radius: int = 6,
    dilation_radius: int = 3,
) -> np.ndarray:
    volume = volume.astype(np.float32)
    nonzero = volume[volume > 0]

    threshold = min(filters.threshold_otsu(nonzero),
                    np.percentile(nonzero, intensity_percentile))
    binary = volume > threshold

    labeled = measure.label(binary, connectivity=2)
    binary = labeled == max(measure.regionprops(labeled), key=lambda p: p.area).label

    binary = closing(binary, ball(closing_radius))
    eroded = erosion(binary, ball(closing_radius - 1))
    labeled = measure.label(eroded, connectivity=2)
    props = measure.regionprops(labeled)
    eroded = labeled == max(props, key=lambda p: p.area).label if props else eroded

    filled = (
        np.stack([ndimage.binary_fill_holes(eroded[z])        for z in range(eroded.shape[0])])
        | np.stack([ndimage.binary_fill_holes(eroded[:, y, :]) for y in range(eroded.shape[1])], axis=1)
        | np.stack([ndimage.binary_fill_holes(eroded[:, :, x]) for x in range(eroded.shape[2])], axis=2)
    )

    mask = dilation(filled, ball(dilation_radius)) & (volume > 0)
    return mask.astype(bool)


def apply_brain_mask(volume: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = volume.astype(np.float32).copy()
    out[~mask] = fill_value
    return out
def Claude_OSEM(
    measured_sino,
    datain,
    muMaps,
    scanner_params,
    hst,
    n_iterations=4,
    n_subsets=14,
    randoms = [],
    scatter = [],
    init_image=None,
    verbose=True,):
    Cnt = scanner_params['Cnt']
    img_shape = (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX'])

    if init_image is not None:
        image = np.array(init_image, dtype=np.float32)
    else:
        image = np.ones(img_shape, dtype=np.float32)

    measured_sino = np.array(measured_sino, dtype=np.float32)  # (837, 252, 344)

    ncmp, _ = mmrnorm.get_components(datain, Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst,
                                 normcomp=ncmp, gpu_dim=False)
    nsng = np.array(nsng, dtype=np.float32)

    muh, muo = muMaps
    mus = mmrimg.convert2dev(muo + muh, Cnt)
    _, acf_rsng = Forward(mus, muMaps, scanner_params, hst,True)
    #acf_sino = np.exp(-np.array(acf_rsng, dtype=np.float32))

    correction = acf_rsng * nsng  # (837, 252, 344)

    n_views = measured_sino.shape[1]   # 252
    all_indices = np.arange(n_views)
    subset_indices = [all_indices[s::n_subsets] for s in range(n_subsets)]

    if verbose:
        print("Pre-computing sensitivity image …")

    sens = Back(correction, muMaps, scanner_params, hst)
    #sens = np.array(sens, dtype=np.float32)
    sens = np.maximum(sens, 1e-10)
    subset_sens = []
    n_angles = measured_sino.shape[1]  # 252
    subset_indices = [np.arange(s, n_angles, n_subsets) for s in range(n_subsets)]
    msk = Mask(Cnt)
    # for s_idx, idx in enumerate(subset_indices):
    #     sens_sino = np.zeros_like(correction)
    #     sens_sino[:, idx, :] = correction[:, idx, :]   # axis 1, matches forward step
    #     s = Back(sens_sino, muMaps, scanner_params, hst)
    #     s = np.array(s, dtype=np.float32)
    #     s[~msk] = 0.0
    #     s = np.maximum(s, 1e-10)
    #     subset_sens.append(s)
    
    additive = []
    if len(scatter)!=0 and len(randoms)!=0:
        additive = scatter+randoms

    for it in range(n_iterations):
        for s_idx, idx in enumerate(subset_indices):

            # Forward project + apply corrections
            _, rsng_full = Forward(image, muMaps, scanner_params, hst,False)
            rsng_full = np.array(rsng_full, dtype=np.float32)  # (837, 252, 344)

            # Subset along axis 1
            if len(additive)!=0:
                
                fwd_subset  = (rsng_full[:, idx, :]+ additive[:, idx, :]) #* correction[:, idx, :]
            else:
                 fwd_subset  = (rsng_full[:, idx, :]) #* correction[:, idx, :]
            if len(additive)!=0:
                fwd_subset = fwd_subset 

            meas_subset = measured_sino[:, idx, :]

            # Ratio
            fwd_subset  = np.maximum(fwd_subset, 1e-10)
            ratio_sino  = (meas_subset / fwd_subset) #* correction[:, idx, :]
            if len(additive)!=0:
                ratio_sino = ratio_sino# + additive[:, idx, :]
            # Back-project ratio — insert back into full sinogram at subset views
            ratio_full = np.zeros_like(measured_sino)
            ratio_full[:, idx, :] = ratio_sino

            back_ratio = Back(ratio_full, muMaps, scanner_params, hst)
            back_ratio = np.array(back_ratio, dtype=np.float32)

            # Scale sensitivity by subset fraction to keep update magnitude correct
            subset_fraction = len(idx) / n_views
            sens_scaled = sens * subset_fraction

            # Update
            image = image * (back_ratio / np.maximum(sens_scaled, 1e-10))
            image = np.maximum(image, 0.0)
            #image[~msk] = 0.0 
            if verbose:
                print(
                    f"Iter {it+1}/{n_iterations} | "
                    f"Subset {s_idx+1}/{n_subsets} | "
                    f"views {len(idx)}/{n_views} | "
                    f"img max={image.max():.4f}"
                )

    return image

def OSEM(sinog, datain, muMaps, scanner_params, hst, iterations=1, num_subsets=8):
    
    # Derive shape from actual data - never hardcode
    sinogramShape = sinog.shape  # (837, 252, 344)
    print("Using sinogram shape:", sinogramShape)
    
    Cnt = scanner_params['Cnt']
    targetSubset = 1
    subset_numbers = sinog.shape[0]//num_subsets
    start = subset_numbers*targetSubset
    end = subset_numbers*(targetSubset+1)
    pred = cu.ones((
        Cnt['SO_IMZ'],
        Cnt['SO_IMY'],
        Cnt['SO_IMX']
    ), dtype=np.float32)#[start:end,:,:]

    muh, muo = muMaps
    muhs = mmrimg.convert2dev(muh, Cnt)
    mus  = mmrimg.convert2dev(muo, Cnt)

    fMus = Forward(mus, muMaps, scanner_params, hst)
    acf_sino = fMus[1]  # will also be (837, 252, 344)

    ncmp, _ = mmrnorm.get_components(datain, Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst, normcomp=ncmp, gpu_dim=False)

    # Sensitivity: Back(ones) using CORRECT shape
    # sensitivity = Back(
    #     cu.asarray(np.ones(sinogramShape, dtype=np.float32)),
    #     muMaps, scanner_params, hst
    # )
    sensitivity = Back(
        cu.asarray(nsng, dtype=np.float32),
        muMaps, scanner_params, hst
    )
    sensitivity = np.maximum(sensitivity, 1e-8)

    print("sensitivity min/max:", float(sensitivity.min()), float(sensitivity.max()))
    subset_numbers = sinog.shape[0]//num_subsets
    Subsetted_Sinog = []
    for i in range(num_subsets):
        start = i*subset_numbers
        end = (i+1)*subset_numbers
        Subsetted_Sinog.append(sinog[start:end,:,:])
    start = num_subsets*subset_numbers
    end = (num_subsets+1)*subset_numbers
    if start != sinog.shape[0]:
        Subsetted_Sinog.append(sinog[start:end,:,:])
    print(start,end)
    targetSubset = 1
    start = subset_numbers*targetSubset
    end = subset_numbers*(targetSubset+1)

    for i in range(iterations):
        #for j in range(Subsetted_Sinog):
        #using only one subset rn

        fwd = Forward(pred, muMaps, scanner_params, hst)[1]
        for j in range(0,num_subsets-1):
            targetSubset = j
            start = subset_numbers*targetSubset
            end = subset_numbers*(targetSubset+1)
            
            expected = acf_sino * fwd *nsng
            expected = np.maximum(expected, 1e-10)
            ratio = sinog
            ratio[start:end,:,:] = (sinog[start:end,:,:] / (expected[start:end,:,:] + 1e-8)) * acf_sino[start:end,:,:] * nsng[start:end,:,:]
            
            correction = Back(
                cu.asarray(ratio, dtype=np.float32),
                muMaps, scanner_params, hst
            )

            pred *= (correction / sensitivity)
            pred  = np.maximum(pred, 0.0)

        print(f"Iter {i+1} | pred min/max/mean: "
            f"{float(pred.min()):.4f} / {float(pred.max()):.4f} / {float(pred.mean()):.4f}")

    return pred
def get_patches(arr, r):
    M = arr.shape[0]
    shape = (M - r + 1, M - r + 1, r, r)
    strides = arr.strides + arr.strides
    patches = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    patches = patches.reshape(-1, r, r)
    return patches
def NoiseEstim(image,patch_size):
    import math
    import statistics
    import numpy as np
    from numpy import linalg as LA
    X = get_patches(image,patch_size)
    s = (image.shape[0]-patch_size+1)*(image.shape[1]-patch_size+1)
    X = X.reshape(len(X), -1)  # shape: (num_patches, r*r)
    mu = np.mean(X, axis=0)    # shape: (r*r,)  
    
    
    # Covariance = np.zeros_like(X[0])
    # for i in range(len(X)):
    #     Covariance+= (X[i] - mu)*(X[i]-mu).T
    #     #print(Covariance)
    # Covariance *= 1/s
    diff = X - mu
    Covariance = (diff.T @ diff) / len(X)
    #print(Covariance)
    Eigenvalues = []
    #for i in Covariance:
    r = patch_size**2#len(Eigenvalues)
    Eigenvalues, eigenvectors = LA.eig(np.array(Covariance))
        #Eigenvalues.append(eigenvalues)
    sigma = -1
    #print(Eigenvalues)
    
    for i in range(1,r):
        EigenSum = sum(Eigenvalues[i:r])
        T = (1/(r-i+1)*EigenSum)
        #print(len(Eigenvalues[i:]),EigenSum,statistics.median(EigenSum),T)
        if T <= np.median(np.sort(Eigenvalues[i:r])):
            sigma = math.sqrt(abs(T))
            break
    #print(sigma,i)
    return sigma
"""
import matplotlib.pyplot as plt
NoiseLevels1 = []
NoiseLevels2 = []
NoiseLevels3 = []
NoiseLevels4 = []
for i in range(0,len(recon)):
    NoiseLevels1.append(alanrec.NoiseEstim(recon[i],8))
    NoiseLevels2.append(alanrec.NoiseEstim(recon2[i],8))
    NoiseLevels3.append(alanrec.NoiseEstim(recon3[i],8))
    NoiseLevels4.append(alanrec.NoiseEstim(recon4[i],8))
#NoiseEstimate = alanrec.NoiseEstim(recon[63],8)
plt.plot(NoiseLevels1)
plt.plot(NoiseLevels2)
plt.plot(NoiseLevels3)
plt.plot(NoiseLevels4)
plt.legend(['3,3','3,3+scat+rands','3,6+scat+rands','6,3+scat+rands'])
plt.show()
"""

def scale_scatter_tails(prompt, random, scatter_sss, norm, attn_sino, tail_mask):
    """
    Scale SSS to approximate multi-scatter via tail fitting.
    
    prompt     : measured prompt sinogram
    random     : randoms estimate
    scatter_sss: SSS output from NiftyPET
    norm       : normalisation sinogram
    attn_sino  : attenuation correction factors (ACF)
    tail_mask  : boolean mask of tail (scatter-only) regions
    """
    # Corrected prompts in tails (subtract randoms, apply norm+attn)
    prompts_corrected = (prompt - random) / (norm * attn_sino + 1e-9)
    
    # Ratio in tail regions only
    numerator   = prompts_corrected[tail_mask].sum()
    denominator = scatter_sss[tail_mask].sum()
    
    scale_factor = numerator / (denominator + 1e-9)
    
    return scatter_sss * scale_factor




def iterative_scatter_scaling(prompt, random, norm, acf, mu_h, mu_o,
                               datain, mMRpars, n_iter=3):
    
    # Initial reconstruction (e.g. OSEM, no scatter correction)
    em_recon = nipet.mmrchain(datain, mMRpars, ..., scatter=False)['im']
    
    for i in range(n_iter):
        # Forward project current estimate to sinogram space
        em_sino = nipet.prj.fprj(em_recon, ...)
        
        # Recompute SSS with updated emission image
        sss = nipet.sct.vsm(datain, mumaps=(mu_h, mu_o),
                             em=em_sino, mMRpars=mMRpars)
        scatter_sss = sss['sino']
        
        # Scale to tails
        tail_mask = make_tail_mask(acf)
        scatter_scaled = scale_scatter_tails(
            prompt, random, scatter_sss, norm, acf, tail_mask
        )
        
        # Reconstruct with scatter subtracted
        em_recon = nipet.mmrchain(
            datain, mMRpars,
            ...,
            scatter=scatter_scaled
        )['im']
        
        print(f"Iter {i+1}: scatter scale = {scatter_scaled.sum()/scatter_sss.sum():.4f}")
    
    return em_recon, scatter_scaled



def RawModeRecon(sinog,datain,muMaps,scanner_params,hst,iterations=5):
    # Derive shape from actual data - never hardcode
    sinogramShape = sinog.shape  # (837, 252, 344)
    print("Using sinogram shape:", sinogramShape)
    
    Cnt = scanner_params['Cnt']
    
    pred = cu.ones((
        Cnt['SO_IMZ'],
        Cnt['SO_IMY'],
        Cnt['SO_IMX']
    ), dtype=np.float32)

    muh, muo = muMaps
    muhs = mmrimg.convert2dev(muh, Cnt)
    mus  = mmrimg.convert2dev(muo, Cnt)

    fMus = Forward(mus, muMaps, scanner_params, hst)
    acf_sino = fMus[1]  # will also be (837, 252, 344)

    ncmp, _ = mmrnorm.get_components(datain, Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst, normcomp=ncmp, gpu_dim=False)

    # Sensitivity: Back(ones) using CORRECT shape
    # sensitivity = Back(
    #     cu.asarray(np.ones(sinogramShape, dtype=np.float32)),
    #     muMaps, scanner_params, hst
    # )# imports & helper functions
