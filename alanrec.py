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
import cuvec as cu
import numpy as np
import scipy.ndimage as ndi
from tqdm.auto import trange

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
#tmpsens = cu.ones((837,344,344),dtype=np.float32)
#bksens = Back(tmpsens,)
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
    mus  = mmrimg.convert2dev(muo, Cnt)

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
        fwd = Forward(pred, muMaps, scanner_params, hst)[1]
        
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

def apply_psf(sino, psf_kernel):
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
    result = sino.copy().astype(np.float32)
    for axis, kernel in enumerate(psf_kernel):       # iterate over 3 axes
        kernel = kernel / kernel.sum()               # normalise — critical
        result = convolve1d(result, kernel, axis=axis, mode='reflect')
    return result

    
def Scatter_EMML(sinog,datain,muMaps,scanner_params,hst,
    iterations=4,randoms=[],scatter=[],verbose=True):
    
    # Derive shape from actual data - never hardcode
    sinogramShape = sinog.shape  # (837, 252, 344)
    print("Using sinogram shape:", sinogramShape)
    
    Cnt = scanner_params['Cnt']
    
    pred = cu.ones((
        Cnt['SO_IMZ'],
        Cnt['SO_IMY'],
        Cnt['SO_IMX']
    ), dtype=np.float32)
    if len(randoms) == 0:
        randoms = cu.zeros(sinog.shape)
    if len(scatter) == 0:
        scatter = cu.zeros(sinog.shape)
    muh, muo = muMaps
    #muhs = mmrimg.convert2dev(muh, Cnt)
    #mus  = mmrimg.convert2dev(muo, Cnt)

    #fMus = Forward(mus, muMaps, scanner_params, hst)
    #acf_sino = fMus[1]  # will also be (837, 252, 344)

    ncmp, _ = mmrnorm.get_components(datain, Cnt)
    nsng = mmrnorm.get_norm_sino(datain, scanner_params, hst)#, normcomp=ncmp, gpu_dim=False)
    # Attenuation correction factors
    muh, muo = muMaps
    mus = mmrimg.convert2dev(muo + muh, Cnt)
    _, fwd_mu_gaps = Forward(mus, muMaps, scanner_params, hst,True)
    fwd_mu_gaps = np.array(fwd_mu_gaps, dtype=np.float32)
    #acf_sino = np.exp(-fwd_mu_gaps)
    #correction = acf_sino * nsng
    # Sensitivity: Back(ones) using CORRECT shape
    # sensitivity = Back(
    #     cu.asarray(np.ones(sinogramShape, dtype=np.float32)),
    #     muMaps, scanner_params, hst
    # )
    sensitivity = Back(
        cu.asarray(nsng*fwd_mu_gaps, dtype=np.float32),
        muMaps, scanner_params, hst
    )
    sensitivity = np.maximum(sensitivity, 1e-8)
    #return sensitivity
    #PSF
    fwhm = 2.5
    psf = fwhm
    #psfkernel = psf_config(psf, Cnt)
    #print(psfkernel)
    print("sensitivity min/max:", float(sensitivity.min()), float(sensitivity.max()))
    additive = np.array(randoms, dtype=np.float32)
    #if len(scatter) != 0 and len(randoms) != 0:
    additive += np.array(scatter, dtype=np.float32) #+ np.array(randoms, dtype=np.float32)
    msk = Mask(Cnt)
    for i in range(iterations):
        fwd = Forward(pred, muMaps, scanner_params, hst,False)[1]
        if additive is not None:
            fwd += additive
        expected = fwd#*acf_sino*nsng#*psfkernel
        expected = np.maximum(expected, 1e-8)
        

        ratio = (sinog/ (expected)) #* acf_sino * nsng
        correction = Back(
            cu.asarray(ratio, dtype=np.float32),
            muMaps, scanner_params, hst
        )
        
        pred = (pred/ sensitivity) * (correction )
        pred  = np.maximum(pred, 0.0)
        #pred[~msk] = 0.0
        


        print(f"Iter {i+1} | pred min/max/mean: "
              f"{float(pred.min()):.4f} / {float(pred.max()):.4f} / {float(pred.mean()):.4f}")
    pred[~msk] = 0.0
    return pred

def Mask(Cnt):
    mask_radius = 29.
    msk = mmrimg.get_cylinder(Cnt, rad=mask_radius, xo=0, yo=0, unival=1, gpu_dim=False) > 0.5
    return msk
def Claude_OSEM_Test(
    measured_sino,
    datain,
    muMaps,
    scanner_params,
    hst,
    n_iterations=4,
    n_subsets=14,
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
    _, fwd_mu_gaps = Forward(mus, muMaps, scanner_params, hst)
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
    sens = Back(cu.asarray(nsng, dtype=np.float32), muMaps, scanner_params, hst)
    sens = np.array(sens, dtype=np.float32)
    sens = np.maximum(sens, 1e-10)

    # OSEM iterations
    for it in range(n_iterations):
        for s_idx, idx in enumerate(subset_indices):

            _, fwd_gaps = Forward(image, muMaps, scanner_params, hst)
            fwd_gaps = np.array(fwd_gaps, dtype=np.float32)

            fwd_subset = fwd_gaps[:, idx, :] * correction[:, idx, :]

            if additive is not None:
                fwd_subset = fwd_subset + additive[:, idx, :]

            meas_subset = measured_sino[:, idx, :]
            fwd_subset  = np.maximum(fwd_subset, 1e-10)

            # Ratio — no double correction
            ratio_si
            # Scale global sensitivity by this subset's fraction — same as working original
            subset_fraction = len(idx) / n_angles
            sens_scaled = sens * subset_fraction

            image = image * (back_ratio / np.maximum(sens_scaled, 1e-10))
            image = np.maximum(image, 0.0)
            image[~msk] = 0.0

            if verbose:
                print(
                    f"Iter {it+1}/{n_iterations} | "
                    f"Subset {s_idx+1}/{n_subsets} | "
                    f"img max={image.max():.4f}"
                )

    return image
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
    _, acf_rsng = Forward(mus, muMaps, scanner_params, hst)
    acf_sino = np.exp(-np.array(acf_rsng, dtype=np.float32))

    correction = acf_sino * nsng  # (837, 252, 344)

    n_views = measured_sino.shape[1]   # 252
    all_indices = np.arange(n_views)
    subset_indices = [all_indices[s::n_subsets] for s in range(n_subsets)]

    if verbose:
        print("Pre-computing sensitivity image …")

    sens = Back(nsng, muMaps, scanner_params, hst)
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
            _, rsng_full = Forward(image, muMaps, scanner_params, hst)
            rsng_full = np.array(rsng_full, dtype=np.float32)  # (837, 252, 344)

            # Subset along axis 1
            if len(additive)!=0:
                
                fwd_subset  = (rsng_full[:, idx, :]+ additive[:, idx, :]) * correction[:, idx, :]
            else:
                 fwd_subset  = (rsng_full[:, idx, :]) * correction[:, idx, :]
            if len(additive)!=0:
                fwd_subset = fwd_subset 

            meas_subset = measured_sino[:, idx, :]

            # Ratio
            fwd_subset  = np.maximum(fwd_subset, 1e-10)
            ratio_sino  = (meas_subset / fwd_subset) * correction[:, idx, :]
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
