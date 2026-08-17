# imports & helper functions
from __future__ import print_function, division
from collections import OrderedDict
from glob import glob
from os import path
import functools
import logging
import os

if os.getenv("OMP_NUM_THREADS", None) != "1":
    raise EnvironmentError("should run `export OMP_NUM_THREADS=1` before notebook launch")

from miutil.plot import apply_cmap, imscroll
from niftypet import nipet
from niftypet.nimpa import getnii
from scipy.ndimage.filters import gaussian_filter
from tqdm.auto import trange
import matplotlib.pyplot as plt
import numpy as np
import pydicom

logging.basicConfig(level=logging.INFO)
print(nipet.gpuinfo())
# get all the scanner parameters
mMRpars = nipet.get_mmrparams('/home/di0/Downloads/store/mmr_hardwareumaps-v-hdr/')
folderin = "/home/di0/Downloads/amyloidPET_FBP_TP0"
folderout = "."  # realtive to `{folderin}/niftyout`
itr = 7  # number of iterations (will be multiplied by 14 for MLEM)
fwhm = 2.5  # mm (for resolution modelling)
totCnt = None  # bootstrap sample (e.g. `300e6`) counts

# datain
folderin = path.expanduser(folderin)

# automatically categorise the input data
datain = nipet.classify_input(folderin, mMRpars, recurse=-1)

# output path
opth = path.join(datain['corepath'], "niftyout")

datain
mMRpars['Cnt']['HMUDIR'] = '/home/di0/Downloads/store/mmr_hardwareumaps-v-hdr/'
# hardware mu-map (bed, head/neck coils)
mu_h = nipet.hdw_mumap(datain, [1,2,4], mMRpars, outpath=opth, use_stored=True)

mu_o = nipet.obj_mumap(datain, mMRpars, outpath=opth, store=True)

# create histogram
mMRpars['Cnt']['BTP'] = 0
m = nipet.mmrhist(datain, mMRpars, outpath=opth, store=True, use_stored=True)
hst = m

import nibabel as nib
import os
import numpy
from miutil.plot import apply_cmap, imscroll
import cuvec as cu

Cnt = mMRpars['Cnt']
sinog = cu.zeros((127,344,344),dtype=np.float32)#cu.zeros(out_shape, dtype=np.float32)
import numpy as np


def create_circle_array(shape, center=None, radius=None):
    """Creates a 2D boolean numpy array with a filled circle."""
    h, w = shape
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    # Create coordinate grid
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # Create mask
    mask = dist_from_center <= radius
    return mask

# Example: Create a 100x100 array with a circle
circle_array = create_circle_array((344, 344), center=(172, 172), radius=50)
cylinder = np.stack((circle_array,)*127,axis=0)
print(cylinder.shape)
isub = np.array([-1], dtype=np.int32)
#sinog = cu.zeros((txLUT['Naw'], nsinos), dtype=np.float32)
from niftypet.nipet.img import mmrimg
print(Cnt['SO_IMX'],Cnt['SO_IMY'])
#ims = mmrimg.convert2dev(cylinder, Cnt)
im = cu.asarray(cylinder,dtype=np.float32)
#nip.prj.petprj.fprj(sinog, im, txLUT, axLUT, isub, Cnt,1)

A = nipet.frwd_prj(im, mMRpars, attenuation=False)

from niftypet.nipet.prj import alanrec
muMaps = (mu_h['im'],mu_o['im'])
f = alanrec.SigForward(im,(mu_h['im'],mu_o['im']),mMRpars,m)
