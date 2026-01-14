
"""Image reconstruction from raw PET data"""
import logging
import os
import time
from collections import namedtuple
from collections.abc import Iterable
from numbers import Real

import cuvec as cu
import numpy as np
import scipy.ndimage as ndi
from tqdm.auto import trange

from niftypet import nimpa

# resources contain isotope info
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/NiftyPETThings/NIPET/niftypet/nipet/prj')

from .. import mmraux, mmrnorm, resources
from .. img import mmrimg
from .. lm.mmrhist import randoms
from .. sct import vsm
from . import petprj


"""
petprj.fprj()
petprj.bprj()
petprj.osem()
"""
def reconstructionScript(
        datain, mumaps, hst, scanner_params, recmod=3, itr=4, fwhm=0., psf=None,
        mask_radius=29., decay_ref_time=None, attnsino=None, sctsino=None, randsino=None,
        normcomp=None, emmskS=False, frmno='', fcomment='', outpath=None, fout=None,
        store_img=False, store_itr=None, ret_sinos=False
    ):
    #estimate scatter
    #estimate randoms


    #forward project from listmode
    muh, muo = mumaps
    # get the GPU version of the image dims
    mus = mmrimg.convert2dev(muo + muh, Cnt)

    psng = mmraux.remgaps(hst['psino'], txLUT, Cnt)
    asng = cu.zeros(psng.shape, dtype=np.float32)
    print(asng)
    petprj.fprj(asng, cu.asarray(mus), txLUT, axLUT, np.array([-1], dtype=np.int32), Cnt, 1)
    print(asng)

    #backproject from forward project
    return 0


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
mMRpars = nipet.get_mmrparams('/store/mmr_hardwareumaps/')
folderin = "/store/amyloid_brain"
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

# hardware mu-map (bed, head/neck coils)
mu_h = nipet.hdw_mumap(datain, [1,2,4], mMRpars, outpath=opth, use_stored=True)

mu_o = nipet.obj_mumap(datain, mMRpars, outpath=opth, store=True)

# create histogram
mMRpars['Cnt']['BTP'] = 0
m = nipet.mmrhist(datain, mMRpars, outpath=opth, store=True, use_stored=True)
hst = mmrhist(datain, mMRpars, t0=t0, t1=t1)

reconstructionScript(
        datain=datain, mumaps=mumaps, hst=hst, scanner_params=mMRpars, recmod=3, itr=4, fwhm=0., psf=None,
        mask_radius=29., decay_ref_time=None, attnsino=None, sctsino=None, randsino=None,
        normcomp=None, emmskS=False, frmno='', fcomment='', outpath=None, fout=None,
        store_img=False, store_itr=None, ret_sinos=False
    )