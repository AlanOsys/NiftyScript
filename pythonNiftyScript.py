
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
    Cnt = scanner_params['Cnt']
    mus = convert2dev(muo + muh, Cnt)

    psng = remgaps(hst['psino'], txLUT, Cnt)
    asng = cu.zeros(psng.shape, dtype=np.float32)
    log.info(asng)
    petprj.fprj(asng, cu.asarray(mus), txLUT, axLUT, np.array([-1], dtype=np.int32), Cnt, 1)
    log.info(asng)

    #backproject from forward project
    return 0

"""
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
hst = m

#try:  # needs HW mu-maps
#    imscroll(mu_o['im'] + mu_h['im'], cmap='bone')  # title=r"$\mu$-map"
#except:
#    imscroll(mu_o['im'], cmap='bone')





import plotext as plt
import numpy
#recon['im']
#m['dsino']
InitList = [x.tolist()[0] if type(x)==numpy.ndarray else x for x in mu_o['im']]

plt.matrix_plot(InitList)
plt.show()


print(InitList)






# sinogram index (<127 for direct sinograms, >=127 for oblique sinograms)
#imscroll([m['psino'], m['dsino']],
#         titles=["Prompt sinogram (%.3gM)" % (m['psino'].sum() / 1e6),
#                 "Delayed sinogram (%.3gM)" % (m['dsino'].sum() / 1e6)],
#         cmap='inferno',
         #colorbars=[1]*2,
#         fig=plt.figure(figsize=(9.5, 3.5), tight_layout=True, frameon=False));
#axs = plt.gcf().axes
#axs[-1].set_xlabel("bins")
#[i.set_ylabel("angles") for i in axs]

# built-in default: 14 subsets
fcomment = f"_fwhm-{fwhm}_recon"
outpath = path.join(opth, folderout)
recon = glob(
    f"{outpath}/PET/single-frame/a_t-*-*sec_itr-{itr}{fcomment}.nii.gz"
)


nipet.reconstructionScript(
        datain=datain, mumaps=[mu_h['im'],mu_o['im']], hst=hst, scanner_params=mMRpars, recmod=3, itr=4, fwhm=0., psf=None,
        mask_radius=29., decay_ref_time=None, attnsino=None, sctsino=None, randsino=None,
        normcomp=None, emmskS=False, frmno='', fcomment='', outpath=None, fout=None,
        store_img=False, store_itr=None, ret_sinos=False
    )