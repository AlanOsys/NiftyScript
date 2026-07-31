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
fwd = alanrec.Forward(im,(mu_h['im'],mu_o['im']),mMRpars,m)[1]
rands = alanrec.Randoms(mMRpars,hst)[0]
eim = alanrec.Scatter_EMML(m['psino'],datain,(mu_h['im'],mu_o['im']),mMRpars,m,1)
#imscroll(eim)
#scatter = alanrec.Scatter(datain, muMaps, mMRpars,hst,rands,eim)
#recon = alanrec.Scatter_EMML(m['psino'],datain,(mu_h['im'],mu_o['im']),mMRpars,m,10)
#recon2 = alanrec.Claude_OSEM(m['psino'],datain,(mu_h['im'],mu_o['im']),mMRpars,m,4,6)
recon3 = np.asarray(alanrec.Scatter_BSREM_Updated(m['psino'],datain,(mu_h['im'],mu_o['im']),mMRpars,m,5,14,rands,beta=10))
imscroll((recon3))
#scatter = alanrec.Scatter(datain, muMaps, mMRpars,hst,rands,eim)
#recon5 = alanrec.Scatter_EMML(m['psino'],datain,(mu_h['im'],mu_o['im']),mMRpars,m,10,rands,scatter)
recon2 = nipet.mmrchain(datain, mMRpars,itr=5,histo=m,mu_h=mu_h,mu_o=mu_o,psf=2.5,recmod=1,outpath=None,fcomment='',store_img=True)
#imscroll(recon2['im'])
diff = recon3-recon2['im']
#imscroll(diff,cmap="magma")
#mask = alanrec.create_brain_mask(recon3,15.0,10.0)
#brain = alanrec.apply_brain_mask(recon3, mask)
#imscroll(brain)
#msk = mmrimg.get_cylinder(Cnt, rad=11.5, xo=0, yo=1, unival=1, gpu_dim=False) > 0.9
#recon3[~msk] = 0.0
#recon2['im'][~msk] = 0.0
#diff[~msk] = 0.0
from niftypet.nipet import resources
ncmp,rest,res = alanrec.getNormParameters(datain,Cnt)
tref = hst['t0']
lmbd = np.log(2) / resources.riLUT[Cnt['ISOTOPE']]['thalf']
dcycrr = np.exp(lmbd * tref) * lmbd * hst['dur'] / (1 - np.exp(-lmbd * hst['dur']))
# apply quantitative correction to the image
qf = ncmp['qf'] / resources.riLUT[Cnt['ISOTOPE']]['BF'] / float(hst['dur'])
qf_loc = ncmp['qf_loc']
recon3Adjusted = recon3*qf*dcycrr*qf_loc #recon2['im']
reconAgainst = recon2['im']#recon3*qf*dcycrr*qf_loc
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=3)

slide = 50
im1 = axes[0].imshow(reconAgainst[slide], cmap='inferno')
im2 = axes[1].imshow(recon3Adjusted[slide], cmap='inferno')
im3 = axes[2].imshow((recon3Adjusted-reconAgainst)[slide], cmap='bwr')


fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
fig.colorbar(im3, ax=axes[2])

np.sqrt(np.mean((recon3Adjusted - reconAgainst)) ** 2)#




# def normalize_array(arr):
#     norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
#     return norm_arr
# noise1 = []
# noise2 = []
# noise3 = []
# recon1 = np.array(alanrec.Scatter_EMML(m['psino'],datain,(mu_h['im'],mu_o['im']),mMRpars,m,42,rands))
# for i in recon1:
#     noise1.append(alanrec.NoiseEstim(i,6))
# noise1 = normalize_array(noise1)

# for i in recon2['im']:
#     noise2.append(alanrec.NoiseEstim(i,6))
    # noise2 = normalize_array(noise2)
    # for i in recon3:
#     noise3.append(alanrec.NoiseEstim(i,6))
# noise3 = normalize_array(noise3)
# plt.plot(noise1)
# plt.plot(noise2)
# plt.plot(noise3)
# plt.legend(['EMML','True OSEM','My OSEM'])
# add space for colour bar



import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 2.5),
                         gridspec_kw={'wspace': 0.05})
slide = 45

# Get image dimensions and calculate crop bounds
h, w = reconAgainst[slide].shape
cy, cx = h // 2, w // 2
half = 70  # 30x30 crop → 15 pixels each side

# Crop each image
crop_against = reconAgainst[slide][cy-half:cy+half, cx-half:cx+half]
crop_adjusted = recon3Adjusted[slide][cy-half:cy+half, cx-half:cx+half]
crop_diff = (recon3Adjusted - reconAgainst)[slide][cy-half:cy+half, cx-half:cx+half]

im1 = axes[0].imshow(crop_against, cmap='hot')
im2 = axes[1].imshow(crop_adjusted, cmap='hot')
im3 = axes[2].imshow(crop_diff, cmap='bwr')

fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
fig.colorbar(im3, ax=axes[2])

axes[0].set_title('OSEM')
axes[1].set_title('BSREM')
axes[2].set_title('Difference')

plt.tight_layout()
plt.show()
