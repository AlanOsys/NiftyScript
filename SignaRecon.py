import numpy as np
from pathlib import Path
import h5py
import cuvec as cu
from niftypet import nipet, nimpa
from niftypet.nipet.prj import alanrec

mfldr = Path('/home/di0/Downloads/Signa/recon/Work')

# --- scanner setup ---
pthlm = mfldr/'rdf_f1b1.rdf'
Cnt, txLUT, axLUT = nipet.sigaux.init_sig(pthlm)
scanner_params = dict(Cnt=Cnt, txLUT=txLUT, axLUT=axLUT)
Cnt['SPN'] = 2
nsinos = Cnt['NSN'] if Cnt['SPN'] == 2 else Cnt['NSN1']

# --- helper: load a .sav sinogram in the same layout as sinog/nrm/etc expect ---
def load_sav(fname):
    dat = np.fromfile(mfldr/fname, dtype=np.float32)
    arr = np.reshape(dat[6:], (224, 1981, 357))
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.reshape(arr, (Cnt['NAW'], -1))
    return arr

prm  = load_sav('prompts_f1b1.sav')
nrm  = load_sav('norm.sav')
dtpu = load_sav('dtPuc_f1b1.sav')
rnd  = load_sav('randoms_f1b1.sav')
sct  = load_sav('scatter_f1b1.sav')

nrmcmp = {'nrm': nrm, 'dtpu': dtpu}

# --- mu-map (PIFA) ---
# NOTE: only one PIFA file is available in this flat-folder layout, i.e. the
# object mu-map. No separate hardware mu-map (bed/coils) was provided here,
# so muh is set to zeros. If you have a hardware map, load and pass it
# instead -- otherwise attenuation correction will be object-only.
fpifa = mfldr/'pifa_f1b1.pifa'
with h5py.File(fpifa, 'r') as f:
    SP_VXY = f['HeaderData/ctacDfov'][0] / f['HeaderData/xMatrix'][0] / 10
    pifa = np.array(f['PifaData'])

from scipy.ndimage import zoom
scale = SP_VXY / Cnt['SZ_VOXY']
muo = zoom(pifa, (1.0, scale, scale), order=1)
muo = 10 * muo[::-1, ...]   # raw (Z, Y, X) -- axis fix happens inside Signa_BSREM

muh = np.zeros_like(muo)
muMaps = (muh, muo)

# --- reconstruct ---
pred = alanrec.Signa_BSREM(
    sinog=prm,
    nrmcmp=nrmcmp,
    muMaps=muMaps,
    scanner_params=scanner_params,
    hst=None,
    iterations=20,
    n_subsets=10,
    randomsinp=rnd,
    scatterinp=[],
)

# --- view result ---
nimpa.imscroll(np.asarray(pred), view='s')
