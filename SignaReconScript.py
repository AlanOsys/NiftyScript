import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
import h5py
import cuvec as cu
from niftypet import nipet, nimpa
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Scanner setup
# ==============================================================================

mfldr = Path('/home/di0/Downloads/Signa/recon/Work')

pthlm = mfldr/'rdf_f1b1.rdf'
Cnt, txLUT, axLUT = nipet.sigaux.init_sig(pthlm)
scanner_params = dict(Cnt=Cnt, txLUT=txLUT, axLUT=axLUT)
Cnt['SPN'] = 2
nsinos = Cnt['NSN'] if Cnt['SPN'] == 2 else Cnt['NSN1']
im_shape = (Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ'])
sino_shape = (Cnt['NAW'], nsinos)

# ==============================================================================
# 2. Load PIFA (mu-map), fixing axis order (Z,Y,X) -> (Y,X,Z)
# ==============================================================================

fpifa = mfldr/'pifa_f1b1.pifa'
with h5py.File(fpifa, 'r') as f:
    SP_VXY = f['HeaderData/ctacDfov'][0] / f['HeaderData/xMatrix'][0] / 10
    pifa = np.array(f['PifaData'])   # (Z, Y_lowres, X_lowres)

scale = SP_VXY / Cnt['SZ_VOXY']
mu = zoom(pifa, (1.0, scale, scale), order=1)   # still (Z, Y, X)
mu = 10 * mu[::-1, ...]

# fix axis order to match im_shape = (SZ_IMY, SZ_IMX, SZ_IMZ)
mu_correct = np.transpose(mu, (1, 2, 0))
print('mu_correct shape:', mu_correct.shape, ' expected im_shape:', im_shape)

# ==============================================================================
# 3. Forward / back projectors (Signa)
# ==============================================================================

def Forward(image, scanner_params, attenuation=True):
    Cnt, txLUT, axLUT = scanner_params['Cnt'], scanner_params['txLUT'], scanner_params['axLUT']
    nsinos = Cnt['NSN'] if Cnt['SPN'] == 2 else Cnt['NSN1']
    sinogramShape = (Cnt['NAW'], nsinos)
    ims = cu.asarray(np.array(image), dtype=np.float32)
    isub = np.array([-1], dtype=np.int32)
    asng = cu.zeros(sinogramShape, dtype=np.float32)
    nipet.prjsig.fprj(asng, ims, txLUT, axLUT, isub, Cnt, int(attenuation), sync=True)
    return asng

def Back(sino, scanner_params):
    Cnt, txLUT, axLUT = scanner_params['Cnt'], scanner_params['txLUT'], scanner_params['axLUT']
    isub = np.array([-1], dtype=np.int32)
    im_shape = (Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ'])
    img = cu.zeros(im_shape, dtype=np.float32)
    sino_cu = cu.asarray(np.asarray(sino, dtype=np.float32))
    nipet.prjsig.bprj(img, sino_cu, txLUT, axLUT, isub, Cnt, sync=True)
    return img

# ==============================================================================
# 4. Forward project the mu-map
# ==============================================================================

sino = Forward(mu_correct, scanner_params, attenuation=False)
sino_np = np.asarray(sino)

# reshape (NAW, nsinos) -> (NSBINS, NSANGLES, nsinos), matching get_txLUT's
# iw*NSANGLES + ia indexing (bin outer, angle inner)
sino_3d = np.reshape(sino_np, (Cnt['NSBINS'], Cnt['NSANGLES'], nsinos))

mid_plane = nsinos // 2
plt.matshow(sino_3d[:, :, mid_plane].T)   # .T -> (angle, bin) for display
plt.colorbar()
plt.title(f'Mu-map sinogram, plane {mid_plane} (angle x bin)')
plt.show()

# ==============================================================================
# 5. Raw (uncorrected) backprojection
# ==============================================================================

bim = Back(sino, scanner_params)
bim_np = np.asarray(bim)

nimpa.imscroll(bim_np, view='s')
plt.matshow(bim_np[..., bim_np.shape[-1] // 2])
plt.colorbar()
plt.title('Raw backprojection, mid axial slice')
plt.show()

# ==============================================================================
# 6. Sensitivity image: backproject a sinogram of all ones
# ==============================================================================

ones_sino = cu.ones(sino_shape, dtype=np.float32)
sens_img = Back(ones_sino, scanner_params)
sens_img_np = np.asarray(sens_img)

print('sens_img stats: min', sens_img_np.min(), 'max', sens_img_np.max(),
      'zero frac', np.mean(sens_img_np == 0))

nimpa.imscroll(sens_img_np, view='s')
plt.matshow(np.sum(sens_img_np, axis=2))
plt.colorbar()
plt.title('Sensitivity image (summed over z)')
plt.show()

# ==============================================================================
# 7. Sensitivity-corrected backprojection
# ==============================================================================

bim_corrected = np.zeros_like(bim_np)
nz = sens_img_np != 0
bim_corrected[nz] = bim_np[nz] / sens_img_np[nz]

nimpa.imscroll(bim_corrected, view='s')
plt.matshow(bim_corrected[..., bim_corrected.shape[-1] // 2])
plt.colorbar()
plt.title('Sensitivity-corrected backprojection, mid axial slice')
plt.show()
