"""
BSREM-RDP reconstruction adapted for GE Signa PET/CT (PETpp .rdf / .sav data),
based on the mMR `Scatter_BSREM_Updated` you shared.

WHAT CHANGED VS. THE mMR VERSION, AND WHY
-------------------------------------------------------------------------------
1. Forward/Back projectors
   mMR uses `petprj.fprj/bprj` + `mmraux.putgaps/remgaps` because mMR sinograms
   have physical block gaps that need to be stripped/reinserted around the
   projector call. Signa sinograms (as read in your PETpp script) are already
   gap-free in the `(NAW, nsinos)` layout that `nipet.prjsig.fprj/bprj` expects
   directly -- so `Forward_Signa`/`Back_Signa` below call `nipet.prjsig`
   directly with no gap handling.

2. Attenuation
   mMR forward-projects `muMaps` (mu-maps) fresh through `Forward()` each time
   attenuation is needed (see `fwd_mu_gaps` in your BSREM code). Signa instead
   gives you a *precomputed* attenuation-correction-factor (ACF) sinogram
   (`acf_f1b1.sav`) straight from the scanner -- there's no mu-map to forward
   project. So instead of `nsng_fwd = nsng * fwd_mu_gaps`, I combine
   norm + dead-time + ACF into one sinogram, `asng`, exactly the way your
   basic-recon script does:

       nsng = nrm * dtpu                  # norm x dead-time
       nsng[:, 1:89:2] *= 2                # span-2 correction (from your script)
       asng = acf / nsng                  # combined norm+atten weighting

   `asng` plays the same role `nsng_fwd` plays in the mMR BSREM code: it's
   what the per-subset sensitivity images are backprojected from, and what
   randoms/scatter get divided by before being added as the additive term.

3. Attenuation flag during the iterative loop
   Because attenuation is already folded into `asng`/sensitivity/additive
   terms, the forward projection of the *current image estimate* inside the
   loop should NOT reapply attenuation -- same as mMR's
   `Forward(pred, ..., attenuation=False)`. Your basic Signa recon script
   confirms this: it calls
   `nipet.prjsig.fprj(esng_s, imr, txLUT, axLUT, ISUB_DEFAULT, Cnt, 0, sync=True)`
   with the attenuation flag set to `0`. `Forward_Signa` defaults to
   `attenuation=False` for the same reason.

4. Subsetting axis
   mMR subsets by `sinog.shape[1]` because mMR's gapped sinogram is already
   3D with angle as its own axis. Signa's `(NAW, nsinos)` layout flattens
   `NSBINS x NSANGLES` into the single `NAW` axis (confirmed by the reshape
   `(nsinos, NSBINS, NSANGLES)` used in the commented-out display code in
   your read script). So subsetting here explicitly reshapes to
   `(NSBINS, NSANGLES, nsinos)`, takes an interleaved slice of the
   NSANGLES axis, and flattens back to NAW before/after each projector call.

THINGS YOU SHOULD VERIFY BEFORE TRUSTING OUTPUT
-------------------------------------------------------------------------------
- The ACF convention: your script computes `asng = acfg/nsng` (ACF divided by
  norm, not multiplied). I've kept that convention as-is since it's what your
  working basic-recon script does, but confirm this matches how GE defines
  their ACF sinogram (some vendors store ACF as attenuation *correction*
  factors, i.e. 1/transmission, in which case you'd expect a product, not a
  ratio, in a "correction sinogram" -- your script's `/` may already be
  correct for GE's PIFA/ACF convention, just flagging it since I can't run it).
- `Cnt['NAW'] == Cnt['NSBINS'] * Cnt['NSANGLES']` -- confirm this holds for
  your `Cnt` dict; the subset reshape assumes it.
- Whether `dtpu` (dead-time) needs the same span-2 doubling as `nrm`, or only
  `nrm` does -- your script only doubles `nrm*dtpu` combined (`nsng`) as one
  array, so I kept that combined-then-doubled order.
- `Mask`, `div_nzer`, `gradient`, `_alpha` are assumed to come from the same
  module your original `Scatter_BSREM_Updated` lives in -- I haven't
  redefined them, only replaced the mask construction with the Signa
  `get_cylinder` call from your read script (see `signa_mask` below).
"""

import functools
import numpy as np
import cuvec as cu

from scipy.ndimage import gaussian_filter
from niftypet import nipet

# These are assumed to already exist in your mMR BSREM module -- imported
# here for clarity, adjust the import path to wherever they actually live.
# from your_bsrem_module import div_nzer, gradient, _alpha


# ==============================================================================
# Signa forward / back projectors  (replaces mMR's Forward()/Back())
# ==============================================================================

def Forward_Signa(image, scanner_params, attenuation=False, isub=None):
    """
    Forward projector for GE Signa data.

    Unlike mMR's Forward(), this does NOT take muMaps -- Signa attenuation is
    handled via a precomputed ACF sinogram folded into `asng` upstream (see
    module docstring), so `attenuation` just toggles the projector's internal
    flag and defaults to False, matching your basic recon script's use of `0`
    during the main iterative loop.

    Returns
    -------
    esng : CuVec, shape (Cnt['NAW'], nsinos), float32
    """
    Cnt   = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    if isub is None:
        isub = np.array([-1], dtype=np.int32)

    nsinos = Cnt['NSN'] if Cnt['SPN'] == 2 else Cnt['NSN1']
    sino_shape = (Cnt['NAW'], nsinos)

    img  = cu.asarray(np.asarray(image, dtype=np.float32))
    esng = cu.zeros(sino_shape, dtype=np.float32)

    nipet.prjsig.fprj(esng, img, txLUT, axLUT, isub, Cnt,
                       int(attenuation), sync=True)

    return esng


def Back_Signa(sino, scanner_params, isub=None):
    """
    Back projector for GE Signa data.

    Returns
    -------
    img : CuVec, shape (Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), float32
    """
    Cnt   = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    if isub is None:
        isub = np.array([-1], dtype=np.int32)

    im_shape = (Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ'])
    img  = cu.zeros(im_shape, dtype=np.float32)
    sino = cu.asarray(np.asarray(sino, dtype=np.float32))

    nipet.prjsig.bprj(img, sino, txLUT, axLUT, isub, Cnt, sync=True)

    return img


# ==============================================================================
# Signa mask (replaces mMR's Mask(Cnt))
# ==============================================================================

def signa_mask(Cnt, rad=27, xo=0, yo=0):
    """FOV mask, same call as used in your Signa read script."""
    return nipet.sigaux.get_cylinder(
        Cnt, rad=rad, xo=xo, yo=yo, unival=1, gpu_dim=True, mask=True)


# ==============================================================================
# Signa sinogram loader (mirrors the .sav reading block in your read script)
# ==============================================================================
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
def load_signa_sinograms(mfldr, Cnt, raw_shape=(224, 1981, 357)):
    """
    Load prompts / randoms / scatter / norm / dead-time / ACF sinograms for a
    Signa dataset and reshape them into the (Cnt['NAW'], nsinos)-flattened
    layout that Forward_Signa/Back_Signa expect -- same transpose/reshape
    your read script applies before use in the basic recon loop.

    Returns a dict with keys: prompts, randoms, scatter, norm, deadtime, acf
    -- each shape (Cnt['NAW'], -1).
    """
    from pathlib import Path
    mfldr = Path(mfldr)

    def _load(fname):
        dat = np.fromfile(mfldr / fname, dtype=np.float32)
        arr = np.reshape(dat[6:], raw_shape)
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.reshape(arr, (Cnt['NAW'], -1))
        return arr

    return {
        'prompts':  _load('prompts_f1b1.sav'),
        'randoms':  _load('randoms_f1b1.sav'),
        'scatter':  _load('scatter_f1b1.sav'),
        'norm':     _load('norm.sav'),
        'deadtime': _load('dtPuc_f1b1.sav'),
        'acf':      _load('acf_f1b1.sav'),
    }


# ==============================================================================
# Subset helpers for Signa's flattened NAW axis
# ==============================================================================

def _naw_to_bins_angles(sino, Cnt):
    """(NAW, nsinos) -> (NSBINS, NSANGLES, nsinos)."""
    nsinos = sino.shape[1]
    return np.reshape(np.asarray(sino), (Cnt['NSBINS'], Cnt['NSANGLES'], nsinos))


def _bins_angles_to_naw(arr, Cnt):
    """(NSBINS, NSANGLES, nsinos) -> (NAW, nsinos)."""
    nsinos = arr.shape[-1]
    return np.reshape(arr, (Cnt['NAW'], nsinos))


def signa_interleaved_subsets(n_subsets, Cnt):
    """
    Interleaved angular subsets over Cnt['NSANGLES'], mirroring the mMR
    BSREM subsetting logic but operating on the angle axis after unflattening
    NAW -> (NSBINS, NSANGLES).
    """
    n_angles = Cnt['NSANGLES']
    interleaved = np.array([
        a for s in range(n_subsets) for a in range(s, n_angles, n_subsets)
    ])
    return [np.sort(chunk) for chunk in np.array_split(interleaved, n_subsets)]


def signa_mask_subset(sino, idx, Cnt):
    """
    Zero out every angle except those in `idx`, operating on a (NAW, nsinos)
    sinogram. Used the same way mMR code does
    `masked_nsng[:, idx, :] = nsng_fwd[:, idx, :]`, just with an extra
    reshape round-trip since Signa's angle axis is flattened into NAW.
    """
    arr = _naw_to_bins_angles(sino, Cnt)
    masked = np.zeros_like(arr)
    masked[:, idx, :] = arr[:, idx, :]
    return _bins_angles_to_naw(masked, Cnt)

def _alpha(global_subit: int, alpha_0: float, alpha_decay: float) -> float:
    """
    Harmonic decay:  alpha_0 / (1 + global_subit / alpha_decay).

    Guaranteed to satisfy Σ alpha_n = ∞  and  Σ alpha_n^2 < ∞  when
    alpha_decay > 0, which is the standard sufficient condition for BSREM
    convergence (Ahn & Fessler 2003).
    """
    return alpha_0 / (1.0 + global_subit / alpha_decay)
# ==============================================================================
# BSREM-RDP for Signa
# ==============================================================================
def div_nzer(x, y):
    return np.divide(x, y, out=np.zeros_like(y), where=y!=0)
def Scatter_BSREM_Updated_Signa(
    sinog,
    signa_sinograms,          # dict from load_signa_sinograms(): norm/deadtime/acf/randoms/scatter
    scanner_params,           # dict(Cnt=Cnt, txLUT=txLUT, axLUT=axLUT)  (your SIGpars)
    # Iteration control -- same names/defaults as the mMR version
    iterations: int = 100,
    n_subsets: int = 10,
    pred=None,
    # BSREM-specific parameters -- unchanged from the mMR version
    beta: float = 0.3,
    gamma: float = 2.0,
    alpha_0: float = 1.0,
    alpha_decay: float = None,
    delta_frac: float = 1e-4,
):
    """
    BSREM-RDP PET reconstruction for GE Signa data. Structurally a drop-in
    counterpart to your mMR `Scatter_BSREM_Updated` -- same update rule,
    same RDP prior, same EM preconditioner -- with the projector, attenuation
    handling, and subsetting swapped for Signa's data layout (see module
    docstring for the full rationale of each change).

    Parameters
    ----------
    sinog : (Cnt['NAW'], nsinos) array
        Prompts sinogram (equivalent to `signa_sinograms['prompts']`; passed
        separately to mirror the mMR signature where `sinog` is explicit).
    signa_sinograms : dict
        Output of `load_signa_sinograms()` -- needs 'randoms', 'scatter',
        'norm', 'deadtime', 'acf'.
    scanner_params : dict
        Cnt / txLUT / axLUT, i.e. your `SIGpars`.

    Returns
    -------
    pred : CuVec array (Y, X, Z), float32
    """

    Cnt = scanner_params['Cnt']

    if pred is None:
        pred = cu.ones(
            (Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)

    randoms  = np.asarray(signa_sinograms['randoms'],  dtype=np.float32)
    scatter  = np.asarray(signa_sinograms['scatter'],  dtype=np.float32)
    nrm      = np.asarray(signa_sinograms['norm'],     dtype=np.float32)
    dtpu     = np.asarray(signa_sinograms['deadtime'], dtype=np.float32)
    acf      = np.asarray(signa_sinograms['acf'],      dtype=np.float32)
    sinog    = np.asarray(sinog, dtype=np.float32)

    fwhm         = 2.5
    SIGMA2FWHMmm = (8 * np.log(2))**0.5 * np.array(
        [Cnt['SO_VX' + i] for i in 'ZYX']) * 10
    psf = functools.partial(gaussian_filter, sigma=fwhm / SIGMA2FWHMmm)

    msk = signa_mask(Cnt)
    pred[msk] = 0.0

    # ── Combined norm + dead-time + attenuation sinogram ─────────────────────
    # Replaces mMR's `nsng_fwd = nsng * fwd_mu_gaps`. Signa gives attenuation
    # as a precomputed ACF sinogram rather than a mu-map to forward project,
    # so it's combined here exactly as in your basic-recon script:
    #   nsng = nrm * dtpu ; nsng[:, 1:89:2] *= 2 ; asng = acf / nsng
    nsng = nrm * dtpu
    nsng[:, 1:89:2] *= 2  # span-2 correction, same rows as your read script
    asng = np.zeros_like(nsng)
    nz = nsng != 0
    asng[nz] = acf[nz] / nsng[nz]

    # ── Interleaved subsets over the (unflattened) angle axis ────────────────
    subset_indices = signa_interleaved_subsets(n_subsets, Cnt)
    for s, idx in enumerate(subset_indices):
        print(f"  Subset {s}: {len(idx)} angles, "
              f"range [{idx.min()}, {idx.max()}]")

    # ── Pre-compute per-subset sensitivities ──────────────────────────────
    #   s_sub_j = PSF( A_s^T[ asng_s ] )     (asng plays nsng_fwd's role)
    subset_sensitivities = []
    for idx in subset_indices:
        masked_asng = signa_mask_subset(asng, idx, Cnt)
        s_sub = Back_Signa(masked_asng, scanner_params)
        s_sub = (s_sub)  # psf
        s_sub[msk] = 0.0
        subset_sensitivities.append(s_sub)

    # ── Additive correction ──────────────────────────────────────────────
    randoms_scaled = np.zeros_like(randoms)
    randoms_scaled[nz] = randoms[nz] / asng[nz]
    scatter_scaled = np.zeros_like(scatter)
    scatter_scaled[nz] = scatter[nz] / asng[nz]
    additive = (cu.asarray(randoms_scaled, dtype=np.float32)
                + cu.asarray(scatter_scaled, dtype=np.float32))

    # ── BSREM setup ──────────────────────────────────────────────────────
    if alpha_decay is None:
        alpha_decay = float(n_subsets * iterations) / 3.0

    global_subit = 0

    print(f"\nBSREM (Signa): {iterations} iters x {n_subsets} subsets"
          f" | beta={beta}, gamma={gamma}"
          f" | alpha_0={alpha_0}, alpha_decay={alpha_decay:.1f}")

    # ── Main reconstruction loop ──────────────────────────────────────────
    for i in range(iterations):

        for s, idx in enumerate(subset_indices):

            alpha = _alpha(global_subit, alpha_0, alpha_decay)
            s_sub = subset_sensitivities[s]

            # ── Forward projection ──────────────────────────────────────
            # attenuation=False: already folded into asng/additive/s_sub,
            # matching your basic recon script's `fprj(..., 0, sync=True)`.
            fwd = np.asarray(Forward_Signa(pred, scanner_params,
                                            attenuation=False))
            fwd += np.asarray(additive)
            fwd = np.maximum(fwd, 1e-15)

            # ── Subset likelihood gradient ───────────────────────────────
            ratio = div_nzer(sinog, fwd)
            masked_ratio = signa_mask_subset(ratio, idx, Cnt)

            bp_ratio = Back_Signa(masked_ratio, scanner_params)
            bp_ratio = (bp_ratio)  # psf

            grad_L = bp_ratio - s_sub

            # ── RDP prior gradient ────────────────────────────────────────
            grad_R = gradient(pred, gamma, beta) / n_subsets

            # ── EM preconditioner ────────────────────────────────────────
            delta = max(float(delta_frac * float(pred.max())), 1e-8)
            D_s = (pred + delta) / np.maximum(s_sub, 1e-15)
            D_s[msk] = 0.0

            # ── BSREM update ──────────────────────────────────────────────
            pred = pred + alpha * D_s * (grad_L + grad_R)
            pred = np.maximum(pred, 0.0)
            pred[msk] = 0.0

            global_subit += 1

        print(f"\x1b[31m\"Iter {i+1:4d} | "
              f"alpha={_alpha(global_subit, alpha_0, alpha_decay):.5f}| "
              f"pred min/max/mean: {float(pred.min()):.4f} / "
              f"{float(pred.max()):.4f} / {float(pred.mean()):.4f}\"\x1b[0m")

    return pred


# ==============================================================================
# Example usage, following your Signa read script
# ==============================================================================
"""
from pathlib import Path
from niftypet import nipet

mfldr = Path('/sdata/PETpp/20170516_e00179_ANON179')
pthlm = mfldr/'rdf_f1b1.rdf'
Cnt, txLUT, axLUT = nipet.sigaux.init_sig(pthlm)
scanner_params = dict(Cnt=Cnt, txLUT=txLUT, axLUT=axLUT)
Cnt['SPN'] = 2

signa_sinograms = load_signa_sinograms(mfldr, Cnt)

pred = Scatter_BSREM_Updated_Signa(
    sinog=signa_sinograms['prompts'],
    signa_sinograms=signa_sinograms,
    scanner_params=scanner_params,
    iterations=100,
    n_subsets=10,
    beta=0.3,
    gamma=2.0,
)
"""
