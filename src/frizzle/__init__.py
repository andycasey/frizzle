import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import linear_solve
from functools import partial
from sklearn.neighbors import KDTree
from typing import Optional

from jax_finufft import nufft1, nufft2
from .utils import check_inputs, combine_flags

def frizzle(
    λ_out: jnp.array,
    λ: jnp.array,
    flux: jnp.array,
    ivar: Optional[jnp.array] = None,
    mask: Optional[jnp.array] = None,
    flags: Optional[jnp.array] = None,
    censor_missing_regions: Optional[bool] = True,
    n_modes: Optional[int] = None,
) -> tuple[jnp.array, jnp.array, jnp.array, dict]:
    """
    Combine spectra by forward modeling.

    :param λ_out:
        The wavelengths to sample the combined spectrum on.
    
    :param λ:
        The wavelengths of the individual spectra. This should be shape (N, ) where N is the number of pixels.
    
    :param flux:
        The flux values of the individual spectra. This should be shape (N, ).
    
    :param ivar: [optional]
        The inverse variance of the individual spectra. This should be shape (N, ).
    
    :param mask: [optional]
        The mask of the individual spectra. If given, this should be a boolean array (pixels with `True` get ignored) of shape (N, ).
        The mask is used to ignore pixel flux when combining spectra, but the mask is not used when computing combined pixel flags.
    
    :param flags: [optional]
        An optional integer array of bitwise flags. If given, this should be shape (N, ).
        
    :param censor_missing_regions: [optional]
        If `True`, then regions where there is no data will be set to NaN in the combined spectrum. If `False` the values evaluated
        from the model will be reported (and have correspondingly large uncertainties) but this will produce unphysical features.

    :param n_modes: [optional]
        The number of Fourier modes to use. If `None` is given then this will default to `len(λ_out)`.
            
    :returns:
        A four-length tuple of:
            - the combined fluxes;
            - the diagonal of the covariance matrix of combined fluxes;
            - the combined flags; and
            - a metadata dictionary.
    """

    λ_out, λ, flux, ivar, mask = check_inputs(λ_out, λ, flux, ivar, mask)
    n_modes = n_modes or min([len(λ_out), int(np.sum(~mask))])

    y_star, C_star = _frizzle_materialized(λ_out, λ[~mask], flux[~mask], ivar[~mask], n_modes)

    meta = dict(n_modes=n_modes)
    if censor_missing_regions:
        # Set NaNs for regions where there were NO data.
        # Here we check to see if the closest input value was more than the output pixel width.
        tree = KDTree(λ.reshape((-1, 1)))
        distances, indices = tree.query(λ_out.reshape((-1, 1)), k=1)

        no_data = jnp.hstack([distances[:-1, 0] > jnp.diff(λ_out), False])
        meta["no_data_mask"] = no_data
        if jnp.any(no_data):
            y_star = jnp.where(no_data, jnp.nan, y_star)
            C_star = jnp.where(no_data, jnp.inf, C_star)

    flags_star = combine_flags(λ_out, λ, flags)

    return (y_star, C_star, flags_star, meta)


@partial(jax.jit, static_argnames=("n_modes",))
def matvec(c, x, n_modes, weights):
    return jnp.real(nufft2(_pre_matvec(c, n_modes), x)) * weights

@partial(jax.jit, static_argnames=("n_modes",))
def rmatvec(f, x, n_modes, weights):
    dtype = jnp.array(0.0 + 0.0j).dtype
    return _post_rmatvec(nufft1(n_modes, f.astype(dtype) * weights, x), n_modes)

matmat = jax.vmap(matvec, in_axes=(0, None, None, None))
rmatmat = jax.vmap(rmatvec, in_axes=(0, None, None, None))

@partial(jax.jit, static_argnames=("n_modes", ))
def _frizzle_materialized(λ_out, λ, flux, ivar, n_modes):
    """
    frizzle some input spectra using materialized matrices.
    """
    # The domain over which we care about.
    λ_min, λ_max = (λ_out[0], λ_out[-1])

    edge = 2 * jnp.pi / len(λ_out)
    scale = 2 * jnp.pi * (1 - edge) / (λ_max - λ_min)

    x = (λ - λ_min) * scale 
    x_star = (λ_out - λ_min) * scale

    I = jnp.eye(n_modes)

    C_inv_sqrt = jnp.sqrt(ivar)    
    weighted_x_mv = lambda c: matvec(c, x, n_modes, C_inv_sqrt)
    θ = linear_solve.solve_normal_cg(
        weighted_x_mv,
        flux * C_inv_sqrt, 
        init=jnp.zeros(n_modes)
    )

    ATCinv = matmat(I, x, n_modes, 1) * ivar
    ATCinvA = rmatmat(ATCinv, x, n_modes, 1)
    A_star_T = matmat(I, x_star, n_modes, 1)
    
    # Cholesky decomposition is faster, but fails for some edge cases.
    #cho_factor = jax.scipy.linalg.cho_factor(ATCinvA)        
    #θ = jax.scipy.linalg.cho_solve(cho_factor, ATCinv @ flux)
    #y_star = matvec(θ, x_star, n_modes)
    #ATCinvA_inv = jax.scipy.linalg.cho_solve(cho_factor, I)

    ATCinvA_inv, *extras = jnp.linalg.lstsq(ATCinvA, I, rcond=None)

    y_star = matvec(θ, x_star, n_modes, 1)
    C_star = jnp.diag(A_star_T.T @ ATCinvA_inv @ A_star_T)
    return (y_star, C_star)


@partial(jax.jit, static_argnames=("p", ))
def _pre_matvec(c, p):
    """
    Enforce Hermitian symmetry on the Fourier coefficients.

    :param c:
        The Fourier coefficients (real-valued).
    
    :param p:
        The number of modes.
    """
    f = (
        0.5  * jnp.hstack([c[:p//2+1],   jnp.zeros(p-p//2-1)])
    +   0.5j * jnp.hstack([jnp.zeros(p//2+1), c[p//2+1:]])
    )
    return f + jnp.conj(jnp.flip(f))

@partial(jax.jit, static_argnames=("p",))
def _post_rmatvec(f, p):
    f_flat = f.flatten()
    return jnp.hstack([jnp.real(f_flat[:p//2+1]), jnp.imag(f_flat[-(p-p//2-1):])])

