from .utils import extract_attrs

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit


def chargeDensity(species):
    """
    Computes the charge density using the distribution function of each species.
    """
    # Extract attributes from species as jax arrays for easier computation
    charges = extract_attrs(species, 'charge')
    curt_distrib_fct = extract_attrs(species, 'curt_distrib_fct')
    dv = extract_attrs(species, 'dv')
    
    # Compute charge density for each species
    rho_species = charges * jnp.sum(curt_distrib_fct, axis=2) * dv
    
    # Sum over all species to get total charge density
    rho = jnp.sum(rho_species, axis=0)
    
    return rho


def Poisson(params):
    """
    Solves the Poisson equation in Fourier space to compute the electric field using the Fourier-Galerkin method.

    Args:
        fs (numpy array)      : Distribution function of shape (Nx, Nv, Ns).
        grids (list of dicts) : Grid information for each species.
        charge (list)         : Charge of each species.

    Returns:
        Efield (numpy array)  : Computed electric field.
    """
    # Get grid parameters
    N_x = params.N_x
    L_x = params.L_x
    
    ### Compute charge density for each species
    rho = chargeDensity(params.sim_species)
   
    # Define wave numbers
    kx = (2 * jnp.pi / L_x) * jnp.fft.fftshift(jnp.arange(-N_x//2, N_x//2))
    
    # Laplacian operator -|k|^2 (avoid division by zero)
    K2 = kx**2
    K2 = K2.at[0].set(1) # avoid devision by 0     ###TODO: Understand why we do this
    # to avoid a division by zero, we set the zeroth wavenumber to one.
    # this leaves it's respective Fourier coefficient unaltered, so the
    # zero mode of Sk is conserved.dphi_dx_h = 1i*phi_fft.*kx(1,:); This way, Sk's zero mode implicitly
    # defined the zero mode of the result
    # Note that the zero mode is NOT uniquely defined: in a periodic
    # setting, the solution of Laplace's (or Poisson's) equation is only
    # defined up to a constant! You can freely overwrite the zero mode,
    # therefore.
    
    # Solve Poisson equation in Fourier space
    b = jnp.fft.fft(rho)
    phi_fft = -b / K2
    phi_fft = phi_fft.at[0].set(0)  # Set mean to zero
    
    # Compute electric field E = -dphi/dx
    dphi_dx_h = 1j * phi_fft * kx
    Efield = -jnp.real(jnp.fft.ifft(dphi_dx_h))  # Ensure real output
    
    return Efield