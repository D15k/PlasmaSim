import jax.numpy as jnp
import params.PARAMS_ion_acoustic_waves as params

def vPoisson(fs, grids, charge):
    """
    Solves the Poisson equation in Fourier space to compute the electric field.

    Args:
        fs (numpy array)      : Distribution function of shape (Nx, Nv, Ns).
        grids (list of dicts) : Grid information for each species.
        charge (list)         : Charge of each species.

    Returns:
        Efield (numpy array)  : Computed electric field.
    """
    
    ### Compute charge density for each species
    rho = jnp.zeros(grids[0]["Nx"])  # Initialize rho with zeros
    Ns = params.Ns                   # Number of species, but why not use params.Ns?
    
    for s in range(Ns):
        rho += charge[s] * jnp.sum(fs[:, :, s], axis=1) * grids[s]["dv"]
        
    # Get grid parameters
    Nx = params.Nx
    Lx = params.Lx     
    
    # Define wave numbers
    kx = (2 * jnp.pi / Lx) * jnp.fft.fftshift(jnp.arange(-Nx//2, Nx//2))
    
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