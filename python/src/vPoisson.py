import jax.numpy as jnp

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
    Ns = len(grids)                  # Number of species, but why not use params.Ns?
    
    for s in range(Ns):
        rho += charge[s] * jnp.sum(fs[:, :, s], axis=1) * grids[s]["dv"]

    # Get grid parameters
    Nx = grids[0]["Nx"]
    Lx = grids[0]["Lx"]
    
    # Define wave numbers
    kx = (2 * jnp.pi / Lx) * jnp.fft.fftshift(jnp.arange(-Nx//2, Nx//2))
    
    # Laplacian operator -|k|^2 (avoid division by zero)
    K2 = kx**2
    K2[0] = 1  # Set the first element to 1 to avoid division by zero
    
    # Solve Poisson equation in Fourier space
    b = jnp.fft.fft(rho)
    phi_fft = -b / K2
    phi_fft[0] = 0  # Set mean to zero
    
    # Compute electric field E = -dphi/dx
    dphi_dx_h = 1j * phi_fft * kx
    Efield = -jnp.real(jnp.fft.ifft(dphi_dx_h))  # Ensure real output
    
    return Efield