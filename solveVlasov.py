import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from .solvePoisson import Poisson

def NuFi(iter, params, sim_species, hist_Efield):
    """
    Update distribution functions and electric field using symplectic flow.
    
    Args:
        params: Simulation parameters
        fs: Distribution functions array
    
    Returns:
        tuple: Updated (fs, Efield)
    """
    dt = params.dt
    
    for species_i in sim_species:
        X, V = species_i.grid[0], species_i.grid[1]
        charge = species_i.charge
        mass = species_i.mass
        f = species_i.curt_distrib_fct
        
        X_new, V_new = sympl_flow_Half(
            iter, 
            dt, 
            X, 
            V, 
            (charge / mass) * hist_Efield, 
            X.shape
        )
        
        fini = species_i.init_distrib_fct
        f = f.at[:, :].set(fini(X_new, V_new).T)
    
    return sim_species

def sympl_flow_Half(n, dt, X, V, Efield, grid):
    """
    Perform symplectic flow for half timestep.
    
    Args:
        n: Current timestep
        dt: Time step size
        X: Position array
        V: Velocity array
        Efield: Electric field array
        grid: Grid parameters
    
    Returns:
        tuple: Updated (X, V)
    """
    if n == 1:
        return X, V
        
    def periodic(x):
        return jnp.mod(x, grid.Lx - grid.dx)
    
    while n > 2:
        n = n - 1
        X = X - dt * V  # Inverse signs; going backwards in time
        E_interp = InterpolatedUnivariateSpline(X, Efield[:, n])
        V = V + dt * E_interp(periodic(X)).reshape(grid.size)
    
    X = X - dt * V
    E_interp = InterpolatedUnivariateSpline(X, Efield[:, 1])
    V = V + (dt / 2) * E_interp(periodic(X)).reshape(grid.size)
    
    return X, V