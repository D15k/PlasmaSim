import jax.numpy as jnp
from scipy.interpolate import interp1d

def NuFi(iter, species_i, params):
    """
    Update distribution functions and electric field using symplectic flow.
    
    Args:
        params: Simulation parameters
        fs: Distribution functions array
    
    Returns:
        tuple: Updated (fs, Efield)
    """
    dt = params.dt
    
    X, V = species_i.grid[0], species_i.grid[1]
    charge = species_i.charge
    mass = species_i.mass
    fini = species_i.init_distrib_fct
    f = species_i.curt_distrib_fct
    
    X_new, V_new = sympl_flow_Half(iter, dt, X, V, (charge / mass) * params.hist_Efield, params)
        
    f = f.at[:, :].set(fini(X_new, V_new).T)
    
    return f

def sympl_flow_Half(n, dt, X, V, Efield, params):
    """
    Perform symplectic flow for half timestep.
    
    Args:
        n: Current timestep
        dt: Time step size
        X: Position array
        V: Velocity array
        Efield: Electric field array
        species_params: species_params parameters
    
    Returns:
        tuple: Updated (X, V)
    """
    if n == 0:
        return X, V
    
    def periodic(x):
        return jnp.mod(x, params.L_x - params.dx)
    
    Xshape = X.shape

    while n > 1:
        n = n - 1
        X = X - dt * V  # Inverse signs; going backwards in time
        E_interp = interp1d(X[0,:], Efield[n, :], kind='cubic', bounds_error=False, fill_value=(Efield[n, 0], Efield[n, -1]))
        V = V + dt * E_interp(periodic(X))
    
    X = X - dt * V
    E_interp = interp1d(X[0,:], Efield[1, :], kind='cubic', bounds_error=False, fill_value=(Efield[1, 0], Efield[1, -1]))
    V = V + (dt / 2) * E_interp(periodic(X))
    
    return X, V