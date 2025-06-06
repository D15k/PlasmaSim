import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.interpolate import CubicSpline


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
    
    X_new, V_new = sympl_flow_Half(iter + 1, dt, X, V, (charge / mass) * params.hist_Efield, params) # not sure its iter + 1 or iter
    
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
    if n == 1:
        return X, V
       
    def periodic(x):
        m = params.L_x - params.dx
        return jnp.mod(x, m)
    
    x = jnp.arange(params.N_x) * params.L_x / params.N_x
    Xshape = X.shape # size (N_x, N_v)

    while n > 2:
        n = n - 1
        X = X - dt * V  # Inverse signs; going backwards in time
        
        P = periodic(X) # size (N_x * N_v, 1)
        RP = P.reshape((-1, 1), order='F')
        E = (Efield[n-1, :])
        E_interp = CubicSpline(x, E)
        I = E_interp(RP)
        RI = I.reshape(Xshape, order='F')
        V = V + dt * RI
  
    X = X - dt * V
    P = periodic(X)
    RP = P.reshape((-1, 1), order='F') # size (N_x * N_v, 1)
    E_interp = CubicSpline(x, Efield[0, :])
    I = E_interp(RP)
    RI = I.reshape(Xshape, order='F')
    V = V + (dt / 2) * RI
    
    return X, V