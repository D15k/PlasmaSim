import jax.numpy as jnp

def make_periodic_grid(Lx, Lv, Nx, Nv):
    
    ''' 
    Creates a periodic grid for the spatial and velocity coordinates.
    
    Args:
        Lx (float): Spatial domain size.
        Lv (float): Velocity domain size.
        Nx (int): Number of spatial grid points.
        Nv (int): Number of velocity grid points.   
    
    Returns:
        grid (dict): Dictionary containing the grid information.        
    '''
    
    x = jnp.arange(Nx) * Lx / Nx  # Spatial grid
    v = jnp.linspace(-Lv, Lv, Nv)  # Velocity grid

    dx = x[1] - x[0]  # Spatial grid spacing
    dv = v[1] - v[0]  # Velocity grid spacing

    # Create meshgrid for spatial and velocity coordinates
    X, V = jnp.meshgrid(x, v)

    grid = {
        'x': x,
        'v': v,
        'X': X,
        'V': V,
        'dx': dx,
        'dv': dv,
        'Lx': Lx,
        'Lv': Lv,
        'Nx': Nx,
        'Nv': Nv,
        'size': X.shape,
        'dom': [0, -Lv, Lx - dx, Lv - dv]
    }

    return grid