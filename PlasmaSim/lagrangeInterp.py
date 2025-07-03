import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial


def periodicLocLagInterp(x_target, xgrid, ygrid):
    return JAX_lagrange_local_interp_periodic(x_target, xgrid, ygrid, 4)
   
    
@partial(jax.jit, static_argnames=['order'])
def JAX_lagrange_local_interp_periodic(x_target, xgrid, ygrid, order):
    """
    Interpolates y = f(x) at x_target using local Lagrange interpolation.
    xgrid: equispaced periodic grid (assumed 1D)
    ygrid: function values at xgrid
    order: number of points used in local stencil (e.g., 4 for cubic)
    x_target: array of target points in [0,1)
    """
    N = len(xgrid)
    delta_x = xgrid[1] - xgrid[0]  # uniform spacing assumed
    f_interp = jnp.zeros_like(x_target)
    half = order // 2
    x_local = jnp.arange(-half, order - half + 1)  # -half to half+order inclusive

    idx0 = x_target / delta_x
    j0 = jnp.floor(idx0).astype(jnp.int64)
    delta_idx = idx0.flatten() - j0.flatten()    
    
    # Local grid and values with periodic wrap-around
    idx_list = (j0[:,None] - half + jnp.arange(order + 1)) % N
    y_local = ygrid[idx_list]

    
    # Evaluate local Lagrange interpolant
    Ljs  = JAXlagrange_basis(delta_idx, x_local, order)
    p = jnp.sum(y_local.T * Ljs, axis = 0)

    f_interp = p

    return f_interp


def JAXlagrange_basis(x: jnp.array, x_nodes: jnp.array, order: int):
    """
    Computes the j-th Lagrange basis polynomial evaluated at x.
    """
    x = jnp.array(x)
    x_nodes = jnp.array(x_nodes)
     
    xj = x_nodes[:order + 1]
    
    tempA = (x[:, None] - x_nodes)
    tempB = (xj[:, None] - x_nodes + 1e-32)
       
    temp = jnp.array(tempA[ :, None] / tempB) # epsilon to prevent division by zero
    temp = jnp.transpose(temp, axes = (1,2,0))
    
    #mask = jnp.eye(order + 1, dtype=bool)
    mask = jnp.broadcast_to(jnp.eye(order + 1), (len(x), order + 1, order + 1)).T
    A = jnp.where(mask.astype(bool), mask, temp)

    Lj = jnp.prod(A, axis=1)
    
    return Lj