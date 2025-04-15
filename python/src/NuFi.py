from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from copy import deepcopy

import jax.numpy as jnp
import pandas as pd
import os
import sys
from pathlib import Path

params_path = Path(__file__).resolve().parent.parent
print(params_path)
sys.path.append(str(params_path))

import params.PARAMS_ion_acoustic_waves as params

from src.vPoisson import *

def NuFi(params, fs):
    iT = params.it + 1
    dt = params.dt

    for s in range(1, params.Ns):
        X = params.grids[s]['X']
        V = params.grids[s]['V']
        Efield_list = params.Efield_list
        charge = params.charge[s]
        mass = params.Mass[s]
        grid = params.grids[s]
        
        Efield_array = jnp.array(Efield_list)  # Convert Efield_list to a NumPy array
        X_new, V_new = sympl_flow_Half(iT, dt, X, V, (charge / mass) * Efield_array, grid)
        fini = params.fini[s]
        fs = fs.at[:, :, s].set(fini(X_new, V_new).T) 

    Efield = vPoisson(fs, params.grids, params.charge)
    params.Efield = Efield
    params.Efield_list.append(Efield)

    return fs, params


def sympl_flow_Half(n, dt, X, V, Efield, grid):
    if n == 1:
        return X, V

    def periodic(x):
        
        return jnp.mod(x, grid['Lx'] - grid['dx'])

    X = deepcopy(X)         ##TODO: Check if deepcopy is necessary (GPT)
    V = deepcopy(V)

    while n > 2:
        n -= 1
        X = X - dt * V
             
        # Interpolate Efield at timestep n
        E_interp = InterpolatedUnivariateSpline(grid['x'], Efield.T[:, n])
        V = V + dt * E_interp(periodic(X)).reshape(grid['size'])

    # Final step with half update
    X = X - dt * V
    E_interp = InterpolatedUnivariateSpline(grid['x'], Efield.T[:, 0])
    V = V + 0.5 * dt * E_interp(periodic(X)).reshape(grid['size'])

    return X, V