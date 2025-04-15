import jax.numpy as jnp
import pandas as pd
import os
import sys
from pathlib import Path

params_path = Path(__file__).resolve().parent.parent
print(params_path)
sys.path.append(str(params_path))

import params.PARAMS_ion_acoustic_waves as params

def measure(params, fs):
    iT = params.it
    time = params.dt * iT
    Efield = params.Efield
    
    for s in range(params.Ns):
        grid = params.grids[s]
        f = fs[:, :, s]             ## May not work
        Vgrid = grid['V']

        dx = grid['dx']
        dv = grid['dv']

        Mass = jnp.sum(f) * dx * dv
        Momentum = jnp.sum(f * Vgrid.T) * dx * dv
        Epot = 0.5 * jnp.sum(Efield ** 2) * dx
        Ekin = 0.5 * jnp.sum(f * (Vgrid.T ** 2)) * dx * dv
        Etot = Epot + Ekin
        L2norm = jnp.sum(jnp.abs(f) ** 2) * dx * dv

        species_name = params.species_name[s]
        filename = Path(params.data_dir) / f"{species_name}.csv"

        new_row = pd.DataFrame([{
            'it': iT,
            'time': time,
            'Mass': Mass,
            'Momentum': Momentum,
            'Epot': Epot,
            'Ekin': Ekin,
            'Etot': Etot,
            'L2norm': L2norm
        }])

        if filename.exists() and iT > 1:
            existing_table = pd.read_csv(filename)
            updated_table = pd.concat([existing_table, new_row], ignore_index=True)
        else:
            updated_table = new_row
            if not hasattr(params, 'diagnostic_filename'):
                params.diagnostic_filename = [None] * params.Ns
            params.diagnostic_filename[s] = str(filename)

        updated_table.to_csv(filename, index=False)

    return params


def fourier_modes(Efield, k_list, grid):
    # Implements basic version of modes based on FFT
    Ek = jnp.fft.fft(Efield)
    Emode_abs = jnp.abs(Ek[1:5]) / grid.size[0]
    
    return Emode_abs