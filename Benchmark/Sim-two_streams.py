import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import PlasmaSim as ps

# Constants
alpha = 0.05 # amplitude of the perturbation
k = 0.2      # wavenumber
Ve = 3       # electron drift velocity

pert = lambda x: alpha * jnp.cos(k * x) # Perturbation function

# Initial distribution function
f0 = lambda x, v: (1 + pert(x)) / (2 * jnp.sqrt(2 * jnp.pi)) * (jnp.exp(-(v - Ve)**2 / 2) + jnp.exp(-(v + Ve)**2 / 2))

# Parameters
params = ps.Parameters(
    N_t = 200, #dt = 1/4
    L_t = 50,
    N_x = 2**9,
    L_x = 2 * jnp.pi / k,
    N_v = 2**9,
    
    computation_method = 'NuFI',
    
    save_freq = 1,
    save_dir = r'C:\Users\Clément\Documents\M2 Physique Fondamentale & Applications - Univers & Particules\Internship\plasma-sim_python-conversion\Benchmark\two_streams',
    name = 'two_streams',
    export = True,
    plot = True
)

# Species
electron = ps.Species(species_name = 'electron', mass = 1, charge = -1, init_distrib_fct = f0, L_v = 5 * jnp.pi)

# Simulation
sim = ps.Simulation(params, [electron])
sim.run()