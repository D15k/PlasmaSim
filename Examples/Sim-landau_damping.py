import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import PlasmaSim as ps

# Constants
alpha = 0.05 # amplitude of the perturbation
k = 0.5     # wavenumber
Ue = 0      # ion drift velocity
M = 1       # mass

pert = lambda x: alpha * jnp.cos(k * x) # Perturbation function

# Initial distribution functions
f0 = lambda x, v: (1 + pert(x)) / jnp.sqrt(2 * jnp.pi) * jnp.exp(-v**2 / 2) 

# Parameters
params = ps.Parameters(
    N_t = 160, #dt = 1/2
    L_t = 80,
    N_x = 2**9,
    L_x = 2 * jnp.pi / k,
    N_v = 2**9,
    
    computation_method = 'NuFI',
    
    save_freq = 1,
    save_dir = r'C:\Users\Clément\Documents\M2 Physique Fondamentale & Applications - Univers & Particules\Internship\plasma-sim_python-conversion\Benchmark\landau_damping',
    name = 'landau_damping',
    export = True,
    plot = False
)

# Species
electron = ps.Species(species_name = 'electron', mass = M, charge = -1, init_distrib_fct = f0, L_v = 12)

sim = ps.Simulation(params, [electron])
sim.run()