import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import time
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

L_t = 2 #50
dt = 2**(-6)
N_t = int(L_t / dt)

# Parameters
params = ps.Parameters(
    N_t = N_t,
    L_t = L_t,
    N_x = 2**8,
    L_x = 2 * jnp.pi / k,
    N_v = 2**8,
    
    computation_method = 'NuFI'
)

# Species
electron = ps.Species(species_name = 'electron', mass = M, charge = -1, init_distrib_fct = f0, L_v = 12)



start = time.perf_counter()

sim = ps.Simulation(params, [electron])
sim.run()

end = time.perf_counter()
print(f"Temps d'exécution de la simulation : {end - start:.2f} secondes")