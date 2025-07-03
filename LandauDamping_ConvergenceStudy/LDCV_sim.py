import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import PlasmaSim as ps

# Constants
alpha = 0.05 # amplitude of the perturbation
k = 0.5      # wavenumber

pert = lambda x: alpha * jnp.cos(k * x) # Perturbation function

# Initial distribution function
f0 = lambda x, v: (1 + pert(x)) / jnp.sqrt(2 * jnp.pi) * jnp.exp(-v**2 / 2) 

L_t = 2**(-8)
dt = 2**(-8)     # 2**(-8), 2**(-7), 2**(-6), 2**(-5), 2**(-4), 2**(-3) ## 2**(-2)
N = 2**5         # 2**5, 2**6, 2**7, 2**8, 2**9


# Parameters
params = ps.Parameters(
    N_t = int(L_t / dt),
    L_t = L_t,
    N_x = N,
    L_x = 2 * jnp.pi / k,
    N_v = N,
    
    computation_method = 'NuFI',            
    
    save_freq = int(L_t / dt),
    save_dir = r'C:\Users\Clément\Documents\M2 Physique Fondamentale & Applications - Univers & Particules\Internship\plasma-sim_python-conversion\test_err_cv',
    name = f'LDCV_N_{N}_dt_{dt}',
    export = True,
    plot = False
)

# Species
electron = ps.Species(species_name = 'electron', mass = 1, charge = -1, init_distrib_fct = f0, L_v = 4*jnp.pi)

# Simulation
sim = ps.Simulation(params, [electron])
sim.run()