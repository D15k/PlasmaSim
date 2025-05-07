import jax.numpy as jnp
import PlasmaSim as ps

alpha = 0.01 # Perturbation amplitude    
k = 0.5 # Wave number
pert = lambda x: alpha * jnp.cos(k * x) # Perturbation function
Ue = -2 # Electron drift velocity
M = 1000 # Ion mass

fe0 = lambda x, v: ((1 + pert(x)) / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(v - Ue) ** 2 / 2))  # Electron distribution
fi0 = lambda x, v: jnp.sqrt(M / (2 * jnp.pi)) * jnp.exp(-M * v ** 2 / 2)                 # Ion distribution


params = ps.Parameters(
    computation_method = 'NuFi',
 
    N_t = 401,
    L_t = 100,
    N_x = 2**8,
    L_x = 2*jnp.pi / k,
    N_v = 2**9,

    save_freq = 50,
    name = 'ion_acoustic_wave',
    export = True,
    plot = True
)

electron = ps.Species(species_name = 'electron', mass = 1, charge = -1, init_distrib_fct = fe0, L_v = 8)
ion      = ps.Species(species_name = 'ion', mass = 1000, charge = 1, init_distrib_fct = fi0, L_v = 0.1 * jnp.pi)


sim = ps.Simulation(params, [electron, ion])
sim.run()
