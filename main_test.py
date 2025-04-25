import jax.numpy as jnp
import PlasmaSim as ps

'''
### Initial condition parameters ###
k      = 0.5                               # Wave number
alpha  = 0.5                               # Perturbation amplitude
pert   = lambda x: alpha * jnp.cos(k * x)  # Perturbation function
### Electrons ###
Mr     = 1000                                                                                   # Mass ratio for ions
Ue     = -2                                                                                 # Electron drift velocity
fe0    = lambda x, v: (1 + pert(x)) / (jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(v - Ue) ** 2 / 2)  # Electron distribution
fi0    = lambda x, v: jnp.sqrt(Mr / (2 * jnp.pi)) * jnp.exp(-Mr * v ** 2 / 2)               # Ion distribution
'''
fe0    = lambda x, v: (1 + jnp.cos(x/2)/2) / (jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(v - (-2)) ** 2 / 2)  # Electron distribution
fi0    = lambda x, v: jnp.sqrt(1000 / (2 * jnp.pi)) * jnp.exp(-1000 * v ** 2 / 2)                    # Ion distribution


params = ps.Parameters(
    computation_method = 'NuFi',
    
    N_t = 41,
    L_t = 10,
    N_x = 2**3,
    L_x = jnp.pi,
    N_v = 2**4,

    save_freq = 10,
    name = 'test'
)

electron = ps.Species(species_name = 'electron', mass = 1, charge = -1, init_distrib_fct = fe0, L_v = 2)
ion = ps.Species(species_name = 'ion', mass = 1000, charge = 1, init_distrib_fct = fi0, L_v = 1)


sim = ps.Simulation(params, [electron, ion])
sim.run()