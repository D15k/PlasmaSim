import jax.numpy as jnp

### General parameters ###
mycase         = "landau_damping"          # "two_stream"
Nx             = 2**8                      # Number of spatial grid points
Nv             = 2**8                      # Number of velocity grid points
Ns             = 1                         # Number of species (electrons and ions)
method         = "predcorr"
species_name   = ["electrons", "ions"]     # Name of the different species
Mr             = 1                         # Mass ratio for ions
Mass           = [1, Mr]                   # Mass of species
charge         = [-1, 1]                   # Charge of species
Nt_max         = 4000                      # Maximum number of time steps
dt             = 1/2                       # Time step size

Tend           = 40                        # End time of simulation

### Initial condition parameters ###
k      = 0.5                               # Wave number
alpha  = 0.2                               # Perturbation amplitude
pert   = lambda x: alpha * jnp.cos(k * x)  # Perturbation function

### Electrons ###
Ue     = 0                                                                            # Electron drift velocity
fe0    = lambda x, v: (1 + pert(x)) / (jnp.sqrt(2 * jnp.pi)) * jnp.exp(- v ** 2 / 2)  # Electron distribution
fi0    = lambda    v: jnp.sqrt(Mr / (2 * jnp.pi)) * jnp.exp(-Mr * v ** 2 / 2)         # Ion distribution
fini   = [fe0, fi0]

### Grid parameters ###
Lx     = 2 * jnp.pi / k                    # Spatial domain length
Lv_s   = [12, 0.1 * jnp.pi]                # Velocity domain lengths for each species