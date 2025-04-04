import jax.numpy as jnp

### General parameters ###
mycase         = "ion_accoustic_waves"     # "two_stream"
Nx             = 2**8                      # Number of spatial grid points
Nv             = 2**9                      # Number of velocity grid points
Ns             = 2                         # Number of species (electrons and ions)
method         = "NuFi"
species_name   = ["electrons", "ions"]     # Name of the different species
Mr             = 1000                      # Mass ratio for ions
Mass           = [1, Mr]                   # Mass of species
charge         = [-1, 1]                   # Charge of species
Nt_max         = 4000                      # Maximum number of time steps
dt             = 1/4                       # Time step size
dt_save        = 100                       # Save after dt_save time
Tend           = 500                       # End time of simulation

### Initial condition parameters ###
k      = 0.5                               # Wave number
alpha  = 0.5                               # Perturbation amplitude
pert   = lambda x: alpha * jnp.cos(k * x)  # Perturbation function

### Electrons ###
Ue     = -2                                                                                 # Electron drift velocity
fe0    = lambda x, v: (1 + pert(x)) / (jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(v - Ue) ** 2 / 2)  # Electron distribution
fi0    = lambda x, v: jnp.sqrt(Mr / (2 * jnp.pi)) * jnp.exp(-Mr * v ** 2 / 2)               # Ion distribution
fini   = [fe0, fi0]

### Grid parameters ###
Lx     = 2 * jnp.pi / k                    # Spatial domain length
Lv_s   = [8, 0.1 * jnp.pi]                 # Velocity domain lengths for each species