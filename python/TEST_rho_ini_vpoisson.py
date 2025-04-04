from src.grid.make_periodic_grid import make_periodic_grid
import params.PARAMS_ion_acoustic_waves as params
import jax.numpy as jnp

jnp.set_printoptions(threshold=jnp.inf)

charge = params.charge
Ns = params.Ns
Lx = params.Lx
Lv_s = params.Lv_s
Nx = params.Nx
Nv = params.Nv
finis = params.fini

# Initialize grids and distribution functions
grids = []
for s in range(Ns):
    grid = make_periodic_grid(Lx, Lv_s[s], Nx, Nv)
    grid_methode = "spline"
    grids.append(grid)
    
fs = jnp.zeros((Nx, Nv, Ns))
for s in range(Ns):
    #print(finis)
    fini = finis[s]                                      #access the distribution function in the params for the s-th species
    #print(fs[:, :, s].shape)
    #print(fini(grids[s]['X'], grids[s]['V']).shape)
    #print(grids[s]['X'].shape, grids[s]['V'].shape)
    fs.at[:, :, s].set(fini(grids[s]['X'], grids[s]['V']).T)

Ns = len(grids)                  # Number of species, but why not use params.Ns?

rho = jnp.zeros(grids[0]["Nx"])  # Initialize rho with zeros
for s in range(Ns):
    print(fs[:, :, s])
    rho += charge[s] * jnp.sum(fs[:, :, s], axis=1) * grids[s]['dv']
print(rho[255])