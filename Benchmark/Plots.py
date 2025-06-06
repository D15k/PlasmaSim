from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def load_python_data(file_path):
    python = loadmat(file_path)
    python_fs = python['distrib_fct']
    Efield = python['Efield']
    distrib_fcts = python_fs[:,:,:,:]
        
    return Efield, distrib_fcts

def plot(grid, distrib_fcts):
    N_t = distrib_fcts.shape[0]
    N_s = distrib_fcts.shape[1]
    
    T = [0, (N_t-1)//2, N_t-1]
    
    S = 12
    plt.rcParams['font.size'] = S
    plt.rcParams['axes.labelsize'] = S
    plt.rcParams['xtick.labelsize'] = S
    plt.rcParams['ytick.labelsize'] = S
    
    for t in T:
        fig, axes = plt.subplots(N_s, 1, figsize=(10, 4*(N_s + 1)))
        if N_s == 1:
            axes = [axes]  # Convert single Axes to list for consistent handling
        
        # Plot each species' distribution function
        for s in range(N_s):
            ax = axes[s]
            im = ax.pcolormesh(grid[s][0], grid[s][1], distrib_fcts[t, s].T, shading='auto')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$v$')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        print(t)
        plt.show()

main_path = r'C:\Users\Clément\Documents\M2 Physique Fondamentale & Applications - Univers & Particules\Internship\plasma-sim_python-conversion\Benchmark'

#path = main_path + r'\ion_acoustic_waves'
#path = main_path + r'\landau_damping'
path = main_path + r'\two_streams'

_, distrib_fcts = load_python_data(path + r'\data_python.mat')

x = lambda N_x, L_x: jnp.arange(N_x) * L_x / N_x               
v = lambda N_v, L_v: jnp.linspace(-L_v, L_v, N_v) 

k = 0.5
N_x = N_v = 2**9
L_x = 2 * jnp.pi / k
L_v = 8
L_v2 = 0.1 * jnp.pi

grids = [[x(N_x, L_x), v(N_v, L_v)]] #,[x(N_x, L_x), v(N_v, L_v2)]

plot(grids, distrib_fcts)