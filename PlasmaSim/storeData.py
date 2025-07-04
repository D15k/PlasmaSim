import numpy as np
from os import makedirs
from scipy.io import savemat
import matplotlib.pyplot as plt
from tqdm import tqdm
from .utils import copy_attrs

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


class StoreData:
    '''
    Class to store and export data from the simulation.
    '''

    def __init__(self, sim):       
        # Import useful parameters from simulation for later use
        self.sim = type('', (), {})()
        copy_attrs(sim, self.sim, ['__', 'hist_', 'stored_'])
                
        # Create save directory for plotting if necessary
        self.save_path = f'{self.sim.save_dir}/{self.sim.name}'
        if self.sim.plot is True: makedirs(self.save_path, exist_ok=True)
        
        self.N_save = 1 + self.sim.N_t // self.sim.save_freq # Number of time steps where data is saved
        
        # Initialize data structure to store or plot
        self.save_distrib_fct = jnp.zeros((self.N_save, self.sim.N_s, self.sim.N_x, self.sim.N_v)) # History of the distribution functions to be plotted or exported
        self.save_Efield = jnp.zeros((self.N_save, self.sim.N_x)) # History of the electric field to be plotted or exported
        
        # Store initial distribution functions and electric field
        all_init_distrib_fct = jnp.array([species.curt_distrib_fct for species in self.sim.sim_species])
        
        self.save_distrib_fct = self.save_distrib_fct.at[0, :, :, :].set(all_init_distrib_fct)    
        self.save_Efield = self.save_Efield.at[0, :].set(self.sim.curt_Efield)
        
        
    def update(self, iter, sim_species, curt_Efield):
        save_iter = iter / self.sim.save_freq
        # Only save data at intervals specified by save_freq
        # save_iter is the iteration number divided by save_freq
        # If save_iter is a whole number, it means we've hit a save point
        if save_iter % 1 == 0:
            all_curt_distrib_fct = [species_i.curt_distrib_fct for species_i in sim_species]
            
            self.save_distrib_fct = self.save_distrib_fct.at[int(save_iter), :, :, :].set(all_curt_distrib_fct) # Update the saved distribution functions for all species at this save point
            self.save_Efield = self.save_Efield.at[int(save_iter), :].set(curt_Efield) # Update the saved electric field at this save point 


    def export(self):
        # Create a dictionary with the data to save
        data = {
            'sim_parameters': self.sim,
            'distrib_fct': self.save_distrib_fct,  # Distribution functions for all species sampled at save frequency
            'Efield': self.save_Efield  # Electric field values sampled at save frequency
        }
        # Save the data to a .mat file using scipy.io.savemat with the simulation name and the current date and time
        if self.sim.plot is True: savemat(f'{self.save_path}/{self.sim.name}.mat', data)
        else: savemat(f'{self.save_path}.mat', data)
        
        
    def plot(self):
        self.max_E = jnp.max(jnp.abs(self.save_Efield))
        for t in tqdm(range(self.N_save), desc = 'Plotting frames'): 
            self.__plot_frame_and_save(t)


    def __plot_frame_and_save(self, t: int, image_format: str = 'png'):
        '''
        Plot the distribution functions for all species and the electric field
        at timestep index `t`, then save the figure to disk.

        Parameters
        ----------
        save_path : str
            Directory path where to save the figure
        t : int
            Index into the saved data arrays (0 <= t < self.N_save).
        image_format : str, optional
            File format for output (e.g. 'png', 'jpeg', 'svg', 'pdf').
        '''
        
        # Prepare data
        distrib = np.array(self.save_distrib_fct)   # shape (N_save, N_s, N_x, N_v)
        Efield  = np.array(self.save_Efield)        # shape (N_save, N_x)

        # Create figure with subplots
        fig, axes = plt.subplots(self.sim.N_s + 1, 1, figsize=(10, 4*(self.sim.N_s + 1)))
        
        # Plot each species' distribution function
        for s in range(self.sim.N_s):
            ax = axes[s]
            im = ax.pcolormesh(self.sim.sim_species[s].grid[0], self.sim.sim_species[s].grid[1], distrib[t, s].T, shading='auto')
            ax.set_title(f'$f_{{{self.sim.sim_species[s].species_name}}}$')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$v$')
            plt.colorbar(im, ax=ax)

        # Plot electric field
        ax = axes[-1]
        ax.plot(jnp.arange(self.sim.N_x) * self.sim.L_x / self.sim.N_x, Efield[t])
        ax.set_title('$E$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$E(x)$')
        ax.set_ylim(-self.max_E, self.max_E)

        plt.tight_layout()  
        plt.savefig(f'{self.save_path}/frame_{t}.{image_format}') # Save the figure
        plt.close()