import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt

from .initialization import Parameters, Species
from .solvePoisson import Poisson  # Use the Poisson solver from solvePoisson.py
from .solveVlasov import NuFi#, CMM, predictor_corrector
from .storeData import StoreData

class Simulation:
    '''
    Class performing the simulation of a plasma system using the specified parameters and species.
    
    Attributes:
        params (Parameters): Parameters of the simulation.
        sim_species (Species | list[Species]): Species involved in the simulation.
        
        verbose (bool, optional): Whether to print simulation summary and progress. Defaults to True.
    '''
    
    def __init__(self, params: Parameters, sim_species: Species | list[Species], verbose: bool = True):
        self.verbose = verbose
        # Import attributes from params
        for attr in dir(params):
            if not attr.startswith("__") and not callable(getattr(params, attr)):
                setattr(self, attr, getattr(params, attr))
                
        # Compute initial variables that cannot be computed in the Parameters and Species classes
        self.N_s = len(sim_species) # Number of species
        
        self.sim_species = sim_species
        for species_i in self.sim_species:
            species_i.dv = 2*species_i.L_v/(self.N_v - 1) # Velocity step
            species_i.grid = self.__build_PhaseSpaceGrid(species_i) # Grid of points on the spatial and velocity axes
            species_i.curt_distrib_fct = self.__build_InitDistribFct(species_i) # Initial distribution function
        
        self.hist_Efield = jnp.zeros((self.N_t, self.N_x)) # History of the electric field
        self.curt_Efield = Poisson(self) # Compute initial electric field and store it as the current electric field
        self.hist_Efield = self.hist_Efield.at[0, :].set(self.curt_Efield) # Store the initial electric field
        
        # Initialize data structure to export or plot
        if self.plot or self.export: self.stored_data = StoreData(self)

        # Print initial simulation summary for debugging purposes
        if self.verbose:
            print('Simulation initialized')
            print('Simulation parameters:')
            for attr in dir(self):
                if not attr.startswith(('__', 'hist_', 'stored_', 'verbose')) and not callable(getattr(self, attr)):
                    print(f'{attr}: {getattr(self, attr)}')
        print('------------------------------------------')
    
    
    def run(self):
        for iter in tqdm(range(1, self.N_t + 1), 'Simulation running...'): # iter 0 is the initial condition so we start from 1 and finish at (N_t + 1) since we want to include the last time step and the range is exclusive of the end point
            #plt.pcolormesh(self.sim_species[0].curt_distrib_fct.T)
            #plt.show()

            if self.computation_method == "NuFi":
                for species_i in self.sim_species:
                    species_i.curt_distrib_fct = species_i.curt_distrib_fct.at[:].set(NuFi(iter, species_i, self))  # Update the distribution function for each species
                self.curt_Efield = self.curt_Efield.at[:].set(Poisson(self)) # Update the current electric field
                self.hist_Efield = self.hist_Efield.at[iter, :].set(self.curt_Efield) # Store the electric field at the current time step
            
            # elif self.params.computation_method == "predcorr":
            #     fs, Efield = predictor_corrector(fs, Efield)
            
            # elif self.params.computation_method == "CMM":
            #     fs, Efield = CMM(fs, Efield)
            
            else:
                raise ValueError(f'Unknown computation method: {self.computation_method}')

            # Add newly computed data to the stored data
            if self.plot or self.export: self.stored_data.update(iter, self.sim_species, self.curt_Efield)
        
        # Export the stored data
        if self.export: self.stored_data.export()
        # Plot the stored data
        if self.plot: self.stored_data.plot()
     
        
    def __build_PhaseSpaceGrid(self, species):
        '''
        Build the grid for the simulation based on the parameters and species.
        '''
        
        x = jnp.arange(self.N_x) * self.L_x / self.N_x # Spatial axis
        v = jnp.linspace(-species.L_v, species.L_v, self.N_v)        # Velocity axis
        
        X, V = jnp.meshgrid(x, v) # Grid of points on the spatial and velocity axes
        return X, V
    
    
    def __build_InitDistribFct(self, species):
        '''
        Build the distribution functions for each species based on the parameters.
        '''
        
        f = jnp.zeros((self.N_x, self.N_v)) # Initial empty distribution function array
        
        f_ini = species.init_distrib_fct        # Access the initial distribution function chosen for the s-th species
        X, V = species.grid[0], species.grid[1] # Select the grid for the s-th species
            
        f = f.at[:, :].set(f_ini(X, V).T) # Set the initial distribution function on the grid
            
        return f