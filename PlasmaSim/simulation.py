import jax.numpy as jnp
from .initialization import Parameters, Species
from .solvePoisson import Poisson
from .solveVlasov import NuFi#, CMM, predictor_corrector


class Simulation:
    '''
    Class performing the simulation of a plasma system using the specified parameters and species.
    
    Attributes:
        params (Parameters): Parameters of the simulation.
        sim_species (Species | list[Species]): Species involved in the simulation.
        
        verbose (bool, optional): Whether to print simulation summary and progress. Defaults to True.
    '''
    
    def __init__(self, params: Parameters, sim_species: Species | list[Species], verbose: bool = True):
        # Compute initial variables that cannot be computed in the Parameters and Species classes
        self.params = params
        self.params.N_s = len(sim_species) # Number of species
        
        self.sim_species = sim_species
        for species_i in self.sim_species:
            species_i.dv = 2*species_i.L_v/(self.params.N_v - 1) # Velocity step
            species_i.grid = self.__build_PhaseSpaceGrid(species_i) # Grid of points on the spatial and velocity axes
            species_i.curt_distrib_fct = self.__build_InitDistribFct(species_i) # Initial distribution function
        
        self.hist_Efield = jnp.zeros((self.params.N_t, self.params.N_x)) # History of the electric field
        self.curt_Efield = Poisson(self.params, self.sim_species) # Initial electric field
        self.hist_Efield = self.hist_Efield.at[0, :].set(self.curt_Efield) # Store the initial electric field
        
        # Print initial simulation summary for debugging purposes
        if verbose:
            print(f'Simulation initialized')
            print(f'Parameters: {self.params}')
            print(f'Species: {self.sim_species}')
    
    
    def run(self):
        for iter in range(1, self.params.N_t + 1): # iter 0 is the initial condition so we start from 1 and finish at (N_t + 1) since we want to include the last time step and the range is exclusive of the end point
            
            if self.params.computation_method == "NuFi":
                self.sim_species = NuFi(iter, self.params, self.sim_species, self.hist_Efield)  # Update the distribution function for each species
                self.curt_Efield = Poisson(self.params, self.sim_species) # Update the current electric field
                self.hist_Efield = self.hist_Efield.at[iter, :].set(self.curt_Efield) # Store the electric field at the current time step
            
            # elif self.params.computation_method == "predcorr":
            #     fs, Efield = predictor_corrector(fs, Efield)
            
            # elif self.params.computation_method == "CMM":
            #     fs, Efield = CMM(fs, Efield)
            
            else:
                raise ValueError(f'Unknown computation method: {self.params.computation_method}')


    def __build_PhaseSpaceGrid(self, species):
        '''
        Build the grid for the simulation based on the parameters and species.
        '''
        
        x = jnp.arange(self.params.N_x) * self.params.L_x / self.params.N_x # Spatial axis
        v = jnp.linspace(-species.L_v, species.L_v, self.params.N_v)        # Velocity axis
        
        X, V = jnp.meshgrid(x, v) # Grid of points on the spatial and velocity axes
        return X, V
    
    
    def __build_InitDistribFct(self, species):
        '''
        Build the distribution functions for each species based on the parameters.
        '''
        
        f = jnp.zeros((self.params.N_x, self.params.N_v)) # Initial empty distribution function array
        
        f_ini = species.init_distrib_fct        # Access the initial distribution function chosen for the s-th species
        X, V = species.grid[0], species.grid[1] # Select the grid for the s-th species
            
        f = f.at[:, :].set(f_ini(X, V).T) # Set the initial distribution function on the grid
            
        return f