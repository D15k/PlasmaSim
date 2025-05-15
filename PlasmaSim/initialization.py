import jax
jax.config.update("jax_enable_x64", True)
from typing import Callable
from dataclasses import dataclass, field

### Classes to help users initialize the simulation ###
    
@dataclass
class Parameters:
    '''
    Class containing the parameters of the simulation.
    
    Attributes:
        computation_method (str): The method used for computation in the simulation.
        
        N_t (int): Number of time steps.
        L_t (float): Length of the time domain.
        N_x (int): Number of spatial grid points.
        L_x (float): Length of the spatial domain.
        N_v (int): Number of velocity grid points.
        
        export (bool, optional): Whether to export data. Defaults to False.
        plot (bool, optional): Whether to plot data. Defaults to False.
        save_freq (int, optional): Number of time steps between each data recording. Used for exporting data and plotting. Defaults to 1, saves data after every time step.
        name (str, optional): Custom name of the simulation. Defaults to 'PlasmaSim'. Used for saving results.
    ''' 
    
    computation_method: str
    
    N_t: int
    L_t: float
    N_x: int
    L_x: float
    N_v: int

    export: bool = False
    plot: bool = False
    save_freq: int = 1
    name: str = 'PlasmaSim'
    save_dir: str = None
    
    dt: float = field(init = False)
    dx: float = field(init = False)
    
    
    def __post_init__(self):
         
        self.dt = self.L_t / self.N_t     
        self.dx = self.L_x / self.N_x
        

@dataclass
class Species:
    '''
    Class containing the parameters of a species.
    
    Attributes:
        species_name (str): The name of the species.
        mass (int): The mass of the species.
        charge (int): The charge of the species.
        init_distrib_fct (Callable): The initial distribution function of the species.
        L_v (float): Length of the velocity domain for the species.     #TODO: Apparently, renormalisezd VP eqs allow to have the same L_v for all species.
    '''    

    species_name: str
    
    mass: int
    charge: int
    init_distrib_fct: Callable
    
    L_v: float