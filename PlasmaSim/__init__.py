'''
PlasmaSim: A small library for plasma simulation in 1D + 1D phase space.

Provides:
  - Simulator: the main driver class
  - Params: a container for simulation parameters
  - save_config / load_config: io helpers for your “data” dict
'''

__version__ = '0.1.0'

# Package API

from .initialization import Parameters, Species
from .simulation import Simulation
from .solvePoisson import Poisson
from .solveVlasov import NuFi#, CMM, predictor_corrector
#from .io    import save_config, load_config
#from .plot  import plot_results


__all__ = [
  'Simulation',
  'Parameters',
  'Species',
  'Poisson',
  'NuFi'
]
