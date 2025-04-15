#########################################################################
#                     Characteristic Mapping Method                     #
#                              Main script                              #
#########################################################################

import src.DEFAULTS  # Matplotlib parameters and Directories definition
import src.Sim
import time

from src.Sim import *


## Select case: 

#import params.PARAMS_landau_damping as params
import params.PARAMS_ion_acoustic_waves as params


## Simulation: 

# Measure execution time for "NuFi"
start_time = time.time()
params.method = 'NuFi'
time.sleep(2)
params, data = Sim(params)
print(f'CPU execution time for "Nufi" method: {time.time() - start_time:.2f}')

'''
# Measure execution time for "predcorr"
start_time = time.time()
params.method = 'predcorr'
params, data = Sim(params)
print(f'CPU execution time for "predcorr" method: {time.time() - start_time:.2f}')
'''

## Plotting:

# TO DO !!!!