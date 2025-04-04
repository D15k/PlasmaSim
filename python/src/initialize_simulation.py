import numpy as np
import os
import grid.make_periodic_grid as make_periodic_grid

def initialize_simulation(params):
    # Initialize grids and distribution functions
    params['grids'] = []
    for s in range(params['Ns']):
        grid = make_periodic_grid(params['Lx'], params['Lv_s'][s], params['Nx'], params['Nv'])
        grid['method'] = "spline"
        params['grids'].append(grid)

    # Initialize distribution functions
    fs = np.zeros((params['Nx'], params['Nv'], params['Ns']))
    for s in range(params['Ns']):
        fini = params['fini'][s]
        fs[:, :, s] = fini(params['grids'][s]['X'], params['grids'][s]['V'])

    # Maximal Iteration number:
    # check if maximal time iteration number Nt_max fits final time Tend
    if params['Nt_max'] > params['Tend'] / params['dt']:
        params['Nt_max'] = int(np.ceil(params['Tend'] / params['dt']))

    # output data storage allocation
    data = {}
    if 'dt_save' in params:
        dt_save = params['dt_save']  # how often do we want to save solution?
        dit_save = dt_save / params['dt']
        params['dit_save'] = dit_save
        if dit_save % 1 != 0 or dt_save < params['dt']:
            # dit_save is not an integer
            # therefore dt is not a divisor of dt_save
            raise ValueError('Houston we have a problem: dt_save is not correct.')
        
        # allocate
        Nsamples = int(params['Nt_max'] / dit_save)
        Nsize = list(params['grids'][0]['size']) + [Nsamples, params['Ns']]
        data['fs'] = np.zeros(Nsize)
        data['Efield'] = np.zeros((params['grids'][0]['size'][0], Nsamples))
        data['time'] = dt_save * np.arange(1, Nsamples + 1)
    else:
        # never save anything
        params['dit_save'] = params['Nt_max'] + 2
        data = None

    # Directories
    if 'data_dir' not in params:
        root = params.get('root_dir', "./")
        params['data_dir'] = os.path.join(root, "data", f"{params['mycase']}_{params['method']}")
        if not os.path.exists(params['data_dir']):
            os.makedirs(params['data_dir'])

    return params, fs, data