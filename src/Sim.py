import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = False # Disable LaTeX rendering to avoid errors if LaTeX is not installed
import sys
from pathlib import Path

params_path = Path(__file__).resolve().parent.parent
print(params_path)
sys.path.append(str(params_path))

import params.PARAMS_ion_acoustic_waves as params

from src.initialize_simulation import *
from src.vPoisson import *
from src.measure import *
from src.NuFi import *

def Sim(params):
    """
    Runs the plasma simulation.
    
    Args:
        params (dict): Dictionary containing simulation parameters.
    
    Returns:
        params (dict): Updated parameters.
        data (dict): Simulation results.
    """

    # Initialize grids, distribution functions and output data
    params, fs, data = initialize_simulation(params)

    # Compute initial electric field and store it
    params.Efield = vPoisson(fs, params.grids, params.charge)
    params.Efield_list = [params.Efield]  # Store first Efield

    # Plot initial conditions
    # plot_results(params, fs)       #TODO: Will plot everything after the simulation

    # Main time loop
    Nsamples = 0
    for iT in range(1, params.Nt_max + 1):
        params.it = iT  # Store iteration count

        # Perform a single time step
        fs, params = step(params, fs)

        # Increase time
        time = params.dt * iT
        print(f'Time: {time:.2f}')

        # Measurements
        params = measure(params, fs)

        # Plot results
        plot_results(params, fs)   #TODO: Will plot everything after the simulation

        """# Save configuration at specific times
        if iT % params.dit_save == 0:
            Nsamples += 1
            data = save_config(params, data, fs, Nsamples)"""  ##TODO: Convert save_config function 

        # Stop if simulation end time is reached
        if time >= params.Tend:
            break

    return params, data


def step(params, fs):
    """
    Advances the simulation by one time step.
    """
    method = params.method
    
    if method == "predcorr":
        fs, params = predictor_corrector(params, fs)
    elif method == "NuFi":
        fs, params = NuFi(params, fs)
    elif method == "CMM":
        fs, params = CMM(params, fs)
    else:
        print("Error: Unknown step method")

    return fs, params


def CMM(params, fs):   #TODO: Convert CMM function
    """
    Runs the CMM (Conservative Mapping Method) step.
    """
    iT = params["it"] + 1
    dt = params["dt"]

    for s in range(params["Ns"]):
        X, V = sympl_flow_Half(
            iT, dt, params["grids"][s]["X"], params["grids"][s]["V"],
            params["charge"][s] / params["Mass"][s] * params["Efield_list"], 
            params["grids"][s]
        )
        fini = params["fini"][s]
        fs[:, :, s] = fini(X, V)

    params["Efield"] = vPoisson(fs, params["grids"], params["charge"])
    params["Efield_list"].append(params["Efield"])

    return fs, params


def plot_results(params, fs):
    Ns = params.Ns
    grids = params.grids
    Efield = params.Efield

    plt.clf()  # Clear previous figure

    for s in range(Ns):
        plt.subplot(Ns + 1, 1, s + 1)
        
        X = grids[s]['X']
        V = grids[s]['V']
        F = fs[:, :, s].T  # Transpose to match (V, X) shape

        plt.pcolormesh(X, V, F, shading='auto')
        plt.title(rf"$f_\mathrm{{{params.species_name[s]}}}$")
        plt.colorbar()
        plt.xlabel("$x$")
        plt.ylabel("$v$")

    # Plot E-field
    plt.subplot(Ns + 1, 1, Ns + 1)
    x = grids[0]['x']
    plt.plot(x, Efield)
    plt.xlim([x[0], x[-1]])
    plt.title("$E$")
    plt.xlabel("$x$")

    plt.tight_layout()
    plt.pause(0.25)