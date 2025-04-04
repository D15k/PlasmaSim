import numpy as np
import matplotlib.pyplot as plt

import initialize_simulation

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
    params.Efield = vPoisson(fs, params["grids"], params["charge"])
    params["Efield_list"] = [params["Efield"]]  # Store first Efield

    # Plot initial conditions
    plot_results(params, fs)

    # Main time loop
    Nsamples = 0
    for iT in range(1, params["Nt_max"] + 1):
        params["it"] = iT  # Store iteration count

        # Perform a single time step
        fs, params = step(params, fs)

        # Increase time
        time = params["dt"] * iT

        # Measurements
        params = measure(params, fs)

        # Plot results
        plot_results(params, fs)

        # Save configuration at specific times
        if iT % params["dit_save"] == 0:
            Nsamples += 1
            data = save_config(params, data, fs, Nsamples)

        # Stop if simulation end time is reached
        if time >= params["Tend"]:
            break

    return params, data


def step(params, fs):
    """
    Advances the simulation by one time step.
    """
    method = params["method"]
    
    if method == "predcorr":
        fs, params = predictor_corrector(params, fs)
    elif method == "NuFi":
        fs, params = NuFi(params, fs)
    elif method == "CMM":
        fs, params = CMM(params, fs)
    else:
        print("Error: Unknown step method")

    return fs, params


def CMM(params, fs):
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
    """
    Plots the results of the simulation.
    """
    fig, axes = plt.subplots(params["Ns"] + 1, 1, figsize=(8, 6))

    for s in range(params["Ns"]):
        ax = axes[s]
        im = ax.pcolormesh(params["grids"][s]["X"], params["grids"][s]["V"], fs[:, :, s], shading='auto')
        ax.set_title(f"$f_\\mathrm{{{params['species_name'][s]}}}$")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$v$")

    ax = axes[-1]
    ax.plot(params["grids"][0]["x"], params["Efield"])
    ax.set_title("$E$")
    ax.set_xlabel("$x$")
    fig.colorbar(im, ax=ax)

    plt.pause(0.01)  # Pause for visualization
    plt.show(block=False)  # Non-blocking display

