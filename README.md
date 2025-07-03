# PlasmaSim

Python package derived from the original Matlab codebase, designed for semi-Lagrangian kinetic simulations. 


## 🟧 Packages requierements

+ JAX (version 0.5.3)
+ tqdm
+ Numpy
+ Scipy
+ Matplotlib

It is also possible to do: ```pip install -r requirements.txt```


## 🟧 Repository content

+ Benchmark :                       Scripts used to perform the error benchmark between Python and Matlab
+ ClusterSpeedTest :                Acceleration test scripts performed on GPU cluster and results
+ LandauDamping_ConvergenceStudy:   Convergence study using Landau damping test case
+ PlasmaSim :                       Package performing the plasma simulation
+ ProfillingTest :                  Script performing profilling on the package using the Landau Ddmping test case
+ ScriptExamples :                  Scripts excamples using the PlasmaSim package on test cases (Landau damping, two-stream instability, ion acoustic wave)      


## 🟧 How to use

1. Import PlasmaSim and JAX (enforce double precision using flag)
2. Define simulation parameters and species of the simulation
3. Initialisie simulation and run it
4. Outputs (following parameters are False by default, if unspecified no outputs):
    + ```export = True``` : Data sampled at frequency save_freq will be saved as .mat file in the chosen specified folder (if not undefined, it will be saved in the same folder as the scipt).
    + ```plot = True``` : Plot the distribution function of each species in phase space as well as the electric field using data sampled at frequency save_freq.

I advise to look at the ScriptExamples folder to understand how to use the package. 


## 🟧 Package Structure

```
PlasmaSim
├── __init.py               # Package init
├── initalisation.py        # Dataclasses to define parameters of the simulation
├── lagrangeInterp.py       # JAX version of Lagrange Interpolation (⚠️ Not used, NuFI still using CubicSpline)
├── simulation.py           # Class performing the simulation and calling functions from other files
├── solvePoisson.py         # Functions solving Poisson equation using Fourier-Galerkin and computing charge density
├── solveVlasov.py          # Functions evolving the distribution functions in time using NuFI 
├── storeData.py            # Class handling outputs by storing data at set frequency and plotting 
└── utils.py                # Utilitary functions for class attributes handling
```