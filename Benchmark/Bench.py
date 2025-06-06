from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def load_matlab_data(file_path):
    matlab = loadmat(file_path)
    matlab_data = matlab['data']
    #print(matlab_data['time'])
    matlab_fs = matlab_data['fs'][0][0]
    Efield = matlab_data['Efield'][0][0].T
    
    if len(matlab_fs.shape) == 4:
        distrib_fcts = np.transpose(matlab_fs, (2,3,1,0))
    elif len(matlab_fs.shape) == 3:
        distrib_fcts = matlab_fs.T
        distrib_fcts = np.expand_dims(distrib_fcts, axis=1)
    else:
        raise ValueError(f"Invalid shape for distrib_fcts: {matlab_fs.shape}")
    
    return Efield, distrib_fcts


def load_python_data(file_path):
    python = loadmat(file_path)
    python_fs = python['distrib_fct']
    Efield = python['Efield']
    distrib_fcts = python_fs[:,:,:,:]
        
    return Efield, distrib_fcts


def compare_data(data1_filepath, data2_filepath):
    E_1, f_1 = load_matlab_data(data1_filepath)
    E_2, f_2 = load_python_data(data2_filepath)
    
    E_2 = E_2[1:]
    f_2 = f_2[1:]
    
    if E_1.shape != E_2.shape: raise ValueError(f"Efield shapes do not match: Matlab: {E_1.shape} != Python: {E_2.shape}")
    if f_1.shape != f_2.shape: raise ValueError(f"Distrib_fcts shapes do not match: Matlab: {f_1.shape} != Python: {f_2.shape}")
    
    print(jnp.abs(E_1 - E_2).shape)
    print(jnp.abs(f_1 - f_2).shape)
    
    max_abs_error_E = jnp.max(jnp.abs(E_1 - E_2), axis=1)
    max_abs_error_f = jnp.max(jnp.abs(f_1 - f_2), axis=(2,3))
    
    print(f'Max absolute error in E: {jnp.max(max_abs_error_E)}')
    print(f'Max absolute error in f: {jnp.max(max_abs_error_f, axis=0)}')
    
    label_f = ['e', 'i']
    
    S = 12
    plt.rcParams['font.size'] = S
    plt.rcParams['axes.labelsize'] = S
    plt.rcParams['xtick.labelsize'] = S
    plt.rcParams['ytick.labelsize'] = S
    plt.plot(max_abs_error_E, label='$E$')
    for i in range(max_abs_error_f.shape[1]):
        plt.plot(max_abs_error_f[:,i], label=f'$f_{{\\mathrm{{{label_f[i]}}}}}$')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('$L^\infty-norm$')
    plt.tight_layout()
    plt.legend()
    plt.show()


main_path = r'C:\Users\Clément\Documents\M2 Physique Fondamentale & Applications - Univers & Particules\Internship\plasma-sim_python-conversion\Benchmark'

#path = main_path + r'\ion_acoustic_waves'
#path = main_path + r'\landau_damping'
path = main_path + r'\two_streams'

python_path = path + r'\data_python.mat'
matlab_path = path + r'\data_matlab.mat'

compare_data(matlab_path, python_path)