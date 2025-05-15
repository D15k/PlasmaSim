from scipy.io import loadmat
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def load_matlab_data(file_path):
    matlab = loadmat(file_path)
    matlab_data = matlab['data']
    print(matlab_data['time'])
    matlab_fs = matlab_data['fs'][0][0]
    matlab_Efield = matlab_data['Efield'][0][0].T
    matlab_electron = matlab_fs[:,:,:,0].T
    matlab_ion = matlab_fs[:,:,:,1].T
    
    return matlab_Efield, matlab_electron, matlab_ion

def load_python_data(file_path):
    python = loadmat(file_path)
    python_fs = python['distrib_fct']
    python_Efield = python['Efield']
    python_electron = python_fs[:,0,:,:]
    python_ion = python_fs[:,1,:,:]
    
    return python_Efield, python_electron, python_ion

matlab_path = r"C:\Users\Clément\Documents\M2 Physique Fondamentale & Applications - Univers & Particules\Internship\Code_MATLAB\data\ion_accoustic_waves_weak_Tend100_NuFi\config_data.mat"
python_path = r"C:\Users\Clément\Documents\M2 Physique Fondamentale & Applications - Univers & Particules\Internship\plasma-sim_python-conversion\ion_acoustic_wave_2025-05-15_16-17-25\data.mat"

m_E, m_fe, m_fi = load_matlab_data(matlab_path)
p_E, p_fe, p_fi = load_python_data(python_path)


p_E_i = p_E[1:]
p_fe_i = p_fe[1:]
p_fi_i = p_fi[1:]

m_E_i = m_E[:-1]
m_fe_i = m_fe[:-1]
m_fi_i = m_fi[:-1] 

print(m_E_i.shape)
print(p_E_i.shape)
print(m_fe_i.shape)
print(p_fe_i.shape)
print(m_fi_i.shape)
print(p_fi_i.shape)

delta_E = (p_E_i - m_E_i)
delta_fe = (p_fe_i - m_fe_i)
delta_fi = (p_fi_i - m_fi_i)

print(f'Max absolute error in E: {jnp.max(jnp.abs(delta_E))}')
print(f'Max absolute error in fe: {jnp.max(jnp.abs(delta_fe))}')
print(f'Max absolute error in fi: {jnp.max(jnp.abs(delta_fi))}')

print(f'Mean absolute error in E: {jnp.mean(jnp.abs(delta_E))}')
print(f'Mean absolute error in fe: {jnp.mean(jnp.abs(delta_fe))}')
print(f'Mean absolute error in fi: {jnp.mean(jnp.abs(delta_fi))}')

max_error_E_index = np.where(delta_E == np.max(delta_E))
max_error_fe_index = np.where(delta_fe == np.max(delta_fe))
max_error_fi_index = np.where(delta_fi == np.max(delta_fi))

print(max_error_E_index)
print(max_error_fe_index)
print(max_error_fi_index)
