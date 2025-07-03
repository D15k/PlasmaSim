import jax
import jax.numpy as jnp
import numpy as np
from time import perf_counter
from scipy import stats

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['order'])
def JAX_lagrange_local_interp_periodic(x_target, xgrid, ygrid, order):
    """
    Interpolates y = f(x) at x_target using local Lagrange interpolation.
    xgrid: equispaced periodic grid (assumed 1D)
    ygrid: function values at xgrid
    order: number of points used in local stencil (e.g., 4 for cubic)
    x_target: array of target points in [0,1)
    """
    N = len(xgrid)
    delta_x = xgrid[1] - xgrid[0]  # uniform spacing assumed
    f_interp = jnp.zeros_like(x_target)
    half = order // 2
    x_local = jnp.arange(-half, order - half + 1)  # -half to half+order inclusive

    idx0 = x_target / delta_x
    j0 = jnp.floor(idx0).astype(jnp.int64)
    delta_idx = idx0.flatten() - j0.flatten()    
    
    # Local grid and values with periodic wrap-around
    idx_list = (j0[:,None] - half + jnp.arange(order + 1)) % N
    y_local = ygrid[idx_list]

    
    # Evaluate local Lagrange interpolant
    Ljs  = JAXlagrange_basis(delta_idx, x_local, order)
    p = jnp.sum(y_local.T * Ljs, axis = 0)

    f_interp = p

    return f_interp


def JAXlagrange_basis(x: jnp.array, x_nodes: jnp.array, order: int):
    """
    Computes the j-th Lagrange basis polynomial evaluated at x.
    """
    x = jnp.array(x)
    x_nodes = jnp.array(x_nodes)
     
    xj = x_nodes[:order + 1]
    
    tempA = (x[:, None] - x_nodes)
    tempB = (xj[:, None] - x_nodes + 1e-32)
       
    temp = jnp.array(tempA[ :, None] / tempB) # epsilon to prevent division by zero
    temp = jnp.transpose(temp, axes = (1,2,0))
    
    #mask = jnp.eye(order + 1, dtype=bool)
    mask = jnp.broadcast_to(jnp.eye(order + 1), (len(x), order + 1, order + 1)).T
    A = jnp.where(mask.astype(bool), mask, temp)

    Lj = jnp.prod(A, axis=1)
    
    return Lj

# ==================================================================================================================

sizes = [2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20, 2**21, 2**22, 2**23, 2**24, 2**25, 2**26, 2**27, 2**28, 2**29]

order = 4

# Reference solution: very fine grid
N_ref = jnp.max(jnp.array(sizes))
x_ref = (jnp.arange(N_ref) + 0.5) / N_ref

f = lambda x: jnp.sin(2*jnp.pi*x) * jnp.cos(6*jnp.pi*x) + 1


cpu_times = []
gpu_times = []

# Check if GPU is available
try:
    gpu_device = jax.devices("gpu")[0]
    has_gpu = True
    print(f"GPU detected: {gpu_device}")
except:
    has_gpu = False
    gpu_device_name = "None"

cpu_device = jax.devices("cpu")[0]
cpu_device_name = str(cpu_device)

key = jax.random.PRNGKey(0)

# Run performance tests
for N in sizes:
    print(f"\n Testing size {N}... \n")
    
    x_grid = jnp.arange(N) / N
    y_grid = f(x_grid)

    # CPU
    x_ref_cpu = jax.device_put(x_ref, device=cpu_device)
    x_grid_cpu = jax.device_put(x_grid, device=cpu_device)
    y_grid_cpu = jax.device_put(y_grid, device=cpu_device)
    jax.block_until_ready(JAX_lagrange_local_interp_periodic(x_ref_cpu, x_grid_cpu, y_grid_cpu, order))  # warm-up
    start = perf_counter()
    jax.block_until_ready(JAX_lagrange_local_interp_periodic(x_ref_cpu, x_grid_cpu, y_grid_cpu, order))
    cpu_time = perf_counter() - start
    cpu_times.append(cpu_time)

    # GPU (if available)
    if has_gpu:
        x_ref_gpu = jax.device_put(x_ref, device=gpu_device)
        x_grid_gpu = jax.device_put(x_grid, device=gpu_device)
        y_grid_gpu = jax.device_put(y_grid, device=gpu_device)
        jax.block_until_ready(JAX_lagrange_local_interp_periodic(x_ref_gpu, x_grid_gpu, y_grid_gpu, order))  # warm-up
        start = perf_counter()
        jax.block_until_ready(JAX_lagrange_local_interp_periodic(x_ref_gpu, x_grid_gpu, y_grid_gpu, order))
        gpu_time = perf_counter() - start
        gpu_times.append(gpu_time)

# Linear Regression Analysis
log_sizes = np.log10(sizes)
log_cpu_times = np.log10(cpu_times)

# CPU time regression
cpu_slope, cpu_intercept, cpu_r_value, cpu_p_value, cpu_std_err = stats.linregress(log_sizes, log_cpu_times)

# GPU and speedup analysis (if GPU available)
if has_gpu and gpu_times:
    log_gpu_times = np.log10(gpu_times)
    speedups = [c / g for c, g in zip(cpu_times, gpu_times)]
    
    # GPU time regression
    gpu_slope, gpu_intercept, gpu_r_value, gpu_p_value, gpu_std_err = stats.linregress(log_sizes, log_gpu_times)
    
    # Speedup regression
    speedup_slope, speedup_intercept, speedup_r_value, speedup_p_value, speedup_std_err = stats.linregress(log_sizes, np.log10(speedups))

# ============================================================================
# COMPREHENSIVE PERFORMANCE SUMMARY
# ============================================================================

print("=" * 80)
print("JAX MATRIX MULTIPLICATION PERFORMANCE BENCHMARK")
print("=" * 80)

# Device Information
print("\nDEVICE INFORMATION")
print("-" * 40)
print(f"CPU Device: {cpu_device_name}")
if has_gpu:
    print(f"GPU Device: {str(gpu_device)}")
    print(f"GPU Available: Yes")
else:
    print(f"GPU Device: None")
    print(f"GPU Available: No")

# Performance Results
print(f"\nPERFORMANCE RESULTS")
print("-" * 40)
print(f"{'Size':<12} {'CPU Time (s)':<15} {'GPU Time (s)':<15} {'Speedup':<10}")
print("-" * 52)

for i, N in enumerate(sizes):
    if has_gpu and i < len(gpu_times):
        speedup = cpu_times[i] / gpu_times[i]
        print(f"{N:<7} {cpu_times[i]:<15.6f} {gpu_times[i]:<15.6f} {speedup:<10.2f}x")
    else:
        print(f"{N:<7} {cpu_times[i]:<15.6f} {'N/A':<15} {'N/A':<10}")

# Regression Analysis
print(f"\nCOMPLEXITY ANALYSIS (Log-Log Regression)")
print("-" * 40)
#print(f"Theoretical matrix multiplication complexity: O(N³) ≈ 3.0")
print()
print(f"CPU Performance:")
print(f"  • Scaling: O(N^{cpu_slope:.3f})")
print(f"  • R² value: {cpu_r_value**2:.4f}")
print(f"  • P-value: {cpu_p_value:.2e}")

if has_gpu and gpu_times:
    print(f"\nGPU Performance:")
    print(f"  • Scaling: O(N^{gpu_slope:.3f})")
    print(f"  • R² value: {gpu_r_value**2:.4f}")
    print(f"  • P-value: {gpu_p_value:.2e}")
    
    print(f"\nSpeedup Analysis:")
    print(f"  • Scaling: O(N^{speedup_slope:.3f})")
    print(f"  • R² value: {speedup_r_value**2:.4f}")
    print(f"  • P-value: {speedup_p_value:.2e}")
    if speedup_slope > 0:
        print(f"  • Trend: Speedup increases with matrix size")
    else:
        print(f"  • Trend: Speedup decreases with matrix size")

# Summary Statistics
if has_gpu and gpu_times:
    avg_speedup = np.mean(speedups)
    max_speedup = max(speedups)
    min_speedup = min(speedups)
    
    print(f"\nSPEEDUP STATISTICS")
    print("-" * 40)
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Maximum Speedup: {max_speedup:.2f}x (at {sizes[speedups.index(max_speedup)]}x{sizes[speedups.index(max_speedup)]})")
    print(f"Minimum Speedup: {min_speedup:.2f}x (at {sizes[speedups.index(min_speedup)]}x{sizes[speedups.index(min_speedup)]})")

print(f"\nTest completed successfully!")
print("=" * 80)