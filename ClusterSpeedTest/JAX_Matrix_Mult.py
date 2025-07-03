import jax
import jax.numpy as jnp
import numpy as np
from time import perf_counter
from scipy import stats

sizes = [2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15]

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
    print(f"\n Testing matrix size {N}x{N}... \n")
    
    A = jax.random.normal(key, (N, N))
    B = jax.random.normal(key, (N, N))

    # CPU
    A_cpu = jax.device_put(A, device=cpu_device)
    B_cpu = jax.device_put(B, device=cpu_device)
    jax.block_until_ready(jnp.dot(A_cpu, B_cpu))  # warm-up
    start = perf_counter()
    jax.block_until_ready(jnp.dot(A_cpu, B_cpu)) 
    cpu_time = perf_counter() - start
    cpu_times.append(cpu_time)

    # GPU (if available)
    if has_gpu:
        A_gpu = jax.device_put(A, device=gpu_device)
        B_gpu = jax.device_put(B, device=gpu_device)
        jax.block_until_ready(jnp.dot(A_gpu, B_gpu))  # warm-up
        start = perf_counter()
        jax.block_until_ready(jnp.dot(A_gpu, B_gpu))
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
print(f"{'Matrix Size':<12} {'CPU Time (s)':<15} {'GPU Time (s)':<15} {'Speedup':<10}")
print("-" * 52)

for i, N in enumerate(sizes):
    if has_gpu and i < len(gpu_times):
        speedup = cpu_times[i] / gpu_times[i]
        print(f"{N}x{N:<7} {cpu_times[i]:<15.6f} {gpu_times[i]:<15.6f} {speedup:<10.2f}x")
    else:
        print(f"{N}x{N:<7} {cpu_times[i]:<15.6f} {'N/A':<15} {'N/A':<10}")

# Regression Analysis
print(f"\nCOMPLEXITY ANALYSIS (Log-Log Regression)")
print("-" * 40)
print(f"Theoretical matrix multiplication complexity: O(N³) ≈ 3.0")
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