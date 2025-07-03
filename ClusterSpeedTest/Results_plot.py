import matplotlib.pyplot as plt

# Updated data
sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912]
cpu_times = [162.090593, 138.899924, 151.862300, 137.189651, 133.399611, 158.506708, 128.501979, 132.210485, 145.066545, 122.164405, 138.234741, 124.919225, 137.690100, 140.186539, 111.658818, 136.645519, 129.903533, 152.407417, 166.357443, 127.459723, 147.298840, 161.644935, 168.493283]
gpu_times = [0.088656, 0.092368, 0.095029, 0.083844, 0.087724, 0.081465, 0.087704, 0.087618, 0.081530, 0.083335, 0.087658, 0.087354, 0.076154, 0.086010, 0.087780, 0.087723, 0.087760, 0.081972, 0.095743, 0.085818, 0.081699, 0.088225, 0.088270]
speedup = [1828.31, 1503.76, 1598.07, 1636.25, 1520.68, 1945.69, 1465.18, 1508.93, 1779.31, 1465.94, 1576.98, 1430.03, 1808.05, 1629.89, 1272.04, 1557.69, 1480.22, 1859.27, 1737.55, 1485.23, 1802.95, 1832.20, 1908.85]

# Plotting time vs size
plt.figure(figsize=(8, 6))
plt.plot(sizes, cpu_times, marker='o', label='CPU JAX Time(s)')
plt.plot(sizes, gpu_times, marker='o', label='GPU JAX Time(s)')
#plt.plot(sizes, matlab_times, marker='o', label='Matlab CPU Time(s)')
plt.xlabel('Size (N)')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time vs Size')
plt.grid(True)
plt.legend()
plt.xscale('log', base=2)
plt.yscale('log')
plt.tight_layout()
plt.show()

# Plotting speedup vs size
plt.figure(figsize=(8, 6))
plt.plot(sizes, speedup, marker='o', label='Speedup (CPU / GPU)')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Speedup Factor')
plt.title('GPU Speedup vs Matrix Size')
plt.grid(True)
plt.legend()
plt.xscale('log', base=2)
plt.tight_layout()
plt.show()