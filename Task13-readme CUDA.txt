============================================================
TASK 13: PARALLEL PRIME NUMBER FINDER (CUDA Version)
============================================================

DESCRIPTION:
------------
This program efficiently finds all prime numbers up to a given limit (default: 1,000,000) using both:
- A sequential CPU implementation for baseline comparison
- A massively parallel GPU implementation using CUDA

The CUDA version parallelizes the primality check using thousands of threads. It uses `atomicAdd` to safely store prime numbers in shared GPU memory, then transfers results back to CPU memory for validation and benchmarking.

The output clearly displays:
- Number of primes found by each method
- Execution time of CPU vs GPU
- Speedup factor of GPU over CPU

------------------------------------------------------------
REQUIREMENTS:
-------------
- C++17 compatible compiler
- NVIDIA GPU with CUDA Compute Capability 5.0 or higher
  (Tested on: NVIDIA GTX TITAN X)
- CUDA Toolkit 11.x or newer (Tested on CUDA 12.8)

------------------------------------------------------------
FILE:
-----
prime_finder.cu

------------------------------------------------------------
HOW TO COMPILE:
---------------
Using NVCC (NVIDIA CUDA Compiler):

    nvcc -std=c++17 prime_finder.cu -o prime_finder

------------------------------------------------------------
HOW TO RUN:
-----------
    ./prime_finder

Output will include:
- Total primes found by CPU
- Total primes found by GPU
- Time taken for each
- Speedup over CPU

------------------------------------------------------------
MEMORY MANAGEMENT:
------------------
- CPU version uses standard C++ vectors to store primes
- GPU version:
    * Allocates device memory using `cudaMalloc` for:
        - Output array (`int* d_output`)
        - Counter (`int* d_count`)
    * Initializes counter with `cudaMemset`
    * Uses `atomicAdd` in the kernel to update prime count
    * Copies result count and primes from GPU to CPU with `cudaMemcpy`
    * All device memory is freed at the end with `cudaFree`
    * Host-side memory is cleaned using `delete[]`

------------------------------------------------------------
SAMPLE OUTPUT:
--------------
=== Prime Finder up to LIMIT = 1000000 ===

[CPU]   Found 78498 primes in 1.324 sec  
[CUDA]  Found 78498 primes in 0.042 sec  
Speedup over CPU: 31.52x  

------------------------------------------------------------
NOTES:
------
- You can change the upper limit of prime search by modifying the `LIMIT` constant in the code.
- CUDA version achieves better performance as the input size increases.
- Ensure your system has a compatible NVIDIA GPU and driver installed.

------------------------------------------------------------

