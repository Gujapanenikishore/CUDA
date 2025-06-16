===========================================================
TASK 11: CUDA PARALLEL VECTOR ADDITION (C++ with CUDA)
===========================================================

OBJECTIVE:
----------
- Implement vector addition using CUDA (GPU)
- Compare with sequential CPU implementation
- Validate correctness
- Measure performance and speedup

-----------------------------------------------------------
HOW IT WORKS:
-----------------------------------------------------------

GOAL:
Given vectors A and B of size N, compute:
  C[i] = A[i] + B[i] for 0 <= i < N

IMPLEMENTATION:
- CPU version using a simple for loop
- GPU version using CUDA kernel

-----------------------------------------------------------
CUDA KERNEL FUNCTION:
---------------------

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N)

- Executed by multiple threads in parallel on GPU
- Each thread handles one element:
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) → C[i] = A[i] + B[i];

-----------------------------------------------------------
CPU IMPLEMENTATION:
-------------------

void vectorAddCPU(...)
- Uses for loop to compute C[i] = A[i] + B[i]

-----------------------------------------------------------
VALIDATION:
-----------

bool validate(...)
- Compares CPU result and GPU result element-wise
- Returns true if all values match within a small epsilon (1e-5)

-----------------------------------------------------------
MAIN FUNCTION FLOW:
-------------------

1. Define small test vectors A and B of size N=8

2. CPU Calculation:
   - Measure time using std::chrono
   - Store result in C_cpu

3. GPU Calculation:
   - Allocate device memory: cudaMalloc
   - Copy input vectors to device: cudaMemcpy
   - Launch kernel: <<<blocks, threadsPerBlock>>>
   - Copy result back to host
   - Measure time and synchronize

4. Free GPU memory

5. Validate:
   - Compare CPU and GPU results

6. Report:
   - Vector size
   - CPU time
   - GPU time
   - Speedup = CPU time / GPU time
   - Correctness status
   - Print vector additions

-----------------------------------------------------------
SAMPLE OUTPUT:
-----------------------------------------------------------

Vector Size: 8
CPU Time: 1.8e-05 sec
GPU Time: 0.00017 sec
Speedup:   0.106x
Correctness: PASS

Sample Output (Vector Add):
1.5 + 2.5 = 4
3 + -3 = 0
5.2 + 1.8 = 7
7.1 + 0.9 = 8
0.5 + 3.5 = 4
-1 + 1 = 0
2.2 + -2.2 = 0
4.4 + 5.6 = 10

(Note: Exact times and speedup will vary by hardware)

-----------------------------------------------------------
OBSERVATIONS:
-----------------------------------------------------------

- For small N, CPU is often faster (GPU overhead is high)
- For large N (e.g., 1 million), GPU will outperform CPU
- This program is a scalable starting point for larger data

-----------------------------------------------------------
COMPILATION (WITH NVCC):
-------------------------

nvcc -std=c++14 Task11.cu -o vector_add

Run:
  ./vector_add

-----------------------------------------------------------
MODIFICATION OPTIONS:
----------------------

- Change N to a larger number (e.g., 1 million) to test scalability
- Add multiple kernel blocks and threads for better load distribution
- Add timing for memory transfers
- Support double precision (change float to double)

-----------------------------------------------------------
CONCEPTS DEMONSTRATED:
-----------------------

- CUDA programming and memory management
- Parallel execution model: blocks and threads
- CPU vs GPU performance comparison
- Device-host memory transfers (cudaMemcpy)
- Synchronization (cudaDeviceSynchronize)
- Validation and benchmarking

