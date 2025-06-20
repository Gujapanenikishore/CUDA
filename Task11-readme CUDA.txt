README - CUDA PARALLEL VECTOR ADDITION (C++17 + CUDA 11.X)

MAIN TASK:
Implement parallel vector addition using CUDA and compare it
against sequential C++ implementation in terms of:
- Performance (execution time)
- Correctness (matching outputs)
- Scalability (small to large vector sizes)

MAIN GOAL:
- Demonstrate how GPU parallelism improves speed for large datasets.
- Ensure the output from both CPU and GPU match within floating-point tolerance.
- Measure speedup (CPU time / GPU time).

HOW IT WORKS:
1. Vectors A and B are initialized with float values.
2. CPU computes A + B using a loop (sequential method).
3. GPU computes A + B in parallel using a CUDA kernel.
4. Time is measured separately for CPU and GPU operations.
5. GPU result is copied back to CPU memory and compared with CPU result.
6. Output is printed showing vector size, time, speedup, and result match status.

HOW TO GET OUTPUT:
- Compile the file:
  nvcc -std=c++17 -arch=sm_60 -O2 vector_add.cu -o vector_add

- Run the executable:
  ./vector_add

- Output includes:
  - Vector size
  - CPU time
  - GPU time
  - Speedup
  - Correctness status
  - Sample additions (only for small test cases)

IMPLEMENTATION:
- Defined a CUDA kernel: vectorAddKernel for parallel addition.
- Implemented CPU logic: vectorAddCPU using std::vector.
- Used chrono to record CPU and GPU execution times.
- Validated GPU output against CPU result.
- Included tests for small, medium, and large vectors.
- Reported speedup and correctness on terminal.

OBSERVATIONS WHEN VECTOR SIZE INCREASES:
- For small vectors (e.g., 8 elements), CPU is faster due to GPU launch overhead.
- For medium vectors (e.g., 10,000 elements), GPU begins to outperform CPU.
- For large vectors (e.g., 1,000,000 elements), GPU is significantly faster.
- The speedup becomes noticeable only when parallel threads can effectively utilize the GPU cores.
- Scalability benefits appear clearly as data size increases beyond thousands of elements.
