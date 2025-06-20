
**TASK 12 - CUDA PARALLEL MATRIX TRANSPOSE**

**MAIN GOAL**
Implement and compare Matrix Transpose using:

* CUDA (GPU)
* C++ (CPU)

**REQUIREMENTS**

* C++17
* CUDA 11.x or newer
* NVIDIA GPU with CUDA support

**HOW IT WORKS**

1. A 5x5 matrix is created with values from 1 to 25.
2. Transposition is performed:

   * On CPU using nested loops
   * On GPU using CUDA kernel with grid and block structure
3. Time is measured for both CPU and GPU implementations.
4. Results are printed and validated for correctness.

**MEMORY MANAGEMENT**

* Host memory is allocated using C++ `std::vector`.
* Device memory is allocated using `cudaMalloc`.
* Data is copied from host to device using `cudaMemcpy`.
* After GPU execution, results are copied back to host.
* Device memory is freed using `cudaFree` to avoid memory leaks.

**OUTPUT**

* Original Matrix
* CPU Transposed Matrix
* GPU Transposed Matrix
* Validation: PASS / FAIL
* CPU Transposed Matrix Time (ms)
* GPU Transposed Matrix Time (ms)
* Speedup (CPU time / GPU time)

**SCALABILITY NOTE**
As matrix size increases (e.g., 10x10, 100x100), GPU performance improves significantly due to parallelism.

