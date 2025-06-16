===============================================================
C++ & CUDA PROJECT README - Tasks 11 to 14
===============================================================

Author: [Your Name]  
Languages Used: Modern C++17 and CUDA (NVIDIA GPU Required)  
Compilers: g++ and nvcc

---------------------------------------------------------------
SUMMARY
---------------------------------------------------------------
This section covers four GPU-accelerated or parallelized C++/CUDA tasks involving:
- Vector Addition
- Matrix Transposition
- Prime Number Finding
- Histogram Computation

Each task includes:
- A sequential or CPU multithreaded version
- A CUDA version (if ENABLE_CUDA = 1)
- Performance comparison
- Validation for correctness

Tasks Included:
- Task 11: CUDA Parallel Vector Addition
- Task 12: CUDA Parallel Matrix Transpose
- Task 13: Parallel Prime Number Finder (CUDA/Multithreaded)
- Task 14: Histogram Computation (CUDA/Multithreaded)

---------------------------------------------------------------
TASK 11: CUDA PARALLEL VECTOR ADDITION
---------------------------------------------------------------
Objective:
- Compute C = A + B using CUDA
- Compare results and speed with CPU version

Key Points:
- Uses float vectors (length = 8 in test case)
- GPU uses grid-stride indexing
- Validation: compares GPU and CPU outputs

Performance Sample:
CPU Time: ~1.8e-05 sec  
GPU Time: ~0.00017 sec  
Speedup: < 1x (for small size)  
Validation: PASS

---------------------------------------------------------------
TASK 12: CUDA PARALLEL MATRIX TRANSPOSE
---------------------------------------------------------------
Objective:
- Transpose a 2D matrix using a CUDA kernel

Key Points:
- Input: 4x4 matrix
- Kernel uses 2D blocks and grid indexing
- Compared against CPU transpose function

Performance:
- Both CPU and GPU produce identical results
- Validation: PASS

---------------------------------------------------------------
TASK 13: PARALLEL PRIME NUMBER FINDER
---------------------------------------------------------------
Objective:
- Find all prime numbers up to 1,000,000
- Compare:
  - Sequential C++
  - Multithreaded C++
  - CUDA kernel

Key Points:
- Each version uses a basic isPrime() function
- CUDA kernel uses atomicAdd to write results
- Result count and values are validated

Performance Sample:
Sequential: ~1.3 sec  
Multithreaded: ~0.35 sec  
CUDA: ~0.04 sec  
Validation: PASS

---------------------------------------------------------------
TASK 14: HISTOGRAM COMPUTATION (MULTITHREADED / CUDA)
---------------------------------------------------------------
Objective:
- Compute a 256-bin histogram from a dataset of 16 million bytes

Versions:
- Sequential CPU
- Multithreaded CPU (per-thread histograms merged)
- CUDA Kernel using atomicAdd()

Key Points:
- Validates that all versions match
- Uses std::chrono for performance benchmarking

Performance Sample:
Sequential Time: ~0.18 sec  
Multithreaded Time: ~0.11 sec  
CUDA Time: ~0.03 - 0.06 sec (if enabled)  
Validation: PASS

---------------------------------------------------------------
BUILD AND RUN INSTRUCTIONS
---------------------------------------------------------------

To compile CPU-only versions:
  g++ -std=c++17 Task11.cpp -o task11
  ./task11

For multithreaded code:
  g++ -std=c++17 Task13.cpp -o task13 -pthread

To compile CUDA versions:
  nvcc -std=c++14 Task11.cu -o task11_cuda

To enable CUDA mode (for Tasks 13, 14):
  Set: #define ENABLE_CUDA 1
  Then compile with:
    nvcc -std=c++14 Task14.cu -o task14_cuda

---------------------------------------------------------------
NOTES
---------------------------------------------------------------
- CUDA kernels use grid-stride loops and atomic operations
- CPU and GPU results are rigorously validated
- Suitable for NVIDIA GPUs with compute capability 3.5+

---------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------
- C++17 compatible compiler (g++, clang++)
- CUDA Toolkit (nvcc) and NVIDIA GPU for GPU tasks
- Linux or Windows with WSL/MinGW (for CUDA)

---------------------------------------------------------------
CREDITS
---------------------------------------------------------------
Developed as part of a high-performance computing assignment.

