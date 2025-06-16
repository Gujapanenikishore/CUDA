==========================================================
TASK 14: HISTOGRAM COMPUTATION (CPU / THREAD / CUDA)
==========================================================

OBJECTIVE:
----------
- Build a program that computes a histogram from a large dataset.
- Support 3 versions:
  1. Sequential CPU
  2. Multithreaded CPU
  3. CUDA GPU (optional with ENABLE_CUDA = 1)

- Validate results across all modes.
- Measure and compare performance.

----------------------------------------------------------
WHAT IS A HISTOGRAM?
---------------------
A histogram counts how many times each value appears in a dataset.

Here:
- Input = vector of 8-bit integers (0 to 255)
- Output = 256-bin histogram: hist[i] = count of i in dataset

----------------------------------------------------------
DATASET:
---------
- Size = 2^24 = 16,777,216 entries (16 million)
- Random values from [0, 255] using std::uniform_int_distribution

----------------------------------------------------------
VERSION 1: SEQUENTIAL HISTOGRAM
--------------------------------
Function: sequentialHistogram()

Logic:
- Loop through data
- For each value `val`, increment hist[val]

Performance:
- Simple and reliable
- Slowest for large datasets

Time Example:
[Sequential Version]
Time: 0.183 sec

----------------------------------------------------------
VERSION 2: MULTITHREADED HISTOGRAM
-----------------------------------
Function: histogramThread()

Logic:
- Data is split into N parts (N = number of CPU cores)
- Each thread builds its own local histogram
- After all threads complete, local histograms are merged

Benefits:
- Improves performance using all CPU cores
- No locking overhead (thread-safe local histograms)

Time Example:
[Multithreaded Version]
Time: 0.109 sec
Validation: YES (matches sequential)

----------------------------------------------------------
VERSION 3: CUDA GPU HISTOGRAM (Optional)
-----------------------------------------
(Enabled by setting ENABLE_CUDA = 1 and using nvcc)

Kernel: histogramKernel<<<blocks, threads>>>

Logic:
- Each thread processes one data point
- Uses atomicAdd() to update shared histogram safely

Setup:
- Allocate GPU memory (cudaMalloc)
- Copy input data to device (cudaMemcpy)
- Zero-initialize histogram (cudaMemset)
- Run kernel and sync (cudaDeviceSynchronize)
- Copy histogram back to host

Benefit:
- Very fast for large datasets

----------------------------------------------------------
VALIDATION:
------------
For each version:
- Print sample bins [0–9]
- Compare results to sequential version
- Display "Match: YES" if all bins are equal

----------------------------------------------------------
SAMPLE OUTPUT:
---------------

[Sequential Version]
Time: 0.183 sec
Sample bins: [0]=65745 [1]=65250 ...

[Multithreaded Version]
Time: 0.109 sec
Sample bins: [0]=65745 [1]=65250 ...
Validation: YES

[CUDA Version] (if enabled)
Time: ~0.03–0.07 sec
Validation: YES

----------------------------------------------------------
COMPILATION & EXECUTION:
-------------------------

For CPU only (Sequential + Multithreaded):
  g++ -std=c++17 Task14.cpp -o histogram -pthread

For CUDA (Enable CUDA and compile with nvcc):
  #define ENABLE_CUDA 1
  nvcc -std=c++14 Task14.cu -o histogram_cuda

Run:
  ./histogram

----------------------------------------------------------
MODIFICATION OPTIONS:
----------------------

- Increase DATA_SIZE to test scaling
- Compare with OpenMP version
- Write histogram to file for analysis
- Add time plots across versions

----------------------------------------------------------
CONCEPTS COVERED:
------------------

- Histogram computation
- Multithreading (std::thread)
- Mutex-free parallelism
- GPU programming with CUDA
- Atomic operations on device
- Performance measurement (chrono)
- Validation and benchmarking

