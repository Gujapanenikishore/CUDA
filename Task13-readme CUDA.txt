===================================================================
TASK 13: PARALLEL PRIME NUMBER FINDER (C++ SEQ | THREAD | CUDA)
===================================================================

OBJECTIVE:
----------
- Find all prime numbers up to 1,000,000
- Implement and compare 3 versions:
   1. Sequential (C++)
   2. Multithreaded (C++ std::thread)
   3. CUDA (GPU)

- Benchmark performance
- Validate correctness across all versions

-------------------------------------------------------------------
PRIME DEFINITION:
------------------
A prime number is a natural number greater than 1 that is divisible
only by 1 and itself.

E.g., 2, 3, 5, 7, 11, ...

-------------------------------------------------------------------
PRIME CHECK FUNCTION (USED IN ALL VERSIONS):
--------------------------------------------

bool isPrime(int n)
- Returns false if:
  - n < 2
  - n is even and not 2
- Loops i from 3 to sqrt(n) checking divisibility
- Efficient for large-scale testing

-------------------------------------------------------------------
VERSION 1: SEQUENTIAL
----------------------

Logic:
- Loop i from 2 to LIMIT (1,000,000)
- Call isPrime(i)
- If true → push_back to vector

Time Example:
  Found 78498 primes in 1.3 sec

Pros:
- Simple
- Baseline reference

Cons:
- Slow for large ranges

-------------------------------------------------------------------
VERSION 2: MULTITHREADED (C++11 std::thread)
--------------------------------------------

Logic:
- Divide range [2, LIMIT] into N chunks (N = hardware_concurrency)
- Each thread runs:
    findPrimesThreaded(start, end, shared_vector)
- Use mutex to avoid race condition on shared vector

Time Example:
  Found 78498 primes in 0.35 sec

Pros:
- Much faster using CPU parallelism
- Scales with core count

Cons:
- Requires locking (mutex)
- Overhead with many threads

-------------------------------------------------------------------
VERSION 3: CUDA (GPU)
----------------------

ENABLE_CUDA = 1 (to activate)

Kernel:
__global__ void findPrimesCUDA(...)
- Each thread checks one index using isPrimeGPU()
- Uses atomicAdd() to safely store index to output array

CUDA Setup:
- Allocate GPU memory for output and counter
- Launch kernel with 256 threads per block
- Copy results back to CPU

Time Example:
  Found 78498 primes in 0.04 sec

Pros:
- Extremely fast
- Perfect for massive parallelism (1M+ numbers)

Cons:
- Requires NVIDIA GPU
- Debugging and deployment more complex

-------------------------------------------------------------------
SAMPLE OUTPUT:
--------------

[Sequential Version]
Found 78498 primes in 1.3 sec

[Multithreaded Version]
Found 78498 primes in 0.35 sec

[CUDA Version]
Found 78498 primes in 0.04 sec

-------------------------------------------------------------------
COMPILATION & EXECUTION:
-------------------------

To compile without CUDA (sequential + threads):
  g++ -std=c++17 prime_finder.cpp -o primes -pthread

To compile with CUDA:
  nvcc -std=c++14 prime_finder.cu -o primes_cuda

To enable CUDA mode:
  #define ENABLE_CUDA 1 (in source)

Run:
  ./primes

-------------------------------------------------------------------
MODIFICATIONS:
--------------

- Change LIMIT = 1000000 to higher (e.g., 10 million)
- Log time per version in output file
- Validate all 3 outputs match (optional: file compare)
- Replace isPrime() with Sieve of Eratosthenes for optimization

-------------------------------------------------------------------
CONCEPTS COVERED:
------------------

- Sequential vs parallel execution
- Multithreading using std::thread
- Mutex for safe thread interaction
- GPU computing with CUDA (kernel, memory, sync)
- Benchmarking with std::chrono
- Atomic operations (atomicAdd)

