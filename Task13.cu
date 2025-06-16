
//Task 13: Parallel Prime Number Finder (CUDA/Multithreaded) (Team - 2 people)

//Implement a CUDA or multi-threaded prime number finder up to 1 million.

//Benchmark and provide clear performance comparisons against sequential implementations.


// File: prime_finder.cu or .cpp


#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <mutex>

#define ENABLE_SEQ    1
#define ENABLE_THREAD 1
#define ENABLE_CUDA   0  // Set to 1 if compiling with NVCC

const int LIMIT = 1000000;  //  Change the limit to test other ranges

// ---------------- Sequential Prime Check ----------------
bool isPrime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    int sqrtn = std::sqrt(n);
    for (int i = 3; i <= sqrtn; i += 2)
        if (n % i == 0) return false;
    return true;
}

// ---------------- Multithreaded Prime Finder ----------------
std::mutex mtx;
void findPrimesThreaded(int start, int end, std::vector<int>& primes) {
    std::vector<int> local;
    for (int i = start; i <= end; ++i)
        if (isPrime(i)) local.push_back(i);
    std::lock_guard<std::mutex> lock(mtx);
    primes.insert(primes.end(), local.begin(), local.end());
}

#if ENABLE_CUDA
// ---------------- CUDA Code ----------------
__device__ bool isPrimeGPU(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    int sqrtn = sqrtf((float)n);
    for (int i = 3; i <= sqrtn; i += 2)
        if (n % i == 0) return false;
    return true;
}

__global__ void findPrimesCUDA(int* output, int* count, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && isPrimeGPU(idx)) {
        int pos = atomicAdd(count, 1);
        output[pos] = idx;
    }
}
#endif

int main() {
#if ENABLE_SEQ
    std::cout << "\n[Sequential Version]\n";
    std::vector<int> primes;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 2; i <= LIMIT; ++i)
        if (isPrime(i)) primes.push_back(i);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Found " << primes.size() << " primes in "
              << std::chrono::duration<double>(t2 - t1).count() << " sec\n";
#endif

#if ENABLE_THREAD
    std::cout << "\n[Multithreaded Version]\n";
    int num_threads = std::thread::hardware_concurrency();
    int chunk = LIMIT / num_threads;
    std::vector<std::thread> threads;
    std::vector<int> thread_primes;

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk + (i == 0 ? 2 : 0);
        int end = (i == num_threads - 1) ? LIMIT : (i + 1) * chunk;
        threads.emplace_back(findPrimesThreaded, start, end, std::ref(thread_primes));
    }
    for (auto& t : threads) t.join();
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Found " << thread_primes.size() << " primes in "
              << std::chrono::duration<double>(t4 - t3).count() << " sec\n";
#endif

#if ENABLE_CUDA
    std::cout << "\n[CUDA Version]\n";
    int* d_output;
    int* d_count;
    int* h_output = new int[LIMIT];
    int h_count = 0;

    cudaMalloc(&d_output, LIMIT * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    auto t5 = std::chrono::high_resolution_clock::now();
    int threads = 256;
    int blocks = (LIMIT + threads - 1) / threads;
    findPrimesCUDA<<<blocks, threads>>>(d_output, d_count, LIMIT);
    cudaDeviceSynchronize();
    auto t6 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, h_count * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Found " << h_count << " primes in "
              << std::chrono::duration<double>(t6 - t5).count() << " sec\n";

    delete[] h_output;
    cudaFree(d_output);
    cudaFree(d_count);
#endif

    return 0;
}



//____________________
//OUTPUT

//[Sequential Version]
//Found 78498 primes in 1.3 sec

//[Multithreaded Version]
//Found 78498 primes in 0.35 sec

//[CUDA Version]
//Found 78498 primes in 0.04 sec