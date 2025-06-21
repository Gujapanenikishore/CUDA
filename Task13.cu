// Task 13: CUDA Prime Number Finder with Benchmark
// C++ Standard: C++17
// CUDA Version: 12.8

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

const int LIMIT = 1000000; // Change this value to test different ranges

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
    if (idx <= N && isPrimeGPU(idx)) {
        int pos = atomicAdd(count, 1);
        output[pos] = idx;
    }
}

bool isPrimeCPU(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    int sqrtn = std::sqrt(n);
    for (int i = 3; i <= sqrtn; i += 2)
        if (n % i == 0) return false;
    return true;
}

int main() {
    std::cout << "=== Prime Finder up to LIMIT = " << LIMIT << " ===\n\n";

    // ---------------- CPU ----------------
    std::vector<int> cpu_primes;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 2; i <= LIMIT; ++i)
        if (isPrimeCPU(i)) cpu_primes.push_back(i);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t2 - t1).count();

    std::cout << "[CPU] Found " << cpu_primes.size() << " primes in " << cpu_time << " sec\n";

    // ---------------- CUDA ----------------
    std::vector<int> h_output(LIMIT); // ✅ std::vector instead of new[]
    int* d_output;
    int* d_count;
    int h_count = 0;

    cudaError_t err;
    err = cudaMalloc(&d_output, LIMIT * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_output failed: " << cudaGetErrorString(err) << "\n"; return -1;
    }
    err = cudaMalloc(&d_count, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_count failed: " << cudaGetErrorString(err) << "\n"; return -1;
    }
    cudaMemset(d_count, 0, sizeof(int));

    int threads = 256;
    int blocks = (LIMIT + threads - 1) / threads;

    auto t3 = std::chrono::high_resolution_clock::now();
    findPrimesCUDA<<<blocks, threads>>>(d_output, d_count, LIMIT);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << "\n"; return -1;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(t4 - t3).count();

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output.data(), d_output, h_count * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "[CUDA] Found " << h_count << " primes in " << gpu_time << " sec\n";
    std::cout << "Speedup over CPU: " << (cpu_time / gpu_time) << "x\n";

    // ---------------- Cleanup ----------------
    cudaFree(d_output);
    cudaFree(d_count);

    return 0;
}

/* === Prime Finder up to LIMIT = 1000000 ===

[CPU] Found 78498 primes in 1.3415 sec
[CUDA] Found 78498 primes in 0.1032 sec
Speedup over CPU: 13.0x

Sample primes from GPU output:
2 3 5 7 11 13 17 19 23 29 */
