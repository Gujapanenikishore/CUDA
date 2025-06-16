/*Task 11: CUDA Parallel Vector Addition (Team - 2 people)

Implement parallel vector addition using CUDA.

Compare performance, correctness, and scalability against sequential C++ implementations.*/


#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// CUDA kernel
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Sequential CPU implementation
void vectorAddCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < A.size(); ++i)
        C[i] = A[i] + B[i];
}

// Validation
bool validate(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
    for (int i = 0; i < a.size(); ++i)
        if (fabs(a[i] - b[i]) > epsilon)
            return false;
    return true;
}

int main() {
    // Small test case
    int N = 8;

    std::vector<float> A = {1.5, 3.0, 5.2, 7.1, 0.5, -1.0, 2.2, 4.4};
    std::vector<float> B = {2.5, -3.0, 1.8, 0.9, 3.5, 1.0, -2.2, 5.6};
    std::vector<float> C_cpu(N);
    std::vector<float> C_gpu(N);

    // CPU time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(A, B, C_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy to GPU
    cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 8;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // GPU time
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorAddKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Copy result back
    cudaMemcpy(C_gpu.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Validation
    bool ok = validate(C_cpu, C_gpu);

    // Report
    std::cout << "Vector Size: " << N << "\n";
    std::cout << "CPU Time: " << cpu_time.count() << " sec\n";
    std::cout << "GPU Time: " << gpu_time.count() << " sec\n";
    std::cout << "Speedup:   " << cpu_time.count() / gpu_time.count() << "x\n";
    std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << "\n\n";

    std::cout << "Sample Output (Vector Add):\n";
    for (int i = 0; i < N; ++i)
        std::cout << A[i] << " + " << B[i] << " = " << C_gpu[i] << "\n";

    return 0;
}



/*_________-__
OUTPUT
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
 */