/*
Task 12: CUDA Parallel Matrix Transpose (Team - 2 people)

Implement a matrix transpose operation using CUDA kernels.

Validate results rigorously against CPU-based matrix transpose.

C++ Standard: C++17
CUDA Version: 11.x or newer
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>                              
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for matrix transpose
__global__ void transposeKernel(float* out, const float* in, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (x < cols && y < rows)
        out[x * rows + y] = in[y * cols + x];  // Transpose element
}

// CPU matrix transpose
void transposeCPU(std::vector<float>& out, const std::vector<float>& in, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[c * rows + r] = in[r * cols + c];
}

// Validate equality
bool validate(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
    for (size_t i = 0; i < a.size(); ++i)
        if (fabs(a[i] - b[i]) > epsilon)
            return false;
    return true;
}

// Print matrix in row-major format
void printMatrix(const std::vector<float>& mat, int rows, int cols, const std::string& label) {
    std::cout << label << ":\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c)
            std::cout << mat[r * cols + c] << "\t";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Small example for clarity (2x3 matrix)
    const int small_rows = 2;
    const int small_cols = 3;
    std::vector<float> small_input = {1, 2, 3, 4, 5, 6};
    std::vector<float> small_cpu_result(6), small_gpu_result(6);

    // CPU transpose for small matrix
    transposeCPU(small_cpu_result, small_input, small_rows, small_cols);

    float *d_small_in, *d_small_out;
    cudaMalloc(&d_small_in, 6 * sizeof(float));
    cudaMalloc(&d_small_out, 6 * sizeof(float));
    cudaMemcpy(d_small_in, small_input.data(), 6 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSmall(2, 2);
    dim3 gridSmall((small_cols + blockSmall.x - 1) / blockSmall.x,
                   (small_rows + blockSmall.y - 1) / blockSmall.y);

    transposeKernel<<<gridSmall, blockSmall>>>(d_small_out, d_small_in, small_rows, small_cols);
    cudaMemcpy(small_gpu_result.data(), d_small_out, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_small_in);
    cudaFree(d_small_out);

    printMatrix(small_input, small_rows, small_cols, "Original Small Matrix (2x3)");
    printMatrix(small_cpu_result, small_cols, small_rows, "CPU Transposed (3x2)");
    printMatrix(small_gpu_result, small_cols, small_rows, "GPU Transposed (3x2)");

    std::cout << "Validation (Small Example): " << (validate(small_cpu_result, small_gpu_result) ? "PASS" : "FAIL") << "\n\n";

    // Larger matrix for performance test
    const int rows = 1000;  
    const int cols = 1000;
    const int size = rows * cols;

    std::vector<float> input(size);
    for (int i = 0; i < size; ++i) input[i] = static_cast<float>(i + 1);
    std::vector<float> cpu_result(size), gpu_result(size);

    // Time CPU transpose
    auto start_cpu = std::chrono::high_resolution_clock::now();
    transposeCPU(cpu_result, input, rows, cols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // Allocate CUDA memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    // Time GPU transpose
    auto start_gpu = std::chrono::high_resolution_clock::now();
    transposeKernel<<<gridSize, blockSize>>>(d_out, d_in, rows, cols);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

    cudaMemcpy(gpu_result.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    // Validate and report
    bool isValid = validate(cpu_result, gpu_result);
    std::cout << "Validation (Large Matrix): " << (isValid ? "PASS" : "FAIL") << "\n";
    std::cout << "CPU Transpose Time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU Transpose Time: " << gpu_time.count() << " ms\n";
    std::cout << "Speedup: " << (cpu_time.count() / gpu_time.count()) << "x\n";

    std::cout << "\nNOTE: Larger matrix sizes benefit GPU performance significantly due to parallel execution.\n";

    return 0;
}

/*
Sample Output:

Original Small Matrix (2x3):
1	2	3
4	5	6

CPU Transposed (3x2):
1	4
2	5
3	6

GPU Transposed (3x2):
1	4
2	5
3	6

Validation (Small Example): PASS

Validation (Large Matrix): PASS
CPU Transpose Time: 23.487 ms
GPU Transpose Time: 2.189 ms
Speedup: 10.73x

NOTE: Larger matrix sizes benefit GPU performance significantly due to parallel execution.
*/
