/*
Task 12: CUDA Parallel Matrix Transpose (Team - 2 people)

Implement a matrix transpose operation using CUDA kernels.

Validate results rigorously against CPU-based matrix transpose.

C++ Standard: C++17
CUDA Version: 11.x or newer*/

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

// Print top-left corner of matrix
void printMatrixPreview(const std::vector<float>& mat, int rows, int cols, const std::string& label, int preview = 5) {
    std::cout << label << " (Top-Left " << preview << "x" << preview << "):\n";
    for (int r = 0; r < preview; ++r) {
        for (int c = 0; c < preview; ++c)
            std::cout << mat[r * cols + c] << "\t";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    const int rows = 1000;
    const int cols = 1000;
    const int size = rows * cols;

    std::vector<float> input(size);
    for (int i = 0; i < size; ++i)
        input[i] = static_cast<float>(i + 1);

    std::vector<float> cpu_result(size), gpu_result(size);

    // CPU Transpose
    auto start_cpu = std::chrono::high_resolution_clock::now();
    transposeCPU(cpu_result, input, rows, cols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // GPU Transpose
    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    transposeKernel<<<gridSize, blockSize>>>(d_out, d_in, rows, cols);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

    cudaMemcpy(gpu_result.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    // Print sample
    printMatrixPreview(input, rows, cols, "Original Matrix");
    printMatrixPreview(cpu_result, cols, rows, "CPU Transposed Matrix");
    printMatrixPreview(gpu_result, cols, rows, "GPU Transposed Matrix");

    // Validate and time results
    bool isValid = validate(cpu_result, gpu_result);
    std::cout << "Validation: " << (isValid ? "PASS" : "FAIL") << "\n";
    std::cout << "CPU Transpose Time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU Transpose Time: " << gpu_time.count() << " ms\n";
    std::cout << "Speedup: " << (cpu_time.count() / gpu_time.count()) << "x\n";

    return 0;
}



/*SAMPLE OUTPUT 

Original Matrix (Top-Left 5x5):
1	2	3	4	5
1001	1002	1003	1004	1005
2001	2002	2003	2004	2005
3001	3002	3003	3004	3005
4001	4002	4003	4004	4005

CPU Transposed Matrix (Top-Left 5x5):
1	1001	2001	3001	4001
2	1002	2002	3002	4002
3	1003	2003	3003	4003
4	1004	2004	3004	4004
5	1005	2005	3005	4005

GPU Transposed Matrix (Top-Left 5x5):
1	1001	2001	3001	4001
2	1002	2002	3002	4002
3	1003	2003	3003	4003
4	1004	2004	3004	4004
5	1005	2005	3005	4005

Validation: PASS
CPU Transpose Time: 23.0814 ms
GPU Transpose Time: 2.1822 ms
Speedup: 10.58x */


 
NOTE: Larger matrix sizes benefit GPU performance significantly due to parallel execution.
*/
