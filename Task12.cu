/*Task 12: CUDA Parallel Matrix Transpose (Team - 2 people)

Implement a matrix transpose operation using CUDA kernels.

Validate results rigorously against CPU-based matrix transpose.*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))  // Column-major indexing if needed

// CUDA kernel for matrix transpose
__global__ void transposeKernel(float* out, const float* in, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x < cols && y < rows)
        out[x * rows + y] = in[y * cols + x];  // Transpose
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

// Print matrix (row-major)
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
    const int rows = 4;
    const int cols = 4;
    const int size = rows * cols;

    std::vector<float> input = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    std::vector<float> cpu_result(size), gpu_result(size);

    // CPU transpose
    transposeCPU(cpu_result, input, rows, cols);

    // Allocate CUDA memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(2, 2);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    // GPU transpose
    transposeKernel<<<gridSize, blockSize>>>(d_out, d_in, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_result.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    // Print results
    printMatrix(input, rows, cols, "Original Matrix");
    printMatrix(cpu_result, cols, rows, "CPU Transposed Matrix");
    printMatrix(gpu_result, cols, rows, "GPU Transposed Matrix");

    std::cout << "Validation: " << (validate(cpu_result, gpu_result) ? "PASS" : "FAIL") << "\n";

    return 0;
}

/*_______________________
OUTPUT

Original Matrix:
1	2	3	4
5	6	7	8
9	10	11	12
13	14	15	16

CPU Transposed Matrix:
1	5	9	13
2	6	10	14
3	7	11	15
4	8	12	16

GPU Transposed Matrix:
1	5	9	13
2	6	10	14
3	7	11	15
4	8	12	16

Validation: PASS   */