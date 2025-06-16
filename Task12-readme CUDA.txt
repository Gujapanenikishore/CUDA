=============================================================
TASK 12: CUDA PARALLEL MATRIX TRANSPOSE (C++ + CUDA)
=============================================================

OBJECTIVE:
----------
- Perform matrix transpose using GPU CUDA kernel
- Validate correctness by comparing with CPU transpose
- Visualize input, output (CPU and GPU), and verify result

-------------------------------------------------------------
WHAT IS MATRIX TRANSPOSE?
--------------------------
Given a matrix A of size R x C (rows x cols), its transpose Aᵗ is:
- Element Aᵗ[i][j] = A[j][i]

Example:
Original (4x4):
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16

Transpose (4x4):
1 5 9 13
2 6 10 14
3 7 11 15
4 8 12 16

-------------------------------------------------------------
CUDA IMPLEMENTATION LOGIC
--------------------------

__global__ void transposeKernel(...)

- Each thread calculates a single element in the transposed matrix
- Thread indices: 
   x = threadIdx.x + blockIdx.x * blockDim.x (column)
   y = threadIdx.y + blockIdx.y * blockDim.y (row)
- Only update if x < cols and y < rows
- Store transposed element:
  out[x * rows + y] = in[y * cols + x]

This follows **row-major order** indexing.

-------------------------------------------------------------
CPU TRANSPOSE FUNCTION
-----------------------

void transposeCPU(...)

- Uses nested for loops:
  for each row r and column c:
    out[c * rows + r] = in[r * cols + c];

-------------------------------------------------------------
MEMORY MANAGEMENT
------------------

- Device memory is allocated using cudaMalloc
- Data copied from host to device using cudaMemcpy
- After computation, output is copied back to host

dim3 blockSize(2, 2)
dim3 gridSize(...)
- Adjusts grid to match size of input matrix

cudaDeviceSynchronize() ensures kernel completion before data copy

-------------------------------------------------------------
PRINT & VALIDATE
------------------

printMatrix(): prints 2D matrix from 1D array

validate(): compares each element with CPU result within epsilon

-------------------------------------------------------------
SAMPLE OUTPUT
-------------

Original Matrix:
1  2  3  4
5  6  7  8
9 10 11 12
13 14 15 16

CPU Transposed Matrix:
1 5 9 13
2 6 10 14
3 7 11 15
4 8 12 16

GPU Transposed Matrix:
1 5 9 13
2 6 10 14
3 7 11 15
4 8 12 16

Validation: PASS

(Ensures correctness of CUDA kernel)

-------------------------------------------------------------
HOW TO COMPILE & RUN (NVCC)
----------------------------

To compile:
  nvcc -std=c++14 Task12.cu -o matrix_transpose

To run:
  ./matrix_transpose

-------------------------------------------------------------
MODIFICATIONS FOR TESTING:
----------------------------

- Change matrix size: `rows`, `cols`
- Test with non-square matrices (e.g., 3x5, 8x2)
- Add timing logic using `chrono` to compare speed
- Visualize matrices as formatted 2D tables
- Use shared memory for optimization (advanced)

-------------------------------------------------------------
CUDA CONCEPTS COVERED:
------------------------

- Thread blocks and grids
- Memory management (cudaMalloc, cudaMemcpy)
- Global kernel execution
- Synchronization (cudaDeviceSynchronize)
- Row-major vs column-major data layout
- Validation between CPU and GPU results

