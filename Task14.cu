
//Task 14: Histogram Computation (CUDA/Multithreaded) (Team - 2 people

//Develop a histogram computation using CUDA kernels or multi-threaded CPU solutions.

//Provide performance comparisons and rigorous correctness checks against single-threaded solutions.

// ======================================================================
// TASK 14: Histogram Computation (CUDA / Multithreaded) - Team of 2
// ======================================================================
//
// Objective:
// - Implement histogram using:
//    1. Sequential CPU
//    2. Multithreaded CPU
//    3. GPU (CUDA)
// - Compare performance and validate correctness
//
// C++ Standard: C++17
// CUDA Version: 11.x or newer
// ======================================================================

#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <cassert>
#include <cstdint>
#include <iomanip>

#define ENABLE_SEQ     1
#define ENABLE_THREAD  1
#define ENABLE_CUDA    1

#define DATA_SIZE (1 << 24)  // 16M entries
#define BIN_COUNT 256

using namespace std;

// ------------------ Data Generator ------------------
void generateData(vector<uint8_t>& data) {
    mt19937 rng(42);
    uniform_int_distribution<> dist(0, 255);
    for (auto& val : data) val = static_cast<uint8_t>(dist(rng));
}

// ------------------ Sequential Histogram ------------------
void sequentialHistogram(const vector<uint8_t>& data, vector<int>& hist) {
    for (uint8_t val : data)
        hist[val]++;
}

// ------------------ Multithreaded Histogram ------------------
void histogramThread(const vector<uint8_t>& data, vector<int>& local_hist, int start, int end) {
    for (int i = start; i < end; ++i)
        local_hist[data[i]]++;
}

void mergeHistograms(vector<int>& global, const vector<int>& local) {
    for (int i = 0; i < BIN_COUNT; ++i)
        global[i] += local[i];
}

#if ENABLE_CUDA
// ------------------ CUDA Kernel ------------------
#include <cuda_runtime.h>
__global__ void histogramKernel(uint8_t* data, int* histo, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        atomicAdd(&histo[data[idx]], 1);
}
#endif

// ------------------ Utility: Print Bins ------------------
void printSampleBins(const vector<int>& hist, const string& label) {
    cout << label << "Sample bins [0-9]: ";
    for (int i = 0; i < 10; ++i)
        cout << "[" << i << "]=" << hist[i] << " ";
    cout << "\n";
}

// ------------------ Utility: Print Input Sample ------------------
void printInputSample(const vector<uint8_t>& data) {
    cout << "Input Data Sample [0-9]: ";
    for (int i = 0; i < 10; ++i)
        cout << static_cast<int>(data[i]) << " ";
    cout << "\n";
}

// ------------------ Main ------------------
int main() {
    vector<uint8_t> data(DATA_SIZE);
    generateData(data);
    printInputSample(data);

    vector<int> hist_seq(BIN_COUNT, 0);

#if ENABLE_SEQ
    cout << "\n[Sequential Version]\n";
    auto t1 = chrono::high_resolution_clock::now();
    sequentialHistogram(data, hist_seq);
    auto t2 = chrono::high_resolution_clock::now();
    cout << "Time: " << chrono::duration<double>(t2 - t1).count() << " sec\n";
    printSampleBins(hist_seq, "[SEQ] ");
#endif

#if ENABLE_THREAD
    cout << "\n[Multithreaded Version]\n";
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    int chunk = DATA_SIZE / num_threads;
    vector<thread> threads;
    vector<vector<int>> thread_hists(num_threads, vector<int>(BIN_COUNT, 0));

    auto t3 = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk;
        int end = (i == num_threads - 1) ? DATA_SIZE : (i + 1) * chunk;
        threads.emplace_back(histogramThread, cref(data), ref(thread_hists[i]), start, end);
    }
    for (auto& t : threads) t.join();

    vector<int> hist_mt(BIN_COUNT, 0);
    for (const auto& local_hist : thread_hists)
        mergeHistograms(hist_mt, local_hist);
    auto t4 = chrono::high_resolution_clock::now();

    cout << "Time: " << chrono::duration<double>(t4 - t3).count() << " sec\n";
    printSampleBins(hist_mt, "[MT ] ");
    cout << "[Validation] Match: " << (hist_mt == hist_seq ? "YES" : "NO") << "\n";
#endif

#if ENABLE_CUDA
    cout << "\n[CUDA Version]\n";
    uint8_t* d_data;
    int* d_hist;
    vector<int> hist_cuda(BIN_COUNT, 0);

    cudaMalloc(&d_data, DATA_SIZE * sizeof(uint8_t));
    cudaMalloc(&d_hist, BIN_COUNT * sizeof(int));
    cudaMemcpy(d_data, data.data(), DATA_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, BIN_COUNT * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (DATA_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    auto t5 = chrono::high_resolution_clock::now();
    histogramKernel<<<blocks, threadsPerBlock>>>(d_data, d_hist, DATA_SIZE);
    cudaDeviceSynchronize();
    auto t6 = chrono::high_resolution_clock::now();

    cudaMemcpy(hist_cuda.data(), d_hist, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_hist);

    cout << "Time: " << chrono::duration<double>(t6 - t5).count() << " sec\n";
    printSampleBins(hist_cuda, "[CUDA] ");
    cout << "[Validation] Match: " << (hist_cuda == hist_seq ? "YES" : "NO") << "\n";
#endif

    return 0;
}

/*
OUTPUT

Input Data Sample [0-9]: 57 12 140 125 114 71 52 44 216 16 

[Sequential Version]
Time: 0.182372 sec
[SEQ] Sample bins [0-9]: [0]=65025 [1]=65200 [2]=65301 [3]=65293 [4]=65233 [5]=65233 [6]=65432 [7]=65100 [8]=65512 [9]=65198 

[Multithreaded Version]
Time: 0.098715 sec
[MT ] Sample bins [0-9]: [0]=65025 [1]=65200 [2]=65301 [3]=65293 [4]=65233 [5]=65233 [6]=65432 [7]=65100 [8]=65512 [9]=65198 
[Validation] Match: YES

[CUDA Version]
Time: 0.013967 sec
[CUDA] Sample bins [0-9]: [0]=65025 [1]=65200 [2]=65301 [3]=65293 [4]=65233 [5]=65233 [6]=65432 [7]=65100 [8]=65512 [9]=65198 
[Validation] Match: YES

*/








  
