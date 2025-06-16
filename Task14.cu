
//Task 14: Histogram Computation (CUDA/Multithreaded) (Team - 2 people

//Develop a histogram computation using CUDA kernels or multi-threaded CPU solutions.

//Provide performance comparisons and rigorous correctness checks against single-threaded solutions.


// ============================
// Histogram Computation (Unified)
// ============================

#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <cassert>

#define ENABLE_SEQ     1
#define ENABLE_THREAD  1
#define ENABLE_CUDA    0  // Set to 1 if compiling with nvcc and running on GPU

#define DATA_SIZE (1 << 24)  // 16 million entries
#define BIN_COUNT 256

using namespace std;

// ---------------- Generate Random Data ----------------
void generateData(std::vector<uint8_t>& data) {
    std::mt19937 rng(42);  // fixed seed for repeatability
    std::uniform_int_distribution<> dist(0, 255);
    for (auto& val : data) val = dist(rng);
}

// ---------------- Sequential Version ----------------
void sequentialHistogram(const vector<uint8_t>& data, vector<int>& hist) {
    for (auto val : data) hist[val]++;
}

// ---------------- Multithreaded Version ----------------
void histogramThread(const vector<uint8_t>& data, vector<int>& local_hist, int start, int end) {
    for (int i = start; i < end; ++i)
        local_hist[data[i]]++;
}

void mergeHistograms(vector<int>& global, const vector<int>& local) {
    for (int i = 0; i < BIN_COUNT; ++i)
        global[i] += local[i];
}

// ---------------- CUDA Version ----------------
#if ENABLE_CUDA
#include <cuda_runtime.h>

__global__ void histogramKernel(uint8_t* data, int* histo, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&histo[data[idx]], 1);
    }
}
#endif

// ---------------- Main Function ----------------
int main() {
    vector<uint8_t> data(DATA_SIZE);
    generateData(data);
    vector<int> hist_seq(BIN_COUNT, 0);

#if ENABLE_SEQ
    cout << "\n[Sequential Version]\n";
    auto t0 = chrono::high_resolution_clock::now();
    sequentialHistogram(data, hist_seq);
    auto t1 = chrono::high_resolution_clock::now();
    cout << "Time: " << chrono::duration<double>(t1 - t0).count() << " sec\n";

    cout << "Sample bins: ";
    for (int i = 0; i < 10; ++i)
        cout << "[" << i << "]=" << hist_seq[i] << " ";
    cout << "\n";
#endif

#if ENABLE_THREAD
    cout << "\n[Multithreaded Version]\n";
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    int chunk = DATA_SIZE / num_threads;
    vector<thread> threads;
    vector<vector<int>> thread_hists(num_threads, vector<int>(BIN_COUNT, 0));

    auto t2 = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk;
        int end = (i == num_threads - 1) ? DATA_SIZE : (i + 1) * chunk;
        threads.emplace_back(histogramThread, cref(data), ref(thread_hists[i]), start, end);
    }
    for (auto& t : threads) t.join();

    vector<int> hist_mt(BIN_COUNT, 0);
    for (const auto& th : thread_hists)
        mergeHistograms(hist_mt, th);
    auto t3 = chrono::high_resolution_clock::now();
    cout << "Time: " << chrono::duration<double>(t3 - t2).count() << " sec\n";

    cout << "Sample bins: ";
    for (int i = 0; i < 10; ++i)
        cout << "[" << i << "]=" << hist_mt[i] << " ";
    cout << "\n";

    cout << "[Validation] Match: " << (hist_mt == hist_seq ? "YES" : "NO") << "\n";
#endif

#if ENABLE_CUDA
    cout << "\n[CUDA Version]\n";
    uint8_t* d_data;
    int* d_hist;
    vector<int> hist_cuda(BIN_COUNT, 0);

    cudaMalloc(&d_data, DATA_SIZE);
    cudaMalloc(&d_hist, BIN_COUNT * sizeof(int));
    cudaMemcpy(d_data, data.data(), DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, BIN_COUNT * sizeof(int));

    int threads = 256;
    int blocks = (DATA_SIZE + threads - 1) / threads;
    auto t4 = chrono::high_resolution_clock::now();
    histogramKernel<<<blocks, threads>>>(d_data, d_hist, DATA_SIZE);
    cudaDeviceSynchronize();
    auto t5 = chrono::high_resolution_clock::now();

    cudaMemcpy(hist_cuda.data(), d_hist, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_hist);

    cout << "Time: " << chrono::duration<double>(t5 - t4).count() << " sec\n";

    cout << "Sample bins: ";
    for (int i = 0; i < 10; ++i)
        cout << "[" << i << "]=" << hist_cuda[i] << " ";
    cout << "\n";

    cout << "[Validation] Match: " << (hist_cuda == hist_seq ? "YES" : "NO") << "\n";
#endif

    return 0;
}


/*___________________

OUTPUT

[Sequential Version]
Time: 0.183488 sec
Sample bins: [0]=65745 [1]=65250 [2]=65786 [3]=65930 [4]=65757 [5]=65382 [6]=65355 [7]=65517 [8]=65952 [9]=65491 

[Multithreaded Version]
Time: 0.109256 sec
Sample bins: [0]=65745 [1]=65250 [2]=65786 [3]=65930 [4]=65757 [5]=65382 [6]=65355 [7]=65517 [8]=65952 [9]=65491 
[Validation] Match: YES*/