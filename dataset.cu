#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>
#include <curand.h>
#include <curand_kernel.h>

const int N = 30000;
const int Nr = 4;
const int Nt = 8;
const float SNR_dB = 10.0f;
const unsigned long long SEED = 12345ULL;


struct Complex {
    float r;
    float i;
};

__device__ Complex complex_mul(const Complex &a, const Complex &b) {
    Complex c;
    c.r = a.r * b.r - a.i * b.i;
    c.i = a.r * b.i + a.i * b.r;
    return c;
}
__device__ Complex complex_add(Complex a, Complex b) {
    return {a.r + b.r, a.i + b.i};
}

__device__ Complex complex_scale(Complex a, float s) {
    return {a.r * s, a.i * s};
}

void complex_write(const std::string &fname, const std::vector<float> &buf) {
    std::ofstream ofs(fname, std::ios::binary);
    if (!ofs) { 
        std::fprintf(stderr, "Failed to open %s\n", fname.c_str()); 
        std::exit(1); 
    }
    ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(float));
    ofs.close();
}


__constant__ Complex QPSK[4] = {
    {1.0f, 1.0f},
    {1.0f, -1.0f},
    {-1.0f, -1.0f},
    {-1.0f, 1.0f}
};


//Dataset Generation
__global__ void dataset_gen(float *Hbuf, float *Xbuf, float *Ybuf, float noise_sigma) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    curandState_t state;
    curand_init(SEED, m, 0, &state);

    Complex* H = (Complex*)Hbuf + (size_t)m * Nr * Nt; 
    Complex* X = (Complex*)Xbuf + (size_t)m * Nt;      
    Complex* Y = (Complex*)Ybuf + (size_t)m * Nr;

    const float scale = 1.0f / sqrtf(2.0f);
    for (int i = 0; i < Nr * Nt; i++) {
        float hr = curand_normal(&state) * scale;
        float hi = curand_normal(&state) * scale;
        H[i] = {hr, hi};
    }

    for (int i = 0; i < Nt; i++) {
        unsigned int r_bits = curand(&state); // Get random 32-bit integer
        int b0 = (int)(r_bits & 1);
        int b1 = (int)((r_bits >> 1) & 1);
        int idxq = b0 * 2 + b1;
        X[i] = QPSK[idxq];
    }

    for (int i = 0; i < Nr; i++) {
        Complex sum = {0.0f, 0.0f};
        for (int j = 0; j < Nt; j++) {
            sum = complex_add(sum, complex_mul(H[i * Nt + j], X[j]));
        }
        float noise_r = curand_normal(&state) * noise_sigma;
        float noise_i = curand_normal(&state) * noise_sigma;
        Y[i] = complex_add(sum, {noise_r, noise_i});
 
    }
}


int main(int argc, char** argv) {


    float SNR = powf(10.0f, SNR_dB / 10.0f);
    float noise_var = Nt / SNR;
    float noise_sigma = sqrtf(noise_var / 2.0f); 

    std::cout << "Generating dataset: N=" << N << " Nr=" << Nr << " Nt=" << Nt 
              << " SNR(dB)=" << SNR_dB << "\n";

    const size_t H_size_bytes = (size_t)N * Nr * Nt * 2 * sizeof(float);
    const size_t X_size_bytes = (size_t)N * Nt * 2 * sizeof(float);
    const size_t Y_size_bytes = (size_t)N * Nr * 2 * sizeof(float);
    
    std::vector<float> H_host_buf(H_size_bytes / sizeof(float));
    std::vector<float> X_host_buf(X_size_bytes / sizeof(float));
    std::vector<float> Y_host_buf(Y_size_bytes / sizeof(float));

    float *H_dev_buf, *X_dev_buf, *Y_dev_buf;
    if (cudaMalloc(&H_dev_buf, H_size_bytes) != cudaSuccess ||
        cudaMalloc(&X_dev_buf, X_size_bytes) != cudaSuccess ||
        cudaMalloc(&Y_dev_buf, Y_size_bytes) != cudaSuccess) {
        std::fprintf(stderr, "CUDA memory allocation failed!\n");
        return 1;
    }

    // 3. CUDA Event Setup for Timing ⏱️
    cudaEvent_t start_event, stop_event_HtoD, stop_event_kernel, stop_event_DtoH;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event_HtoD);
    cudaEventCreate(&stop_event_kernel);
    cudaEventCreate(&stop_event_DtoH);

    float time_HtoD, time_kernel, time_DtoH, total_time;
    
    // --- Phase 1: HtoD Setup (Allocation is synchronous, copying is timed) ---
    cudaEventRecord(start_event, 0); // Record start of all work

    // Memory Copy: Host -> Device (The data is copied from empty host buffers, so timing this copy primarily measures transfer time)
    cudaMemcpy(H_dev_buf, H_host_buf.data(), H_size_bytes, cudaMemcpyHostToDevice); // Although empty, this ensures the device memory is ready
    cudaMemcpy(X_dev_buf, X_host_buf.data(), X_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y_dev_buf, Y_host_buf.data(), Y_size_bytes, cudaMemcpyHostToDevice);
    
    cudaEventRecord(stop_event_HtoD, 0); // Record end of HtoD

    // --- Phase 2: Kernel Execution (Computation) ---
    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    dataset_gen<<<numBlocks, threadsPerBlock>>>(
        H_dev_buf, 
        X_dev_buf, 
        Y_dev_buf, 
        noise_sigma
    );

    cudaEventRecord(stop_event_kernel, 0); // Record end of Kernel

    // --- Phase 3: DtoH Results Retrieval ---
    // Copy results back: Device -> Host
    cudaMemcpy(H_host_buf.data(), H_dev_buf, H_size_bytes, cudaMemcpyDeviceToHost); 
    cudaMemcpy(X_host_buf.data(), X_dev_buf, X_size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Y_host_buf.data(), Y_dev_buf, Y_size_bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_event_DtoH, 0); // Record end of DtoH (and all work)
    cudaDeviceSynchronize(); // Force the CPU to wait for all GPU commands to complete

    // 4. Calculate Elapsed Time (in milliseconds)
    cudaEventElapsedTime(&time_HtoD, start_event, stop_event_HtoD);
    cudaEventElapsedTime(&time_kernel, stop_event_HtoD, stop_event_kernel);
    cudaEventElapsedTime(&time_DtoH, stop_event_kernel, stop_event_DtoH);
    cudaEventElapsedTime(&total_time, start_event, stop_event_DtoH);

    // 5. Print Timings
    std::cout << "\n--- Timing Results (N=" << N << " samples) ---\n";
    std::printf("Data Copy (H->D) : %10.3f ms\n", time_HtoD);
    std::printf("Kernel Execution : %10.3f ms\n", time_kernel);
    std::printf("Data Copy (D->H) : %10.3f ms\n", time_DtoH);
    std::printf("Total CUDA Time  : %10.3f ms\n", total_time);
    std::cout << "------------------------------------------\n";

    // 6. Write Binary Files
    std::string H_path = "data/H.bin";
    std::string X_path = "data/X.bin";
    std::string Y_path = "data/Y.bin";

    complex_write(H_path, H_host_buf);
    complex_write(X_path, X_host_buf);
    complex_write(Y_path, Y_host_buf);

    // 7. Cleanup
    cudaFree(H_dev_buf);
    cudaFree(X_dev_buf);
    cudaFree(Y_dev_buf);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event_HtoD);
    cudaEventDestroy(stop_event_kernel);
    cudaEventDestroy(stop_event_DtoH);

    std::cout << "Done.\n";
    return 0;
}