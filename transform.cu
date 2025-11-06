#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CHECK(call) { const cudaError_t e = call; if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }

__device__ float2 qpsk_mod(int bits) {
    float re = (bits & 1) ? -1.0f : 1.0f;
    float im = (bits & 2) ? -1.0f : 1.0f;
    return make_float2(re / sqrtf(2.0f), im / sqrtf(2.0f));
}

__device__ int qpsk_demod(float2 s) {
    int b0 = (s.x < 0);
    int b1 = (s.y < 0);
    return (b1 << 1) | b0;
}

__global__ void init_curand(curandState *state, unsigned long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &state[i]);
}

// Compute received vector Y = H * X (elementwise simplification)
__global__ void mimo_forward_kernel(const float *H, const float *X, float2 *Y, int rows, int n_rx, int n_tx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * n_rx) return;

    int row = idx / n_rx;
    int rx = idx % n_rx;

    float2 sum = make_float2(0.f, 0.f);
    for (int tx = 0; tx < n_tx; tx++) {
        int h_idx = row * (n_rx * n_tx) + rx * n_tx + tx;
        float h = H[h_idx];
        float x = X[row * n_tx + tx];
        sum.x += h * x;
    }
    Y[idx] = sum;
}

// Add AWGN noise based on SNR
__global__ void add_noise_kernel(float2 *Y, float noise_std, curandState *state, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState localState = state[i];
    float nx = noise_std * curand_normal(&localState);
    float ny = noise_std * curand_normal(&localState);
    Y[i].x += nx;
    Y[i].y += ny;
    state[i] = localState;
}

// MIMO detection (approx ML + MMSE)
__global__ void mimo_detect_kernel(const float *H, const float2 *Y, float2 *X_hat_ML, float2 *X_hat_MMSE,
                                   int rows, int n_rx, int n_tx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * n_tx) return;

    int row = idx / n_tx;
    int tx = idx % n_tx;

    // Compute matched filter output for MMSE
    float h_sum = 0.0f;
    float2 y_sum = make_float2(0.f, 0.f);

    for (int rx = 0; rx < n_rx; rx++) {
        int h_idx = row * (n_rx * n_tx) + rx * n_tx + tx;
        float h = H[h_idx];
        float2 y = Y[row * n_rx + rx];
        y_sum.x += h * y.x;
        y_sum.y += h * y.y;
        h_sum += h * h;
    }

    // MMSE (scalar approx)
    float inv = 1.0f / (h_sum + 0.01f);
    X_hat_MMSE[idx].x = inv * y_sum.x;
    X_hat_MMSE[idx].y = inv * y_sum.y;

    // ML detection (choose closest QPSK symbol)
    float best_dist = 1e9f;
    float2 best_s;
    for (int b = 0; b < 4; b++) {
        float2 s = qpsk_mod(b);
        float dx = X_hat_MMSE[idx].x - s.x;
        float dy = X_hat_MMSE[idx].y - s.y;
        float dist = dx * dx + dy * dy;
        if (dist < best_dist) {
            best_dist = dist;
            best_s = s;
        }
    }
    X_hat_ML[idx] = best_s;
}

__global__ void compute_ber_kernel(const float *X, const float2 *X_hat, int *bit_errors, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Original transmitted bits approximated by sign
    int true_bits = ((X[i] < 0) ? 1 : 0);
    int est_bits = qpsk_demod(X_hat[i]);
    int diff = __popc(true_bits ^ est_bits);
    atomicAdd(bit_errors, diff);
}

float* load_bin(const char* filename, size_t num_elements) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("File open failed");
        exit(1);
    }
    float *data = (float*)malloc(num_elements * sizeof(float));
    fread(data, sizeof(float), num_elements, fp);
    fclose(fp);
    return data;
}

int main() {
    const int rows = 34935;
    const int n_rx = 4;
    const int n_tx = 8;
    size_t H_size = (size_t)rows * n_rx * n_tx;
    size_t X_size = (size_t)rows * n_tx;
    size_t Y_size = (size_t)rows * n_rx;

    float *h_H = load_bin("H.bin", H_size);
    float *h_X = load_bin("X.bin", X_size);
    float *h_Y = load_bin("Y.bin", Y_size); // used for reference, not detection

    float *d_H, *d_X;
    float2 *d_Y, *d_X_hat_ML, *d_X_hat_MMSE;
    curandState *d_state;
    int *d_bit_errors;

    CHECK(cudaMalloc(&d_H, H_size * sizeof(float)));
    CHECK(cudaMalloc(&d_X, X_size * sizeof(float)));
    CHECK(cudaMalloc(&d_Y, Y_size * sizeof(float2)));
    CHECK(cudaMalloc(&d_X_hat_ML, X_size * sizeof(float2)));
    CHECK(cudaMalloc(&d_X_hat_MMSE, X_size * sizeof(float2)));
    CHECK(cudaMalloc(&d_state, Y_size * sizeof(curandState)));
    CHECK(cudaMalloc(&d_bit_errors, sizeof(int)));

    CHECK(cudaMemcpy(d_H, h_H, H_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_X, h_X, X_size * sizeof(float), cudaMemcpyHostToDevice));

    const float snr_db_values[] = {0, 5, 10, 15, 20, 25, 30};
    const int num_snr = sizeof(snr_db_values) / sizeof(float);
    const int threads = 256;
    const int blocksY = (Y_size + threads - 1) / threads;
    const int blocksX = (X_size + threads - 1) / threads;

    init_curand<<<blocksY, threads>>>(d_state, time(NULL), Y_size);
    cudaDeviceSynchronize();

    printf("SNR(dB), ML_BER, MMSE_BER, Throughput(Mb/s)\n");

    for (int s = 0; s < num_snr; s++) {
        float snr_db = snr_db_values[s];
        float snr_lin = powf(10.0f, snr_db / 10.0f);
        float noise_std = 1.0f / sqrtf(2.0f * snr_lin);

        // Generate clean Y = H * X
        mimo_forward_kernel<<<blocksY, threads>>>(d_H, d_X, d_Y, rows, n_rx, n_tx);
        cudaDeviceSynchronize();

        // Add noise
        add_noise_kernel<<<blocksY, threads>>>(d_Y, noise_std, d_state, Y_size);
        cudaDeviceSynchronize();

        // Detection
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        mimo_detect_kernel<<<blocksX, threads>>>(d_H, d_Y, d_X_hat_ML, d_X_hat_MMSE, rows, n_rx, n_tx);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        // Compute BERs
        int h_bit_errors = 0;
        CHECK(cudaMemset(d_bit_errors, 0, sizeof(int)));
        compute_ber_kernel<<<blocksX, threads>>>(d_X, d_X_hat_ML, d_bit_errors, X_size);
        CHECK(cudaMemcpy(&h_bit_errors, d_bit_errors, sizeof(int), cudaMemcpyDeviceToHost));
        float ber_ml = (float)h_bit_errors / (X_size * 2);

        CHECK(cudaMemset(d_bit_errors, 0, sizeof(int)));
        compute_ber_kernel<<<blocksX, threads>>>(d_X, d_X_hat_MMSE, d_bit_errors, X_size);
        CHECK(cudaMemcpy(&h_bit_errors, d_bit_errors, sizeof(int), cudaMemcpyDeviceToHost));
        float ber_mmse = (float)h_bit_errors / (X_size * 2);

        float throughput = (X_size * 2 / 1e6f) / (ms / 1000.0f);
        printf("%5.1f dB, %8.6f, %8.6f, %8.3f\n", snr_db, ber_ml, ber_mmse, throughput);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    free(h_H); free(h_X); free(h_Y);
    cudaFree(d_H); cudaFree(d_X); cudaFree(d_Y);
    cudaFree(d_X_hat_ML); cudaFree(d_X_hat_MMSE);
    cudaFree(d_state); cudaFree(d_bit_errors);
    return 0;
}
