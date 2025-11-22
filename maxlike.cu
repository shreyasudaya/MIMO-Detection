#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void print_first(float *d_data, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("First value: %f\n", d_data[0]);
}

#define CHECK_CUDA(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                              \
    }                                                         \
} while(0)

__global__ void maximum_likelihood_detector(float *d_H, float *d_Y, int rows, int n_rx, int n_tx) {
    // Kernel implementation goes here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
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
int main() {
    const int rows = 34935;
    const int n_rx = 4;
    const int n_tx = 8;
    size_t x_row = (size_t)rows * n_tx;
    size_t y_row = (size_t)rows * n_rx;
    size_t num_elements = (size_t)rows * n_rx * n_tx;

    float *h_H = load_bin("H.bin", num_elements);
    float *h_X = load_bin("X.bin", x_row);
    float *h_Y = load_bin("Y.bin", y_row);
    float *d_H, *d_X, *d_Y;
    size_t size_H = (size_t)rows * n_rx * n_tx;
    size_t size_X = (size_t)rows * n_tx;
    size_t size_Y = (size_t)rows * n_rx;

    printf("Loading files: H(%zu), X(%zu), Y(%zu)\n", size_H, size_X, size_Y);

    cudaMalloc(&d_H, num_elements * sizeof(float));
    cudaMalloc(&d_X, x_row * sizeof(float));
    cudaMalloc(&d_Y, y_row * sizeof(float));
    float *d_H=NULL, *d_X=NULL, *d_Y=NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_H, size_H * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_X, size_X * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_Y, size_Y * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_H, h_H, size_H * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, size_X * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Y, h_Y, size_Y * sizeof(float), cudaMemcpyHostToDevice));

    cudaMemcpy(d_H, h_H, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    free(h_H);
    cudaFree(d_H);
    return 0;
}