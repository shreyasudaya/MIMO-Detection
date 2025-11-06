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