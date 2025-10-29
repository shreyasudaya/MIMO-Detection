#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void print_first(float *d_data, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("First value: %f\n", d_data[0]);
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
    size_t num_elements = (size_t)rows * n_rx * n_tx;

    float *h_H = load_bin("H.bin", num_elements);

    float *d_H;
    cudaMalloc(&d_H, num_elements * sizeof(float));
    cudaMemcpy(d_H, h_H, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    print_first<<<1, 5>>>(d_H, num_elements);
    cudaDeviceSynchronize();

    // Cleanup
    free(h_H);
    cudaFree(d_H);
    return 0;
}