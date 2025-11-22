#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <complex>
#include <cstdio>
#include <cstdlib>




#define CHECK(call) { const cudaError_t e = call; if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }



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

std::complex<double>* load_complex128(const char* filename, size_t num_elements) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("File open failed");
        exit(1);
    }

    std::complex<double>* data =
        (std::complex<double>*)malloc(num_elements * sizeof(std::complex<double>));

    size_t read = fread(data, sizeof(std::complex<double>), num_elements, fp);
    fclose(fp);

    if (read != num_elements) {
        fprintf(stderr, "File read size mismatch\n");
        exit(1);
    }

    return data;
}