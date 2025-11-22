#ifndef UTILS_H
#define UTILS_H
#endif 
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <complex>


float* load_bin(const char* filename, size_t num_elements);

__device__ float2 qpsk_mod(int bits);

__device__ int qpsk_demod(float2 s);

std::complex<double>* load_complex128(const char* filename, size_t num_elements)