// model.cu
// Compile with: nvcc -O3 model.cu -o model
// Usage: ./model [epochs] [hidden_size] [lr]
// Expects x.bin, y.bin, h.bin in current working directory, float32, shapes auto-detected.
// Assumptions: x.bin: Nsamples * Nt floats (transmitted symbols, e.g. BPSK ±1).
//              y.bin: Nsamples * Nr floats (received vector).
//              h.bin: Nsamples * (Nt*Nr) floats (channel matrices stacked per-sample).
// The program trains a small MLP (y -> hidden -> x_hat) with SGD on GPU and reports MSE and BER.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <cassert>
#include <sys/stat.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

size_t file_size_bytes(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return st.st_size;
}

std::vector<float> load_bin(const char* path) {
    FILE* f = fopen(path,"rb");
    if(!f){ fprintf(stderr,"Failed to open %s\n", path); exit(1); }
    size_t sz = file_size_bytes(path);
    if(sz % sizeof(float) != 0){ fprintf(stderr,"File size not multiple of float: %s\n", path); exit(1); }
    size_t n = sz / sizeof(float);
    std::vector<float> v(n);
    size_t r = fread(v.data(), sizeof(float), n, f);
    if (r != n) { fprintf(stderr,"Read error %s\n", path); exit(1); }
    fclose(f);
    return v;
}

// gcd for size_t
size_t gcd_size(size_t a, size_t b) {
    while (b) { size_t t = a % b; a = b; b = t; }
    return a;
}

// Kernels

// hidden (Ns x H) = y (Ns x Nr) * W1^T (Nr x H) + b1
__global__ void compute_hidden(const float* __restrict__ y, const float* __restrict__ W1, const float* __restrict__ b1, float* __restrict__ hidden, int Ns, int Nr, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ns * H;
    if (idx >= total) return;
    int s = idx / H;
    int h = idx % H;
    const float* y_s = y + (size_t)s * Nr;
    const float* W1_h = W1 + (size_t)h * Nr; // W1 layout: H x Nr
    float sum = b1[h];
    for (int r = 0; r < Nr; ++r) sum += y_s[r] * W1_h[r];
    // ReLU
    hidden[(size_t)s * H + h] = sum > 0.0f ? sum : 0.0f;
}

// out (Ns x Nt) = hidden (Ns x H) * W2^T (H x Nt) + b2
__global__ void compute_out(const float* __restrict__ hidden, const float* __restrict__ W2, const float* __restrict__ b2, float* __restrict__ out, int Ns, int H, int Nt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ns * Nt;
    if (idx >= total) return;
    int s = idx / Nt;
    int t = idx % Nt;
    const float* hidden_s = hidden + (size_t)s * H;
    const float* W2_t = W2 + (size_t)t * H; // W2 layout: Nt x H
    float sum = b2[t];
    for (int h = 0; h < H; ++h) sum += hidden_s[h] * W2_t[h];
    out[(size_t)s * Nt + t] = sum;
}

// compute d_out = 2*(out - x) / Ns  (in-place into dout)
__global__ void compute_dout(const float* __restrict__ out, const float* __restrict__ x, float* __restrict__ dout, int NsNt, int Ns) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NsNt) return;
    dout[idx] = 2.0f * (out[idx] - x[idx]) / float(Ns);
}

// grad_W2 (Nt x H) = sum_s dout[s,t] * hidden[s,h]
__global__ void grad_W2_kernel(const float* __restrict__ dout, const float* __restrict__ hidden, float* __restrict__ gradW2, int Ns, int Nt, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nt * H;
    if (idx >= total) return;
    int t = idx / H;
    int h = idx % H;
    float acc = 0.0f;
    for (int s = 0; s < Ns; ++s) acc += dout[(size_t)s * Nt + t] * hidden[(size_t)s * H + h];
    gradW2[(size_t)t * H + h] = acc;
}

// d_hidden (Ns x H) = dout (Ns x Nt) * W2 (Nt x H)
__global__ void d_hidden_kernel(const float* __restrict__ dout, const float* __restrict__ W2, float* __restrict__ dh, int Ns, int Nt, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Ns * H;
    if (idx >= total) return;
    int s = idx / H;
    int h = idx % H;
    float acc = 0.0f;
    for (int t = 0; t < Nt; ++t) acc += dout[(size_t)s * Nt + t] * W2[(size_t)t * H + h];
    dh[(size_t)s * H + h] = acc;
}

// Apply ReLU derivative: dh *= (hidden > 0)
__global__ void relu_backward(float* __restrict__ dh, const float* __restrict__ hidden, int NsH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NsH) return;
    if (hidden[idx] <= 0.0f) dh[idx] = 0.0f;
}

// grad_W1 (H x Nr) = sum_s dh[s,h] * y[s,r]
__global__ void grad_W1_kernel(const float* __restrict__ dh, const float* __restrict__ y, float* __restrict__ gradW1, int Ns, int H, int Nr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * Nr;
    if (idx >= total) return;
    int h = idx / Nr;
    int r = idx % Nr;
    float acc = 0.0f;
    for (int s = 0; s < Ns; ++s) acc += dh[(size_t)s * H + h] * y[(size_t)s * Nr + r];
    gradW1[(size_t)h * Nr + r] = acc;
}

// bias grads: grad_b2[t] = sum_s dout[s,t]; grad_b1[h] = sum_s dh[s,h]
__global__ void grad_b2_kernel(const float* __restrict__ dout, float* __restrict__ gradb2, int Ns, int Nt) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Nt) return;
    float acc = 0.0f;
    for (int s = 0; s < Ns; ++s) acc += dout[(size_t)s * Nt + t];
    gradb2[t] = acc;
}
__global__ void grad_b1_kernel(const float* __restrict__ dh, float* __restrict__ gradb1, int Ns, int H) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float acc = 0.0f;
    for (int s = 0; s < Ns; ++s) acc += dh[(size_t)s * H + h];
    gradb1[h] = acc;
}

// update weights: W -= lr * grad
__global__ void update_params(float* __restrict__ W, const float* __restrict__ grad, int N, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    W[idx] -= lr * grad[idx];
}

// compute metrics on CPU (copy out)
float compute_mse_cpu(const std::vector<float>& out, const std::vector<float>& x) {
    size_t N = out.size();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double d = out[i] - x[i];
        acc += d*d;
    }
    return float(acc / N);
}
float compute_ber_cpu(const std::vector<float>& out, const std::vector<float>& x) {
    size_t N = out.size();
    size_t bits = 0;
    for (size_t i = 0; i < N; ++i) {
        int a = out[i] >= 0.0f;
        int b = x[i] >= 0.0f;
        if (a != b) ++bits;
    }
    return float(bits) / float(N);
}

int main(int argc, char** argv) {
    int epochs = 20;
    int H = 64;
    float lr = 0.01f;
    if (argc > 1) epochs = atoi(argv[1]);
    if (argc > 2) H = atoi(argv[2]);
    if (argc > 3) lr = atof(argv[3]);

    auto x = load_bin("X.bin");
    auto y = load_bin("Y.bin");
    auto h = load_bin("H.bin");

    size_t len_x = x.size(), len_y = y.size(), len_h = h.size();
    if (!len_x || !len_y || !len_h) { fprintf(stderr,"Empty input files\n"); return 1; }

    size_t Ns = 0;
    for(size_t cand = 1; cand <= len_h; cand++) {
        if(len_h % cand) continue;
        if(len_x % cand) continue;
        if(len_y % cand) continue;

        size_t Nt = len_x / cand;
        size_t Nr = len_y / cand;
        size_t Hs = len_h / cand;

        if(Hs == Nt * Nr) {
            Ns = cand;
            break;
        }
    }

    if (!Ns) { fprintf(stderr, "Could not deduce Ns\n"); return 1; }
    // Ns should divide both len_x and len_y; if gcd is larger than sample count it will still produce integer shapes
    size_t Nt = len_x / Ns;
    size_t Nr = len_y / Ns;
    size_t chan_size = len_h / Ns;
    if (chan_size != Nt * Nr) {
        // fallback: try Ns = gcd(len_x, len_y)
        Ns = gcd_size(len_x, len_y);
        if (Ns == 0) { fprintf(stderr,"Cannot deduce Ns\n"); return 1; }
        Nt = len_x / Ns; Nr = len_y / Ns; chan_size = len_h / Ns;
        if (chan_size != Nt*Nr) {
            fprintf(stderr,"Shape mismatch: len_x=%zu len_y=%zu len_h=%zu. Deduced Ns=%zu Nt=%zu Nr=%zu chan_size=%zu\n",
                    len_x, len_y, len_h, Ns, Nt, Nr, chan_size);
            return 1;
        }
    }

    printf("Detected Ns=%zu Nt=%zu Nr=%zu Hidden=%d epochs=%d lr=%g\n", Ns, Nt, Nr, H, epochs, lr);

    // Allocate device arrays
    float *d_y, *d_x, *d_h;
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)*len_y));
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float)*len_x));
    CUDA_CHECK(cudaMalloc(&d_h, sizeof(float)*len_h));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), sizeof(float)*len_y, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), sizeof(float)*len_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h, h.data(), sizeof(float)*len_h, cudaMemcpyHostToDevice));

    // Model parameters (device)
    size_t W1_size = H * Nr;
    size_t W2_size = Nt * H;
    std::vector<float> W1(W1_size), W2(W2_size), b1(H), b2(Nt);
    // init small random
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    for (size_t i=0;i<W1_size;++i) W1[i] = nd(rng);
    for (size_t i=0;i<W2_size;++i) W2[i] = nd(rng);
    for (int i=0;i<H;++i) b1[i]=0.0f;
    for (size_t i=0;i<(size_t)Nt;++i) b2[i]=0.0f;

    float *d_W1, *d_W2, *d_b1, *d_b2;
    CUDA_CHECK(cudaMalloc(&d_W1, sizeof(float)*W1_size));
    CUDA_CHECK(cudaMalloc(&d_W2, sizeof(float)*W2_size));
    CUDA_CHECK(cudaMalloc(&d_b1, sizeof(float)*H));
    CUDA_CHECK(cudaMalloc(&d_b2, sizeof(float)*Nt));
    CUDA_CHECK(cudaMemcpy(d_W1, W1.data(), sizeof(float)*W1_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2.data(), sizeof(float)*W2_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1.data(), sizeof(float)*H, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2.data(), sizeof(float)*Nt, cudaMemcpyHostToDevice));

    // buffers
    float *d_hidden, *d_out, *d_dout, *d_dh;
    CUDA_CHECK(cudaMalloc(&d_hidden, sizeof(float)*Ns*H));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)*Ns*Nt));
    CUDA_CHECK(cudaMalloc(&d_dout, sizeof(float)*Ns*Nt));
    CUDA_CHECK(cudaMalloc(&d_dh, sizeof(float)*Ns*H));

    // gradients
    float *d_gradW1, *d_gradW2, *d_gradb1, *d_gradb2;
    CUDA_CHECK(cudaMalloc(&d_gradW1, sizeof(float)*W1_size));
    CUDA_CHECK(cudaMalloc(&d_gradW2, sizeof(float)*W2_size));
    CUDA_CHECK(cudaMalloc(&d_gradb1, sizeof(float)*H));
    CUDA_CHECK(cudaMalloc(&d_gradb2, sizeof(float)*Nt));

    int threads = 256;

    for (int ep=0; ep<epochs; ++ep) {
        // forward
        int total_hidden = Ns * H;
        int blocks_h = (total_hidden + threads - 1) / threads;
        compute_hidden<<<blocks_h,threads>>>(d_y, d_W1, d_b1, d_hidden, (int)Ns, (int)Nr, H);
        CUDA_CHECK(cudaPeekAtLastError());

        int total_out = Ns * Nt;
        int blocks_o = (total_out + threads - 1) / threads;
        compute_out<<<blocks_o,threads>>>(d_hidden, d_W2, d_b2, d_out, (int)Ns, H, (int)Nt);
        CUDA_CHECK(cudaPeekAtLastError());

        // loss gradient
        int blocks_dout = (total_out + threads - 1) / threads;
        compute_dout<<<blocks_dout,threads>>>(d_out, d_x, d_dout, total_out, (int)Ns);
        CUDA_CHECK(cudaPeekAtLastError());

        // grad W2
        int total_gradW2 = Nt * H;
        int blocks_gW2 = (total_gradW2 + threads - 1) / threads;
        grad_W2_kernel<<<blocks_gW2,threads>>>(d_dout, d_hidden, d_gradW2, (int)Ns, (int)Nt, H);
        CUDA_CHECK(cudaPeekAtLastError());

        // d_hidden
        int total_dh = Ns * H;
        int blocks_dh = (total_dh + threads - 1) / threads;
        d_hidden_kernel<<<blocks_dh,threads>>>(d_dout, d_W2, d_dh, (int)Ns, (int)Nt, H);
        CUDA_CHECK(cudaPeekAtLastError());

        int blocks_relu = (total_dh + threads - 1) / threads;
        relu_backward<<<blocks_relu, threads>>>(d_dh, d_hidden, total_dh);
        CUDA_CHECK(cudaPeekAtLastError());

        // grad W1
        int total_gradW1 = H * Nr;
        int blocks_gW1 = (total_gradW1 + threads - 1) / threads;
        grad_W1_kernel<<<blocks_gW1,threads>>>(d_dh, d_y, d_gradW1, (int)Ns, H, (int)Nr);
        CUDA_CHECK(cudaPeekAtLastError());

        // grad biases
        int blocks_b2 = (Nt + threads - 1) / threads;
        grad_b2_kernel<<<blocks_b2, threads>>>(d_dout, d_gradb2, (int)Ns, (int)Nt);
        CUDA_CHECK(cudaPeekAtLastError());
        int blocks_b1 = (H + threads - 1) / threads;
        grad_b1_kernel<<<blocks_b1, threads>>>(d_dh, d_gradb1, (int)Ns, H);
        CUDA_CHECK(cudaPeekAtLastError());

        // normalize grads by Ns (already partly scaled by compute_dout). compute_dout scaled by 1/Ns, so grads are already averaged.
        // Update params: W -= lr * grad
        int blocks_up_W1 = (W1_size + threads - 1) / threads;
        update_params<<<blocks_up_W1, threads>>>(d_W1, d_gradW1, (int)W1_size, lr);
        int blocks_up_W2 = (W2_size + threads - 1) / threads;
        update_params<<<blocks_up_W2, threads>>>(d_W2, d_gradW2, (int)W2_size, lr);
        int blocks_up_b1 = (H + threads - 1) / threads;
        update_params<<<blocks_up_b1, threads>>>(d_b1, d_gradb1, H, lr);
        int blocks_up_b2 = (Nt + threads - 1) / threads;
        update_params<<<blocks_up_b2, threads>>>(d_b2, d_gradb2, (int)Nt, lr);
        CUDA_CHECK(cudaPeekAtLastError());

        // occasionally compute metrics (copy out)
        if ((ep % 5 == 0) || ep == epochs-1) {
            std::vector<float> out_host(len_x);
            CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, sizeof(float)*len_x, cudaMemcpyDeviceToHost));
            float mse = compute_mse_cpu(out_host, x);
            float ber = compute_ber_cpu(out_host, x);
            printf("Epoch %d/%d: MSE=%.6f BER=%.6f\n", ep+1, epochs, mse, ber);
        }
    }

    // final evaluate
    std::vector<float> out_host(len_x);
    CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, sizeof(float)*len_x, cudaMemcpyDeviceToHost));
    float mse = compute_mse_cpu(out_host, x);
    float ber = compute_ber_cpu(out_host, x);
    printf("Final: MSE=%.6f BER=%.6f\n", mse, ber);

    // Cleanup
    cudaFree(d_y); cudaFree(d_x); cudaFree(d_h);
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_b1); cudaFree(d_b2);
    cudaFree(d_hidden); cudaFree(d_out); cudaFree(d_dout); cudaFree(d_dh);
    cudaFree(d_gradW1); cudaFree(d_gradW2); cudaFree(d_gradb1); cudaFree(d_gradb2);

    return 0;
}