#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cassert>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA:%s:%d: error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } }

// ==== DATASET CONSTANTS (must match dataset.cu) ====
const int N = 30000; // total samples
const int Nr = 4;
const int Nt = 4;

// ==== MODEL / TRAINING HYPERPARAMS ====
const int HIDDEN = 512;
const int BATCH = 256;
const float LR = 1e-3f;
const int EPOCHS = 12;

// Derived dims
const int IN_DIM = 2*Nr + 2*Nr*Nt; // flatten(Y) + flatten(H)
const int OUT_DIM = 2*Nt;          // predict real & imag for each transmit antenna

// ----------------- GPU kernels -----------------

// Simple row-major matrix multiply: C = A (m x k) * B (k x n) -> C (m x n)
// Blocked kernel (tile size TILE x TILE)
const int TILE = 16;
__global__ void matmul_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    // A: m x k, B: k x n, C: m x n (row-major)
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
        __shared__ float sA[TILE][TILE];
        __shared__ float sB[TILE][TILE];

        int aRow = row;
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        int bCol = col;

        float aval = 0.0f;
        if (aRow < m && aCol < k) aval = A[aRow * k + aCol];
        float bval = 0.0f;
        if (bRow < k && bCol < n) bval = B[bRow * n + bCol];

        sA[threadIdx.y][threadIdx.x] = aval;
        sB[threadIdx.y][threadIdx.x] = bval;

        __syncthreads();

        for (int kk = 0; kk < TILE; ++kk) {
            acc += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = acc;
    }
}

// C = A + b (row-wise): add biases to each row element: for each row r and col c: C[r,c] = A[r,c] + bias[c]
__global__ void add_bias_kernel(float* A, const float* bias, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        A[row * n + col] += bias[col];
    }
}

// ReLU in-place
__global__ void relu_inplace_kernel(float* A, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        if (A[i] < 0.0f) A[i] = 0.0f;
    }
}

// ReLU backward: grad_input = grad_output * (A > 0)
__global__ void relu_backward_kernel(const float* A, float* grad, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        if (A[i] <= 0.0f) grad[i] = 0.0f;
    }
}

// C = A^T * B   with A: (m x k), B: (m x n) => C (k x n)
__global__ void matmul_AT_B_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    // compute C (k x n)
    int row = blockIdx.y * TILE + threadIdx.y; // row in C: 0..k-1
    int col = blockIdx.x * TILE + threadIdx.x; // col in C: 0..n-1
    float acc = 0.0f;
    for (int t = 0; t < (m + TILE - 1) / TILE; ++t) {
        __shared__ float sA[TILE][TILE];
        __shared__ float sB[TILE][TILE];

        int aRow = t * TILE + threadIdx.y; // in A: row aRow (0..m-1)
        int aCol = row;                     // col in A: 0..k-1 (we'll index A[aRow,kcol])
        int bRow = t * TILE + threadIdx.x; // in B: row bRow
        int bCol = col;

        float aval = 0.0f;
        if (aRow < m && aCol < k) aval = A[aRow * k + aCol];
        float bval = 0.0f;
        if (bRow < m && bCol < n) bval = B[bRow * n + bCol];

        sA[threadIdx.y][threadIdx.x] = aval;
        sB[threadIdx.y][threadIdx.x] = bval;
        __syncthreads();

        for (int kk = 0; kk < TILE; ++kk) {
            acc += sA[kk][threadIdx.x] * sB[kk][threadIdx.y]; // careful indexing
        }
        __syncthreads();
    }
    if (row < k && col < n) {
        C[row * n + col] = acc;
    }
}

// Compute MSE gradient for output: grad = 2*(pred - target)/B
__global__ void mse_grad_kernel(const float* pred, const float* target, float* grad, int B, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * out_dim;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        grad[i] = 2.0f * (pred[i] - target[i]) / float(B);
    }
}

// Update weights: W = W - lr * grad, W (rows x cols)
__global__ void sgd_update_kernel(float* W, const float* grad, float lr, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        W[i] -= lr * grad[i];
    }
}

// Update biases: bias = bias - lr * grad (grad shape cols)
__global__ void sgd_update_bias_kernel(float* b, const float* grad_b, float lr, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < cols; i += blockDim.x * gridDim.x) {
        b[i] -= lr * grad_b[i];
    }
}

// A utility host function to launch matmul (A mxk) * (k xn) -> (m xn)
void gpu_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1)/TILE, (m + TILE - 1)/TILE);
    matmul_kernel<<<grid, block>>>(A, B, C, m, n, k);
    CHECK_CUDA(cudaPeekAtLastError());
}

// launch A^T * B kernel: A (m x k), B (m x n) -> C (k x n)
void gpu_matmul_AT_B(const float* A, const float* B, float* C, int m, int k, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1)/TILE, (k + TILE - 1)/TILE);
    matmul_AT_B_kernel<<<grid, block>>>(A, B, C, m, k, n);
    CHECK_CUDA(cudaPeekAtLastError());
}

// ----------------- Helper host utilities -----------------
float* host_alloc_and_load(const char* filename, size_t float_count) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", filename); return nullptr; }
    float* buf = (float*)malloc(sizeof(float) * float_count);
    size_t read = fread(buf, sizeof(float), float_count, f);
    fclose(f);
    if (read != float_count) {
        fprintf(stderr, "Read mismatch for %s: expected %zu floats, got %zu\n", filename, float_count, read);
        free(buf);
        return nullptr;
    }
    return buf;
}

// Compute nearest QPSK decisions and accuracy on host arrays (predicted and gold are floats in row-major with shape B x OUT_DIM)
float compute_qpsk_accuracy_host(const float* pred, const float* gold, int B) {
    // QPSK points: (1,1), (1,-1), (-1,-1), (-1,1)  (real, imag)
    const float qpsk[4][2] = { {1.0f,1.0f}, {1.0f,-1.0f}, {-1.0f,-1.0f}, {-1.0f,1.0f} };
    int correct = 0;
    for (int b=0;b<B;b++) {
        for (int t=0;t<Nt;t++) {
            float pr = pred[b*OUT_DIM + 2*t + 0];
            float pi = pred[b*OUT_DIM + 2*t + 1];
            int best = 0; float bestd = 1e30f;
            for (int q=0;q<4;q++) {
                float dr = pr - qpsk[q][0];
                float di = pi - qpsk[q][1];
                float d = dr*dr + di*di;
                if (d < bestd) { bestd = d; best = q; }
            }
            // gold class
            float gr = gold[b*OUT_DIM + 2*t + 0];
            float gi = gold[b*OUT_DIM + 2*t + 1];
            int gbest = 0; float gbestd = 1e30f;
            for (int q=0;q<4;q++) {
                float dr = gr - qpsk[q][0];
                float di = gi - qpsk[q][1];
                float d = dr*dr + di*di;
                if (d < gbestd) { gbestd = d; gbest = q; }
            }
            if (best == gbest) correct++;
        }
    }
    return float(correct) / float(B*Nt);
}

// ----------------- Main training entry -----------------
int main(int argc, char** argv) {
    printf("DDN CUDA trainer starting...\n");
    // File paths (dataset.cu writes these)
    const char* H_file = "data/H.bin";
    const char* X_file = "data/X.bin";
    const char* Y_file = "data/Y.bin";

    // expected float counts
    size_t H_floats = (size_t)N * Nr * Nt * 2;
    size_t X_floats = (size_t)N * Nt * 2;
    size_t Y_floats = (size_t)N * Nr * 2;

    float* H_host = host_alloc_and_load(H_file, H_floats);
    float* X_host = host_alloc_and_load(X_file, X_floats);
    float* Y_host = host_alloc_and_load(Y_file, Y_floats);
    if (!H_host || !X_host || !Y_host) {
        fprintf(stderr, "Failed to load dataset. Exiting.\n");
        return 1;
    }

    // allocate device buffers for dataset (we'll copy minibatches to smaller buffers for training)
    float *d_batch_in = nullptr;   // batch x IN_DIM
    float *d_batch_out = nullptr;  // batch x OUT_DIM
    CHECK_CUDA(cudaMalloc(&d_batch_in, sizeof(float) * BATCH * IN_DIM));
    CHECK_CUDA(cudaMalloc(&d_batch_out, sizeof(float) * BATCH * OUT_DIM));

    // model parameters: weights are row-major: W (rows x cols): rows = units_in, cols = units_out? we will use (m x n) meaning (batch_row x out)
    // We'll implement as: Layer1: W1 (IN_DIM x HIDDEN), b1 (HIDDEN), Layer2: W2 (HIDDEN x OUT_DIM), b2 (OUT_DIM)
    float *W1, *b1, *W2, *b2;
    CHECK_CUDA(cudaMalloc(&W1, sizeof(float) * IN_DIM * HIDDEN));
    CHECK_CUDA(cudaMalloc(&b1, sizeof(float) * HIDDEN));
    CHECK_CUDA(cudaMalloc(&W2, sizeof(float) * HIDDEN * OUT_DIM));
    CHECK_CUDA(cudaMalloc(&b2, sizeof(float) * OUT_DIM));

    // gradients storage
    float *d_z1, *d_a1, *d_out, *d_loss_grad; // z1 = input*W1 + b1 ; a1 = relu(z1); out = a1*W2+b2
    CHECK_CUDA(cudaMalloc(&d_z1, sizeof(float)*BATCH*HIDDEN));
    CHECK_CUDA(cudaMalloc(&d_a1, sizeof(float)*BATCH*HIDDEN));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)*BATCH*OUT_DIM));
    CHECK_CUDA(cudaMalloc(&d_loss_grad, sizeof(float)*BATCH*OUT_DIM));

    // grads for weights
    float *d_dW2, *d_db2, *d_dW1, *d_db1;
    CHECK_CUDA(cudaMalloc(&d_dW2, sizeof(float)*HIDDEN*OUT_DIM));
    CHECK_CUDA(cudaMalloc(&d_db2, sizeof(float)*OUT_DIM));
    CHECK_CUDA(cudaMalloc(&d_dW1, sizeof(float)*IN_DIM*HIDDEN));
    CHECK_CUDA(cudaMalloc(&d_db1, sizeof(float)*HIDDEN));

    // helper buffers for matmul intermediate grads
    float *d_tmp; // e.g., for computing grad_hidden = grad_out * W2^T -> (BATCH x HIDDEN)
    CHECK_CUDA(cudaMalloc(&d_tmp, sizeof(float)*BATCH*HIDDEN));

    // Initialize weights on host and copy
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 0.1f);

    std::vector<float> hW1(IN_DIM * HIDDEN);
    std::vector<float> hb1(HIDDEN, 0.0f);
    std::vector<float> hW2(HIDDEN * OUT_DIM);
    std::vector<float> hb2(OUT_DIM, 0.0f);

    for (size_t i=0;i<hW1.size();++i) hW1[i] = nd(rng) * sqrtf(2.0f / (IN_DIM + HIDDEN));
    for (size_t i=0;i<hW2.size();++i) hW2[i] = nd(rng) * sqrtf(2.0f / (HIDDEN + OUT_DIM));

    CHECK_CUDA(cudaMemcpy(W1, hW1.data(), sizeof(float)*hW1.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b1, hb1.data(), sizeof(float)*hb1.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(W2, hW2.data(), sizeof(float)*hW2.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b2, hb2.data(), sizeof(float)*hb2.size(), cudaMemcpyHostToDevice));

    // host buffers for loss/acc reporting
    float* h_pred = (float*)malloc(sizeof(float)*BATCH*OUT_DIM);
    float* h_gold = (float*)malloc(sizeof(float)*BATCH*OUT_DIM);

    // Training loop
    int steps_per_epoch = (N + BATCH - 1) / BATCH;
    printf("Training: epochs=%d, batch=%d, steps_per_epoch=%d\n", EPOCHS, BATCH, steps_per_epoch);

    for (int epoch=0; epoch < EPOCHS; ++epoch) {
        // shuffle indices
        std::vector<int> idx(N);
        for (int i=0;i<N;i++) idx[i]=i;
        std::shuffle(idx.begin(), idx.end(), rng);

        double epoch_loss = 0.0;
        double epoch_acc = 0.0;
        int processed = 0;

        for (int step = 0; step < steps_per_epoch; ++step) {
            int offset = step * BATCH;
            int curB = std::min(BATCH, N - offset);

            // build batch inputs on host (flatten Y and H into IN_DIM)
            // We'll pack per sample: [Y_re(0..Nr-1), Y_im(...), H_re(... flattened Nr*Nt), H_im(...)]
            std::vector<float> host_in(curB * IN_DIM);
            std::vector<float> host_out(curB * OUT_DIM);

            for (int bi = 0; bi < curB; ++bi) {
                int sample = idx[offset + bi];
                // Y: sample * (Nr*2) floats, arranged as [re,im, re,im,...] length Nr*2
                size_t Ypos = (size_t)sample * Nr * 2;
                // H: sample * (Nr*Nt*2)
                size_t Hpos = (size_t)sample * Nr * Nt * 2;
                // X: sample * (Nt*2)
                size_t Xpos = (size_t)sample * Nt * 2;

                // fill Y real parts then imag parts (we'll choose layout: [re0,re1,...,im0,im1,...] for Y)
                for (int r = 0; r < Nr; ++r) {
                    host_in[bi*IN_DIM + (0 + r)] = Y_host[Ypos + 2*r + 0]; // re
                }
                for (int r = 0; r < Nr; ++r) {
                    host_in[bi*IN_DIM + (Nr + r)] = Y_host[Ypos + 2*r + 1]; // im
                }
                // Fill H: first all re's for (Nr*Nt) then all im's
                int Hre_off = 2*Nr; // after Y real+imag
                for (int rr = 0; rr < Nr; ++rr) {
                    for (int tt = 0; tt < Nt; ++tt) {
                        int idxH = rr*Nt + tt;
                        host_in[bi*IN_DIM + (Hre_off + idxH)] = H_host[Hpos + 2*idxH + 0]; // re
                    }
                }
                int Him_off = Hre_off + Nr*Nt;
                for (int rr = 0; rr < Nr; ++rr) {
                    for (int tt = 0; tt < Nt; ++tt) {
                        int idxH = rr*Nt + tt;
                        host_in[bi*IN_DIM + (Him_off + idxH)] = H_host[Hpos + 2*idxH + 1]; // im
                    }
                }
                // target X: store [re0,im0,re1,im1,...] (OUT_DIM ordering consistent with earlier notebook)
                for (int t=0;t<Nt;t++){
                    host_out[bi*OUT_DIM + 2*t + 0] = X_host[Xpos + 2*t + 0];
                    host_out[bi*OUT_DIM + 2*t + 1] = X_host[Xpos + 2*t + 1];
                }
            }

            // copy to device batch buffers
            CHECK_CUDA(cudaMemcpy(d_batch_in, host_in.data(), sizeof(float)*curB*IN_DIM, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_batch_out, host_out.data(), sizeof(float)*curB*OUT_DIM, cudaMemcpyHostToDevice));

            // Forward pass:
            // z1 = XW1
            gpu_matmul(d_batch_in, W1, d_z1, curB, HIDDEN, IN_DIM);

            // add bias b1
            {
                dim3 block(16,16);
                dim3 grid((HIDDEN+15)/16, (curB+15)/16);
                add_bias_kernel<<<grid,block>>>(d_z1, b1, curB, HIDDEN);
                CHECK_CUDA(cudaPeekAtLastError());
            }

            // a1 = relu(z1)
            {
                int threads = 256;
                int blocks = (curB*HIDDEN + threads-1)/threads;
                relu_inplace_kernel<<<blocks,threads>>>(d_z1, curB, HIDDEN);
                CHECK_CUDA(cudaPeekAtLastError());
            }
            // a1 buffer is just z1 after relu
            CHECK_CUDA(cudaMemcpy(d_a1, d_z1, sizeof(float)*curB*HIDDEN, cudaMemcpyDeviceToDevice));

            // out = a1 (curB x HIDDEN) * W2 (HIDDEN x OUT_DIM)
            gpu_matmul(d_a1, W2, d_out, curB, OUT_DIM, HIDDEN);

            // add bias b2
            {
                dim3 block(16,16);
                dim3 grid((OUT_DIM+15)/16, (curB+15)/16);
                add_bias_kernel<<<grid,block>>>(d_out, b2, curB, OUT_DIM);
                CHECK_CUDA(cudaPeekAtLastError());
            }

            // Compute loss grad: dL/d(out) = 2*(pred - target)/B
            {
                int threads = 256;
                int blocks = (curB*OUT_DIM + threads-1)/threads;
                mse_grad_kernel<<<blocks,threads>>>(d_out, d_batch_out, d_loss_grad, curB, OUT_DIM);
                CHECK_CUDA(cudaPeekAtLastError());
            }

            // --- BACKWARD PASS ---
            // dW2 = a1^T (HIDDEN x curB) * d_loss_grad(curB x OUT_DIM) → HIDDEN x OUT_DIM
            gpu_matmul_AT_B(d_a1, d_loss_grad, d_dW2, curB, HIDDEN, OUT_DIM);

            // db2 = sum rows of d_loss_grad (curB x OUT_DIM)
            {
                std::vector<float> host_db2(OUT_DIM, 0.0f);
                std::vector<float> tmp_loss(curB * OUT_DIM);
                CHECK_CUDA(cudaMemcpy(tmp_loss.data(), d_loss_grad,
                                      sizeof(float)*curB*OUT_DIM,
                                      cudaMemcpyDeviceToHost));
                for (int r=0;r<curB;r++)
                    for (int c=0;c<OUT_DIM;c++)
                        host_db2[c] += tmp_loss[r*OUT_DIM+c];
                CHECK_CUDA(cudaMemcpy(d_db2, host_db2.data(),
                                      sizeof(float)*OUT_DIM,
                                      cudaMemcpyHostToDevice));
            }

            // grad_hidden = d_loss_grad * W2^T → (curB x HIDDEN)
            // reuse d_tmp as gradient of hidden layer
            gpu_matmul(d_loss_grad, W2, d_tmp, curB, HIDDEN, OUT_DIM); // (curB x OUT_DIM)*(OUT_DIM x HIDDEN)

            // relu backward: zero-out where z1 <= 0
            {
                int threads = 256;
                int blocks = (curB*HIDDEN + threads-1)/threads;
                relu_backward_kernel<<<blocks,threads>>>(d_z1, d_tmp, curB, HIDDEN);
                CHECK_CUDA(cudaPeekAtLastError());
            }

            // dW1 = (batch_in)^T * grad_hidden → IN_DIM x HIDDEN
            gpu_matmul_AT_B(d_batch_in, d_tmp, d_dW1, curB, IN_DIM, HIDDEN);

            // db1 = sum rows of grad_hidden
            {
                std::vector<float> host_db1(HIDDEN, 0.0f);
                std::vector<float> tmp_grad(curB * HIDDEN);
                CHECK_CUDA(cudaMemcpy(tmp_grad.data(), d_tmp,
                                      sizeof(float)*curB*HIDDEN,
                                      cudaMemcpyDeviceToHost));
                for (int r=0;r<curB;r++)
                    for (int c=0;c<HIDDEN;c++)
                        host_db1[c] += tmp_grad[r*HIDDEN + c];
                CHECK_CUDA(cudaMemcpy(d_db1, host_db1.data(),
                                      sizeof(float)*HIDDEN,
                                      cudaMemcpyHostToDevice));
            }

            // --- UPDATE ---
            {
                int threads = 256;

                // update W2
                int blocks_W2 = (HIDDEN*OUT_DIM + threads-1)/threads;
                sgd_update_kernel<<<blocks_W2, threads>>>(W2, d_dW2, LR, HIDDEN, OUT_DIM);

                // update b2
                int blocks_b2 = (OUT_DIM + threads-1)/threads;
                sgd_update_bias_kernel<<<blocks_b2, threads>>>(b2, d_db2, LR, OUT_DIM);

                // update W1
                int blocks_W1 = (IN_DIM*HIDDEN + threads-1)/threads;
                sgd_update_kernel<<<blocks_W1, threads>>>(W1, d_dW1, LR, IN_DIM, HIDDEN);

                // update b1
                int blocks_b1 = (HIDDEN + threads-1)/threads;
                sgd_update_bias_kernel<<<blocks_b1, threads>>>(b1, d_db1, LR, HIDDEN);
            }

            // fetch predictions for accuracy
            CHECK_CUDA(cudaMemcpy(h_pred, d_out, sizeof(float)*curB*OUT_DIM, cudaMemcpyDeviceToHost));
            memcpy(h_gold, host_out.data(), sizeof(float)*curB*OUT_DIM);

            float acc = compute_qpsk_accuracy_host(h_pred, h_gold, curB);
            epoch_acc += acc * curB;

            // simple MSE loss for reporting
            double batch_loss = 0.0;
            for (int i=0;i<curB*OUT_DIM;i++){
                float diff = h_pred[i] - h_gold[i];
                batch_loss += diff*diff;
            }
            batch_loss /= curB;
            epoch_loss += batch_loss;

            processed += curB;

            if (step % 20 == 0)
                printf("Epoch %d step %d/%d  loss=%.4f  acc=%.4f\n",
                       epoch, step, steps_per_epoch,
                       batch_loss, acc);
        }

        printf("==> Epoch %d DONE  avg_loss=%.4f avg_acc=%.4f\n",
               epoch,
               epoch_loss/steps_per_epoch,
               epoch_acc/(processed * 1.0 / Nt));
    }

    printf("Training complete.\n");
    return 0;
}
