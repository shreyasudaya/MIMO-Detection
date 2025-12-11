import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import subprocess

# ==========================================
# PART 1: PYTHON TRAINING (Target >98%)
# ==========================================
print("\n=== STEP 1: TRAINING ROBUST MODEL (PYTHON) ===")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Nt, Nr, LAYERS, HIDDEN = 16, 16, 20, 256
BATCH_SIZE = 1000

class DetNetBlock(nn.Module):
    def __init__(self, in_d, h_d, out_d):
        super().__init__()
        self.fc1 = nn.Linear(in_d, h_d)
        self.ln = nn.LayerNorm(h_d)
        self.fc2 = nn.Linear(h_d, out_d)
    def forward(self, x):
        return self.fc2(self.ln(torch.relu(self.fc1(x))))

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([DetNetBlock(4*Nt, HIDDEN, 2*Nt) for _ in range(LAYERS)])
    def forward(self, y, H):
        H_conj = torch.conj(torch.transpose(H, 1, 2))
        x = torch.bmm(H_conj, y.unsqueeze(2)).squeeze(2) / Nr
        for i in range(LAYERS):
            res = y - torch.bmm(H, x.unsqueeze(2)).squeeze(2)
            grad = torch.bmm(H_conj, res.unsqueeze(2)).squeeze(2)
            feat = torch.cat([x.real, x.imag, grad.real, grad.imag], dim=1)
            delta = self.blocks[i](feat)
            x = x + 0.5 * torch.complex(delta[:,:Nt], delta[:,Nt:])
        return x

def gen_data(bs, snr):
    scale = 10.0**(-snr/20.0)
    H = torch.randn(bs,Nr,Nt,dtype=torch.cfloat,device=DEVICE)*0.7071
    # Ensure x is (bs, Nt) so unsq(2) works
    x = torch.complex((2*torch.randint(0,2,(bs,Nt),device=DEVICE).float()-1)*0.7071,
                      (2*torch.randint(0,2,(bs,Nt),device=DEVICE).float()-1)*0.7071)
    n = torch.randn(bs,Nr,dtype=torch.cfloat,device=DEVICE)*0.7071*scale
    return torch.bmm(H,x.unsqueeze(2)).squeeze(2)+n, H, x

model = Detector().to(DEVICE)
opt = optim.Adam(model.parameters(), lr=0.001)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10)
crit = nn.MSELoss()

# Train 
for ep in range(601):
    model.train()
    y, H, x = gen_data(BATCH_SIZE, 20.0 + torch.rand(1).item()*5.0)
    loss = crit(torch.view_as_real(model(y,H)), torch.view_as_real(x))
    opt.zero_grad(); loss.backward(); opt.step()
    
    if ep % 50 == 0:
        model.eval()
        with torch.no_grad():
            y_t, H_t, x_t = gen_data(BATCH_SIZE, 20.0)
            acc = 100*(1-torch.mean(torch.abs((model(y_t,H_t).real>0).float()-(x_t.real>0).float())).item())
            print(f"Epoch {ep} | 20dB Acc: {acc:.2f}%")
            if acc > 97.5: break

print("Exporting Weights...")
checksum = 0.0
with open("mimo_weights.bin", "wb") as f:
    for b in model.blocks:
        for p in [b.fc1.weight, b.fc1.bias, b.ln.weight, b.ln.bias, b.fc2.weight, b.fc2.bias]:
            d = p.detach().cpu().numpy().astype(np.float32)
            checksum += np.sum(np.abs(d))
            f.write(d.tobytes())
print(f"weights.bin created. CHECKSUM: {checksum:.2f}")

# ==========================================
# PART 2: GENERATE C++ SOURCE
# ==========================================
print("\n=== STEP 2: GENERATING C++ CODE ===")
cpp_code = r"""
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand.h>
#include <fstream>
#include <cmath>
#include <iomanip>

#define CHECK(call) { if(call != 0) { printf("CUDA Error\n"); exit(1); } }

const int Nt=16, Nr=16, LAYERS=20, HIDDEN=256, BATCH=1000;

__global__ void kGen(float* y, float* H, float* x, float* n, float sigma) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < BATCH*Nr) {
        int b=i/Nr, r=i%Nr; float sr=0, si=0;
        for(int c=0; c<Nt; c++) {
            float hr=H[b*2*Nr*Nt + r*Nt + c], hi=H[b*2*Nr*Nt + Nr*Nt + r*Nt + c];
            float xr=x[b*2*Nt + c], xi=x[b*2*Nt + Nt + c];
            sr += hr*xr - hi*xi; si += hr*xi + hi*xr;
        }
        y[b*2*Nr+r] = sr + n[b*2*Nr+r]*sigma;
        y[b*2*Nr+Nr+r] = si + n[b*2*Nr+Nr+r]*sigma;
    }
}

__global__ void kInit(float* H, float* x, float* n, int size_H, int size_x, int size_n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<size_H) H[i] *= 0.70710678f;
    if(i<size_x) x[i] = (x[i]>0)? 0.7071f : -0.7071f;
}

__global__ void kMIMO(float* y, float* H, float* x, int mode) { 
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < BATCH*Nt) {
        int b=i/Nt, c=i%Nt; float sr=0, si=0;
        for(int r=0; r<Nr; r++) {
            float hr=H[b*2*Nr*Nt + r*Nt + c], hi=H[b*2*Nr*Nt + Nr*Nt + r*Nt + c];
            float inr, ini;
            if(mode==0) { inr=y[b*2*Nr+r]; ini=y[b*2*Nr+Nr+r]; } 
            else { inr=x[b*2*Nr+r]; ini=x[b*2*Nr+Nr+r]; }
            sr += hr*inr + hi*ini; si += hr*ini - hi*inr;
        }
        if(mode==0) { x[b*2*Nt+c]=sr/Nr; x[b*2*Nt+Nt+c]=si/Nr; }
        else { y[b*2*Nt+c]=sr; y[b*2*Nt+Nt+c]=si; } 
    }
}

__global__ void kMM(float* y, float* H, float* x) { 
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < BATCH*Nr) {
        int b=i/Nr, r=i%Nr; float sr=0, si=0;
        for(int c=0; c<Nt; c++) {
            float hr=H[b*2*Nr*Nt + r*Nt + c], hi=H[b*2*Nr*Nt + Nr*Nt + r*Nt + c];
            float xr=x[b*2*Nt + c], xi=x[b*2*Nt + Nt + c];
            sr += hr*xr - hi*xi; si += hr*xi + hi*xr;
        }
        y[b*2*Nr+r]=sr; y[b*2*Nr+Nr+r]=si;
    }
}

__global__ void kRes(float* r, float* y, float* Hx) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; 
    if(i<BATCH*2*Nr) r[i] = y[i] - Hx[i];
}

__global__ void kFeat(float* f, float* x, float* g) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<BATCH*Nt) {
        int b=i/Nt, c=i%Nt;
        f[b*4*Nt+c] = x[b*2*Nt+c]; f[b*4*Nt+Nt+c] = x[b*2*Nt+Nt+c];
        f[b*4*Nt+2*Nt+c] = g[b*2*Nt+c]; f[b*4*Nt+3*Nt+c] = g[b*2*Nt+Nt+c];
    }
}

__global__ void kLin(float* y, float* x, float* w, float* b_bias, int in_d, int out_d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<BATCH*out_d) {
        int b=i/out_d, o=i%out_d; float s=0;
        for(int j=0; j<in_d; j++) s += x[b*in_d+j] * w[o*in_d+j];
        y[i] = s + b_bias[o];
    }
}

__global__ void kAct(float* x, int n) { 
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i] = fmaxf(0.0f, x[i]);
}

__global__ void kLN(float* x, float* g, float* b, int cols) {
    int bid=blockIdx.x;
    float sum=0, var=0;
    for(int i=threadIdx.x; i<cols; i+=blockDim.x) sum += x[bid*cols+i];
    float mean = sum/cols; 
    for(int i=threadIdx.x; i<cols; i+=blockDim.x) var += (x[bid*cols+i]-mean)*(x[bid*cols+i]-mean);
    float std = sqrtf(var/cols + 1e-5f);
    for(int i=threadIdx.x; i<cols; i+=blockDim.x) 
        x[bid*cols+i] = ((x[bid*cols+i]-mean)/std)*g[i] + b[i];
}

__global__ void kUpd(float* x, float* d, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i] += 0.5f * d[i];
}

int main() {
    std::cout << "--- C++ MIMO DETECTOR ---" << std::endl;
    std::ifstream f("mimo_weights.bin", std::ios::binary);
    if(!f) return -1;
    
    // Load Weights - UPDATED: Using HIDDEN constant to suppress warning
    float *w[LAYERS][6]; 
    int sz[] = {64*HIDDEN, HIDDEN, HIDDEN, HIDDEN, HIDDEN*32, 32};
    double sum=0;
    for(int l=0; l<LAYERS; l++) {
        for(int j=0; j<6; j++) {
            cudaMalloc(&w[l][j], sz[j]*4);
            std::vector<float> buf(sz[j]);
            f.read((char*)buf.data(), sz[j]*4);
            for(auto v:buf) sum+=std::abs(v);
            cudaMemcpy(w[l][j], buf.data(), sz[j]*4, cudaMemcpyHostToDevice);
        }
    }
    std::cout << "Weights Checksum: " << std::fixed << std::setprecision(2) << sum << std::endl;

    // Buffers - UPDATED: Using HIDDEN in allocations
    float *d_y, *d_H, *d_x, *d_n, *d_xh, *d_hx, *d_r, *d_g, *d_f, *d_t, *d_d;
    cudaMalloc(&d_y, BATCH*2*Nr*4); cudaMalloc(&d_H, BATCH*2*Nr*Nt*4);
    cudaMalloc(&d_x, BATCH*2*Nt*4); cudaMalloc(&d_n, BATCH*2*Nr*4);
    cudaMalloc(&d_xh, BATCH*2*Nt*4); cudaMalloc(&d_hx, BATCH*2*Nr*4);
    cudaMalloc(&d_r, BATCH*2*Nr*4); cudaMalloc(&d_g, BATCH*2*Nt*4);
    cudaMalloc(&d_f, BATCH*4*Nt*4); 
    cudaMalloc(&d_t, BATCH*HIDDEN*4); // Using HIDDEN constant here
    cudaMalloc(&d_d, BATCH*2*Nt*4);

    curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // UPDATED: Output Table Header
    std::cout << "\nSNR(dB) | BER        | Accuracy(%)" << std::endl;
    std::cout << "--------|------------|------------" << std::endl;

    for(float snr=0; snr<=20; snr+=2) {
        curandGenerateNormal(gen, d_H, BATCH*2*Nr*Nt, 0, 1);
        curandGenerateNormal(gen, d_x, BATCH*2*Nt, 0, 1);
        curandGenerateNormal(gen, d_n, BATCH*2*Nr, 0, 1);
        kInit<<<1000,256>>>(d_H, d_x, d_n, BATCH*2*Nr*Nt, BATCH*2*Nt, BATCH*2*Nr);
        
        float sig = sqrtf(0.5f * powf(10.0f, -snr/10.0f));
        kGen<<<1000,256>>>(d_y, d_H, d_x, d_n, sig);
        
        kMIMO<<<1000,256>>>(d_y, d_H, d_xh, 0);

        for(int l=0; l<LAYERS; l++) {
            kMM<<<1000,256>>>(d_hx, d_H, d_xh);
            kRes<<<1000,256>>>(d_r, d_y, d_hx);
            kMIMO<<<1000,256>>>(d_g, d_H, d_r, 1);
            kFeat<<<1000,256>>>(d_f, d_xh, d_g);
            
            // Using HIDDEN constant in kernel calls
            kLin<<<1000,256>>>(d_t, d_f, w[l][0], w[l][1], 64, HIDDEN);
            kAct<<<1000,256>>>(d_t, BATCH*HIDDEN);
            kLN<<<BATCH,1>>>(d_t, w[l][2], w[l][3], HIDDEN);
            kLin<<<1000,256>>>(d_d, d_t, w[l][4], w[l][5], HIDDEN, 32);
            kUpd<<<1000,256>>>(d_xh, d_d, BATCH*2*Nt);
        }

        std::vector<float> hx(BATCH*2*Nt), hh(BATCH*2*Nt);
        cudaMemcpy(hx.data(), d_x, BATCH*2*Nt*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hh.data(), d_xh, BATCH*2*Nt*4, cudaMemcpyDeviceToHost);
        int err=0;
        for(int i=0; i<BATCH*2*Nt; i++) if((hx[i]>0)!=(hh[i]>0)) err++;
        
        float ber = (float)err/(BATCH*2*Nt);
        float acc = 100.0f * (1.0f - ber);
        std::cout << std::fixed << std::setprecision(1) << snr << "    | " 
                  << std::setprecision(6) << ber << "   | " 
                  << std::setprecision(2) << acc << "%" << std::endl;
    }
    return 0;
}
"""
with open("mimo_final.cu", "w") as f: f.write(cpp_code)

# ==========================================
# PART 3: COMPILE & RUN
# ==========================================
print("\n=== STEP 3: COMPILING & RUNNING ===")
#!nvcc -o mimo_final mimo_final.cu -lcublas -lcurand -std=c++17 -arch=sm_75
#!./mimo_final