// dataset_gen.cu
// Compile: nvcc -O3 -arch=sm_70 -o dataset_gen dataset_gen.cu
// (adjust -arch to your GPU SM version)
// This program generates synthetic Rayleigh MIMO data and writes H.bin, X.bin, Y.bin
// Format written:
//  - H.bin: floats in order [sample0 (Nr*Nt complex float32 interleaved as real,imag), sample1, ...]
//    Each complex stored as two float32: real then imag.
//    Total floats per sample = Nr * Nt * 2
//  - X.bin: float32 complex symbols, shape (N, Nt) stored same complex layout (real, imag)
//  - Y.bin: float32 complex received vectors, shape (N, Nr) same layout
//
// Defaults match your python parameters:
//   N=30000, Nr=4, Nt=8, SNR_dB=10
//
// The program writes three binary files and prints some summary info.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>

int main(int argc, char** argv){
    // parameters (can be changed or read from argv)
    const int N = 30000;
    const int Nr = 4;
    const int Nt = 8;
    const float SNR_dB = 10.0f;

    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    auto complex_write = [&](const std::string &fname, const std::vector<float> &buf) {
        std::ofstream ofs(fname, std::ios::binary);
        if(!ofs){ std::fprintf(stderr, "Failed to open %s\n", fname.c_str()); std::exit(1); }
        ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size()*sizeof(float));
        ofs.close();
    };

    std::cout << "Generating dataset: N="<<N<<" Nr="<<Nr<<" Nt="<<Nt<<" SNR(dB)="<<SNR_dB<<"\n";

    // allocate buffers (store as interleaved real,imag floats)
    std::vector<float> Hbuf((size_t)N * Nr * Nt * 2);
    std::vector<float> Xbuf((size_t)N * Nt * 2);
    std::vector<float> Ybuf((size_t)N * Nr * 2);

    // constellation mapping (QPSK)
    const std::complex<float> QPSK[4] = {
        {1.0f, 1.0f},
        {1.0f, -1.0f},
        {-1.0f, -1.0f},
        {-1.0f, 1.0f}
    };

    // SNR
    float SNR = powf(10.0f, SNR_dB/10.0f);
    float noise_var = Nt / SNR;            // same normalization as in Python
    float noise_sigma = sqrtf(noise_var/2.0f);

    // Generate per-sample H (i.i.d. CN(0,1)), X (random QPSK), noise, Y = H X + n
    for(int m=0;m<N;++m){
        // generate H: Nr x Nt complex
        for(int r=0;r<Nr;++r){
            for(int t=0;t<Nt;++t){
                float re = nd(rng)/sqrtf(2.0f);
                float im = nd(rng)/sqrtf(2.0f);
                size_t idx = ((size_t)m * Nr * Nt + r * Nt + t) * 2;
                Hbuf[idx + 0] = re;
                Hbuf[idx + 1] = im;
            }
        }
        // generate X: Nt complex QPSK
        for(int t=0;t<Nt;++t){
            int b0 = (int)(rng() & 1);
            int b1 = (int)((rng()>>1) & 1);
            int idxq = b0*2 + b1;
            std::complex<float> sym = QPSK[idxq];
            size_t idx = ((size_t)m * Nt + t) * 2;
            Xbuf[idx + 0] = sym.real();
            Xbuf[idx + 1] = sym.imag();
        }
        // compute Y = H * x (Nr x 1) and add noise
        for(int r=0;r<Nr;++r){
            std::complex<float> acc = {0.0f, 0.0f};
            for(int t=0;t<Nt;++t){
                size_t hidx = ((size_t)m * Nr * Nt + r * Nt + t) * 2;
                std::complex<float> h(Hbuf[hidx+0], Hbuf[hidx+1]);
                size_t xidx = ((size_t)m * Nt + t) * 2;
                std::complex<float> x(Xbuf[xidx+0], Xbuf[xidx+1]);
                acc += h * x;
            }
            // noise
            float nr = nd(rng)*noise_sigma;
            float ni = nd(rng)*noise_sigma;
            acc += std::complex<float>(nr, ni);
            size_t yidx = ((size_t)m * Nr + r) * 2;
            Ybuf[yidx+0] = acc.real();
            Ybuf[yidx+1] = acc.imag();
        }
    }

    std::cout << "Writing H.bin (" << Hbuf.size()*sizeof(float)/1e6 << " MB), "
              << "X.bin (" << Xbuf.size()*sizeof(float)/1e6 << " MB), "
              << "Y.bin (" << Ybuf.size()*sizeof(float)/1e6 << " MB)\n";

    complex_write("H.bin", Hbuf);
    complex_write("X.bin", Xbuf);
    complex_write("Y.bin", Ybuf);

    std::cout << "Done.\n";
    return 0;
}
