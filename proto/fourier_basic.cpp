#include <cmath>
#include "fourier_capi.h"

template<typename Float>
struct density_fourier_params {
    const Float* const samples;
    Float* const reharmonics;
    Float* const imharmonics;
    const long scount;
    const long hcount;

    const Float shift;
    const Float basek;

    density_fourier_params(
        const Float* const data_, Float* const reharmonics_, Float* const imharmonics_,
        const long scount_, const long hcount_, const Float shift_, const Float basek_)
        : samples(data_), reharmonics(reharmonics_), imharmonics(imharmonics_), 
        scount(scount_), hcount(hcount_), shift(shift_), basek(basek_) {}

    density_fourier_params(
        Float* const reharmonics_, Float* const imharmonics_,
        const long hcount_, const Float shift_, const Float basek_)
        : samples(nullptr), reharmonics(reharmonics_), imharmonics(imharmonics_), 
        scount(0l), hcount(hcount_), shift(shift_), basek(basek_) {}
};

template struct density_fourier_params<float>;
template struct density_fourier_params<double>;

#define PREFETCH(ADDR, RW, LOCALITY) __builtin_prefetch(ADDR, RW, LOCALITY)

template<typename Float>
inline void step(const long& h, Float& cre, Float& cim,
                 const Float& cos, const Float& sin, 
                 const density_fourier_params<Float>& params) {
    params.reharmonics[h] += cre;
    params.imharmonics[h] += cim;
    const Float pre = cre;
    cre = cre * cos - cim * sin;
    cim = pre * sin + cim * cos;
}

template<typename Float, long block = 32>
inline void handle_sample(const Float& sample, 
        const density_fourier_params<Float>& params) {
    const Float arg = params.basek * (sample - params.shift);
    const Float cos = std::cos(arg), sin = std::sin(arg);
    Float cre = cos, cim = sin;
    //Zero harmonic
    (*params.reharmonics) += Float(1); 
    //Normal harmonics
    long h = 1;
    for(; (h + block) < params.hcount; h += block) {
        #pragma unroll
        for(long i = 0; i < block; ++i) {
            const long iharm = h + i;
            step<Float>(iharm, cre, cim, cos, sin, params);
        }
    }
    for(; h < params.hcount; ++h) {
        step<Float>(h, cre, cim, cos, sin, params);
    }
}

template<typename Float, long dblock = 512, long hblock = 32> 
void density_fourier(const density_fourier_params<Float>& params) {
    long s = 0;
    //PREFETCH((params.reharmonics), 1, 3);
    //PREFETCH((params.imharmonics), 1, 3);
    for(; (s + dblock) < params.scount; s += dblock) {
        //PREFETCH((params.samples + s), 0, 2);
        //PREFETCH((params.samples + s + dblock), 0, 1);
        for(long i = 0; i < dblock; ++i) {
            const long isamp = s + i;
            const auto& sample = params.samples[isamp];
            handle_sample<Float, hblock>(sample, params);
        }
    }
    for(; s < params.scount; ++s) {
        const auto& sample = params.samples[s];
        handle_sample<Float, hblock>(sample, params);
    }
}

template void density_fourier<float>(const density_fourier_params<float>& params);
template void density_fourier<double>(const density_fourier_params<double>& params);

int density_fourier_capi_float(
    const float* const data, float* const reharmonics, float* const imharmonics,
    const long scount, const long hcount, const float shift, const float basek) {
    try {
        const density_fourier_params<float> params(
            data, reharmonics, imharmonics, scount, hcount, shift, basek);
        density_fourier<float>(params);
    } catch( ... ) {
        return 1;
    }
    return 0;
}

int density_fourier_capi_double(
    const double* const data, double* const reharmonics, double* const imharmonics,
    const long scount, const long hcount, const double shift, const double basek) {
    try {
        const density_fourier_params<double> params(
            data, reharmonics, imharmonics, scount, hcount, shift, basek);
        density_fourier<double>(params);
    } catch( ... ) {
        return 1;
    }
    return 0;
}

template<typename Float>
inline Float ev_step(const long& h, Float& cre, Float& cim, 
                     const Float& sin, const Float& cos, 
                     const density_fourier_params<Float>& params) {
    const Float res = 
        params.reharmonics[h] * cre + 
        params.imharmonics[h] * cim;
    const Float pre = cre;
    cre = cre * cos - cim * sin;
    cim = pre * sin + cim * cos;
    return res;
}

template<typename Float, long hblock = 32>
Float evaluate(const Float& sample, 
        const density_fourier_params<Float>& params) {
    const Float arg = params.basek * (sample - params.shift);
    Float acc = (*params.reharmonics);
    long h = 1;
    const Float cos = std::cos(arg), sin = std::sin(arg);
    Float cre = cos, cim = sin;
    for(; (h + hblock) < params.hcount; ++h) {
        #pragma unroll
        for(long i = 0; i < hblock; ++i) {
            const long iharm = i + h;
            acc += ev_step<Float>(iharm, cre, cim, sin, cos, params);
        }
    }
    for(; h < params.hcount; ++h) {
        acc += ev_step<Float>(h, cre, cim, sin, cos, params);
    }
    return acc;
}

template float evaluate<float>(const float& x, const density_fourier_params<float>& params);
template double evaluate<double>(const double& x, const density_fourier_params<double>& params);

float evaluate_fourier_capi_float(
    float x, float* const reharmonics, float* const imharmonics,
    const long hcount, const float shift, const float basek) {
    const density_fourier_params<float> params(
        reharmonics, imharmonics, hcount, shift, basek);
    return evaluate<float>(x, params);
}

double evaluate_fourier_capi_double(
    double x, double* const reharmonics, double* const imharmonics,
    const long hcount, const double shift, const double basek) {
    const density_fourier_params<double> params(
        reharmonics, imharmonics, hcount, shift, basek);
    return evaluate<double>(x, params);
}