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
void density_fourier_naive(const density_fourier_params<Float>& params) {
    long s = 0;
    for(; (s + dblock) < params.scount; s += dblock) {
        #pragma unroll
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

#ifdef __AVX__

#include <immintrin.h>
#include <algorithm>

inline float _mm128_hsum_ps(__m128 varg) {
    __m128 vshuf = _mm_movehdup_ps(varg);
    __m128 vsums = _mm_add_ps(varg, vshuf);
    vshuf = _mm_movehl_ps(vshuf, vsums);
    vsums = _mm_add_ss(vsums, vshuf);
    return _mm_cvtss_f32(vsums);
}

inline float _mm256_hsum_ps(__m256 varg) {
    __m128 vlow  = _mm256_castps256_ps128(varg);
    __m128 vhigh = _mm256_extractf128_ps(varg, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    return _mm128_hsum_ps(vlow);
}

inline void step_avx(const long& h, 
                     __m256& cre, __m256& cim,
                     const __m256 cos, const __m256 sin, 
                     const density_fourier_params<float>& params) {
    params.reharmonics[h] += _mm256_hsum_ps(cre);
    params.imharmonics[h] += _mm256_hsum_ps(cim);
    const __m256 pre = cre;
    cre = _mm256_sub_ps(
                        _mm256_mul_ps(cre, cos), 
                        _mm256_mul_ps(cim, sin));
    cim = _mm256_add_ps(
                        _mm256_mul_ps(pre, sin), 
                        _mm256_mul_ps(cim, cos));
}

template<long dblock = 128, long hblock = 32> 
void density_fourier_avx(const density_fourier_params<float>& params) {
    constexpr long vec_width = 8;
    static_assert((dblock % vec_width) == 0);
    alignas(32) float sins[dblock];
    alignas(32) float coss[dblock];
    long s = 0;
    for(; (s + dblock) < params.scount; s += dblock) {
        const auto* const from = params.samples + s;
        #pragma unroll
        for(long i = 0; i < dblock; ++i) {
            const float arg = params.basek * (from[i] - params.shift);
            sins[i] = std::sin(arg);
            coss[i] = std::cos(arg);
        }
        #pragma unroll
        for(long i = 0; i < dblock; i += vec_width) {
            const __m256 sin = _mm256_load_ps(sins + i);
            const __m256 cos = _mm256_load_ps(coss + i);
            __m256 cre = cos, cim = sin;
            long h = 1;
            *(params.reharmonics) += float(vec_width);
            for(; (h + hblock) < params.hcount; h += hblock) {
                #pragma unroll
                for(long j = 0; j < hblock; ++j) {
                    const long c = h + j;
                    step_avx(c, cre, cim, cos, sin, params);
                }
            }
            for(; h < params.hcount; ++h) {
                step_avx(h, cre, cim, cos, sin, params);
            }
        }
    }
    for(; s < params.scount; ++s) {
        const auto& sample = params.samples[s];
        handle_sample<float, hblock>(sample, params);
    }
}

#endif

template<typename Float, long dblock = 128, long hblock = 32> 
void density_fourier(const density_fourier_params<Float>& params) {
    #ifdef __AVX__
    if constexpr (std::is_same_v<Float, float>) {
        density_fourier_avx<dblock, hblock>(params);
        return;
    }
    #endif
    density_fourier_naive<Float, dblock, hblock>(params);
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