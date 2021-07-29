#pragma once

extern "C" {
    int density_fourier_capi_float(
        const float* const data, float* const reharmonics, float* const imharmonics,
        const long scount, const long hcount, const float shift, const float basek);
    int density_fourier_capi_double(
        const double* const data, double* const reharmonics, double* const imharmonics,
        const long scount, const long hcount, const double shift, const double basek);
    float evaluate_fourier_capi_float(
        float x, float* const reharmonics, float* const imharmonics,
        const long hcount, const float shift, const float basek);
    double evaluate_fourier_capi_double(
        double x, double* const reharmonics, double* const imharmonics,
        const long hcount, const double shift, const double basek);
};