#ifndef MATH_AVX_H
#define MATH_AVX_H

#include "common.h"

// Horizontal add for AVX __m256
static inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Horizontal add for SSE __m128
static inline float hsum128_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Dot product of two vectors (x,y,z) stored in __m128
static inline __m128 dot_product_sse(__m128 a, __m128 b) {
    // mask 0x71 = 0111 0001 (calc x,y,z; store x)
    return _mm_dp_ps(a, b, 0x71);
}

// Normalize __m128 vector
static inline __m128 normalize_sse(__m128 v) {
    __m128 dot = dot_product_sse(v, v);
    __m128 inv_sqrt = _mm_rsqrt_ps(dot);
    return _mm_mul_ps(v, inv_sqrt);
}

// AVX sqrt with Newton-Raphson refinement for better precision
static inline __m256 sqrt_nr_ps(__m256 x) {
    __m256 rsqrt = _mm256_rsqrt_ps(x);
    // Newton-Raphson: rsqrt' = rsqrt * (1.5 - 0.5 * x * rsqrt^2)
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 three_half = _mm256_set1_ps(1.5f);
    __m256 rsqrt2 = _mm256_mul_ps(rsqrt, rsqrt);
    __m256 refined = _mm256_mul_ps(rsqrt, _mm256_sub_ps(three_half, _mm256_mul_ps(half, _mm256_mul_ps(x, rsqrt2))));
    return _mm256_mul_ps(x, refined);
}

// SSE sqrt with Newton-Raphson refinement
static inline __m128 sqrt_nr_sse(__m128 x) {
    __m128 rsqrt = _mm_rsqrt_ps(x);
    __m128 half = _mm_set1_ps(0.5f);
    __m128 three_half = _mm_set1_ps(1.5f);
    __m128 rsqrt2 = _mm_mul_ps(rsqrt, rsqrt);
    __m128 refined = _mm_mul_ps(rsqrt, _mm_sub_ps(three_half, _mm_mul_ps(half, _mm_mul_ps(x, rsqrt2))));
    return _mm_mul_ps(x, refined);
}

// Floor for AVX (requires SSE4.1/AVX)
static inline __m256 floor_ps(__m256 x) {
    return _mm256_floor_ps(x);
}

// Clamp AVX vector to [min, max]
static inline __m256 clamp_ps(__m256 x, __m256 min_val, __m256 max_val) {
    return _mm256_min_ps(_mm256_max_ps(x, min_val), max_val);
}

// Convert __m256 to __m256i (truncate to int)
static inline __m256i cvt_ps_epi32(__m256 x) {
    return _mm256_cvttps_epi32(x);
}

// Blend based on mask (AVX)
static inline __m256 blendv_ps(__m256 a, __m256 b, __m256 mask) {
    return _mm256_blendv_ps(a, b, mask);
}

// FMA: a * b + c (if FMA available, otherwise emulate)
#ifdef __FMA__
static inline __m256 fmadd_ps(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
static inline __m256 fmsub_ps(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmsub_ps(a, b, c);
}
static inline __m128 fmadd_sse(__m128 a, __m128 b, __m128 c) {
    return _mm_fmadd_ps(a, b, c);
}
#else
static inline __m256 fmadd_ps(__m256 a, __m256 b, __m256 c) {
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}
static inline __m256 fmsub_ps(__m256 a, __m256 b, __m256 c) {
    return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
}
static inline __m128 fmadd_sse(__m128 a, __m128 b, __m128 c) {
    return _mm_add_ps(_mm_mul_ps(a, b), c);
}
#endif

#endif // MATH_AVX_H
