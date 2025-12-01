// Test case replicating fluid_advect logic
// Uses: vgatherdps, vroundps, vcvttps2dq, vfnmadd*, vfmsub*, vpinsrd, vinserti128, vpblendw, vpmovsxbd
#include <immintrin.h>

#define GRID_SIZE 64
#define GRID_TOTAL (GRID_SIZE * GRID_SIZE)

// Test gather instruction
__attribute__((noinline))
void test_gather(float* dst, const float* src, const int* indices, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256i vidx = _mm256_loadu_si256((const __m256i*)&indices[i]);
        __m256 result = _mm256_i32gather_ps(src, vidx, 4);
        _mm256_storeu_ps(&dst[i], result);
    }
}

// Test round and convert
__attribute__((noinline))
void test_round_cvt(int* dst, const float* src, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(&src[i]);
        // Round toward negative infinity (floor)
        __m256 rounded = _mm256_round_ps(v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        // Convert to int with truncation
        __m256i ints = _mm256_cvttps_epi32(rounded);
        _mm256_storeu_si256((__m256i*)&dst[i], ints);
    }
}

// Test negative FMA variants (vfnmadd*)
__attribute__((noinline))
void test_fnmadd(float* dst, const float* a, const float* b, const float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        // result = -(a * b) + c  (vfnmadd231ps)
        __m256 result = _mm256_fnmadd_ps(va, vb, vc);
        _mm256_storeu_ps(&dst[i], result);
    }
}

// Test FMA subtract variants (vfmsub*)
__attribute__((noinline))
void test_fmsub(float* dst, const float* a, const float* b, const float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        // result = (a * b) - c  (vfmsub213ps)
        __m256 result = _mm256_fmsub_ps(va, vb, vc);
        _mm256_storeu_ps(&dst[i], result);
    }
}

// Test scalar negative FMA with memory operand
__attribute__((noinline))
float test_scalar_fnmadd_mem(float a, float b, const float* c) {
    // result = -(a * b) + *c  (vfnmadd132ss with mem or similar)
    return -a * b + *c;
}

// Test scalar round
__attribute__((noinline))
float test_scalar_round(float x) {
    // Should generate vroundss
    return __builtin_floorf(x);
}

// Test vpinsrd / vinserti128 / vpblendw pattern
__attribute__((noinline))
__m256i test_insert_blend(int a, int b, int c, int d, __m128i base) {
    // Build vector using inserts
    __m128i v = _mm_insert_epi32(base, a, 0);
    v = _mm_insert_epi32(v, b, 1);
    v = _mm_insert_epi32(v, c, 2);
    v = _mm_insert_epi32(v, d, 3);
    // Insert into YMM
    __m256i result = _mm256_inserti128_si256(_mm256_setzero_si256(), v, 0);
    return result;
}

// Test vpmovsxbd (sign extend bytes to dwords)
__attribute__((noinline))
__m128i test_pmovsxbd(const int8_t* src) {
    // Load 4 bytes and sign-extend to 4 dwords
    return _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(const int*)src));
}

// Test vminss/vmaxss
__attribute__((noinline))
float test_scalar_minmax(float a, float b, float c) {
    float x = (a < b) ? a : b;  // Should use vminss
    float y = (x > c) ? x : c;  // Should use vmaxss
    return y;
}

// Simplified fluid_advect - uses combination of above
__attribute__((noinline))
void fluid_advect_simple(float* __restrict__ dst, const float* __restrict__ src,
                         const float* __restrict__ vel_x, const float* __restrict__ vel_y,
                         float dt) {
    float dt0 = dt * GRID_SIZE;

    for (int j = 1; j < GRID_SIZE - 1; j++) {
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            int idx = j * GRID_SIZE + i;

            // Trace back
            float x = (float)i - dt0 * vel_x[idx];
            float y = (float)j - dt0 * vel_y[idx];

            // Clamp
            if (x < 0.5f) x = 0.5f;
            if (x > GRID_SIZE - 1.5f) x = GRID_SIZE - 1.5f;
            if (y < 0.5f) y = 0.5f;
            if (y > GRID_SIZE - 1.5f) y = GRID_SIZE - 1.5f;

            // Get integer and fractional parts
            int i0 = (int)x;
            int j0 = (int)y;
            float s1 = x - i0;
            float s0 = 1.0f - s1;
            float t1 = y - j0;
            float t0 = 1.0f - t1;

            // Bilinear interpolation
            int idx00 = j0 * GRID_SIZE + i0;
            int idx01 = idx00 + 1;
            int idx10 = idx00 + GRID_SIZE;
            int idx11 = idx10 + 1;

            dst[idx] = s0 * (t0 * src[idx00] + t1 * src[idx10]) +
                       s1 * (t0 * src[idx01] + t1 * src[idx11]);
        }
    }
}

int main() {
    __attribute__((aligned(32))) float data[GRID_TOTAL];
    __attribute__((aligned(32))) float vel_x[GRID_TOTAL];
    __attribute__((aligned(32))) float vel_y[GRID_TOTAL];
    __attribute__((aligned(32))) float dst[GRID_TOTAL];
    __attribute__((aligned(32))) int indices[64];
    __attribute__((aligned(32))) int int_dst[64];

    // Initialize
    for (int i = 0; i < GRID_TOTAL; i++) {
        data[i] = (float)i * 0.01f;
        vel_x[i] = 0.1f;
        vel_y[i] = 0.05f;
        dst[i] = 0.0f;
    }
    for (int i = 0; i < 64; i++) {
        indices[i] = (i * 7) % GRID_TOTAL;
    }

    // Test gather
    test_gather(dst, data, indices, 64);

    // Test round/convert
    test_round_cvt(int_dst, data, 64);

    // Test FNmadd
    test_fnmadd(dst, data, data + 64, data + 128, 64);

    // Test FMsub
    test_fmsub(dst, data, data + 64, data + 128, 64);

    // Test scalar fnmadd with mem
    float c_val = 100.0f;
    float scalar_result = test_scalar_fnmadd_mem(3.0f, 4.0f, &c_val);

    // Test scalar round
    float rounded = test_scalar_round(3.7f);

    // Test insert/blend
    __m128i base = _mm_setzero_si128();
    __m256i blended = test_insert_blend(1, 2, 3, 4, base);

    // Test pmovsxbd
    int8_t bytes[4] = {-1, 2, -3, 4};
    __m128i extended = test_pmovsxbd(bytes);

    // Test scalar minmax
    float minmax_result = test_scalar_minmax(5.0f, 3.0f, 4.0f);

    // Test simplified advect
    fluid_advect_simple(dst, data, vel_x, vel_y, 0.016f);

    return (int)(dst[100] + scalar_result + rounded + minmax_result) & 0xFF;
}
