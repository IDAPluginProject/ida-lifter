// Test case replicating fluid_diffuse_avx logic
// Uses: vmulss, vfmadd132ss, vdivss, vbroadcastss, vmovups, vaddps, vfmadd213ps, vmulps
#include <immintrin.h>

#define GRID_SIZE 64
#define GRID_TOTAL (GRID_SIZE * GRID_SIZE)

__attribute__((noinline))
void fluid_diffuse_avx(float* __restrict__ dst, const float* __restrict__ src,
                       float dt, float diff, int iterations) {
    if (iterations <= 0) return;

    // Scalar setup: a = dt * diff * GRID_SIZE * GRID_SIZE
    float a = dt * diff * (float)(GRID_SIZE * GRID_SIZE);
    // c = 1.0f / (1.0f + 4.0f * a)
    float c = 1.0f / (1.0f + 4.0f * a);

    // Broadcast scalars to YMM
    __m256 va = _mm256_set1_ps(a);
    __m256 vc = _mm256_set1_ps(c);

    for (int iter = 0; iter < iterations; iter++) {
        // Process grid in 8-wide chunks (AVX)
        for (int i = GRID_SIZE + 1; i < GRID_TOTAL - GRID_SIZE - 8; i += 8) {
            // Load neighbors
            __m256 left   = _mm256_loadu_ps(&dst[i - 1]);
            __m256 right  = _mm256_loadu_ps(&dst[i + 1]);
            __m256 up     = _mm256_loadu_ps(&dst[i - GRID_SIZE]);
            __m256 down   = _mm256_loadu_ps(&dst[i + GRID_SIZE]);

            // Sum neighbors
            __m256 sum = _mm256_add_ps(left, right);
            sum = _mm256_add_ps(sum, up);
            sum = _mm256_add_ps(sum, down);

            // Load source and apply FMA: result = a * sum + src[i]
            // This should generate vfmadd213ps with memory operand
            __m256 source = _mm256_loadu_ps(&src[i]);
            __m256 result = _mm256_fmadd_ps(va, sum, source);

            // Multiply by c
            result = _mm256_mul_ps(result, vc);

            // Store result
            _mm256_storeu_ps(&dst[i], result);
        }
    }
}

// Additional test: FMA with memory operand directly
__attribute__((noinline))
void test_fma_mem(float* dst, const float* a, const float* b, const float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        // FMA where c is loaded from memory: va * vb + c[i]
        // Should generate vfmadd213ps ymm, ymm, mem
        __m256 result = _mm256_fmadd_ps(va, vb, _mm256_loadu_ps(&c[i]));
        _mm256_storeu_ps(&dst[i], result);
    }
}

// Test scalar FMA with memory
__attribute__((noinline))
float test_scalar_fma_mem(float a, float b, const float* c) {
    // Should generate vfmadd132ss or similar with memory operand
    return a * b + *c;
}

int main() {
    __attribute__((aligned(32))) float dst[GRID_TOTAL];
    __attribute__((aligned(32))) float src[GRID_TOTAL];

    // Initialize
    for (int i = 0; i < GRID_TOTAL; i++) {
        dst[i] = (float)i * 0.01f;
        src[i] = (float)(GRID_TOTAL - i) * 0.01f;
    }

    // Run diffusion
    fluid_diffuse_avx(dst, src, 0.016f, 0.1f, 4);

    // Test FMA with memory
    __attribute__((aligned(32))) float a[64], b[64], c[64], d[64];
    for (int i = 0; i < 64; i++) {
        a[i] = (float)i;
        b[i] = (float)(64 - i);
        c[i] = 1.0f;
    }
    test_fma_mem(d, a, b, c, 64);

    // Test scalar FMA
    float scalar_c = 100.0f;
    float result = test_scalar_fma_mem(3.0f, 4.0f, &scalar_c);

    return (int)(dst[100] + d[32] + result) & 0xFF;
}
