// AVX-512 test: vaddps with ZMM registers and memory operand
#include <immintrin.h>

// Force memory operand by making b volatile
__m512 test_avx512_mem_addps(__m512 a, volatile __m512 *b_ptr) {
    return _mm512_add_ps(a, *b_ptr);
}

int main() {
    __m512 a = _mm512_setzero_ps();
    __m512 b = _mm512_setzero_ps();
    volatile __m512 bv = b;
    __m512 result = test_avx512_mem_addps(a, &bv);
    (void)result;
    return 0;
}
