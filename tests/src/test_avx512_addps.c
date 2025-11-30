// AVX-512 test: vaddps with ZMM registers
#include <immintrin.h>

__m512 test_avx512_addps(__m512 a, __m512 b) {
    return _mm512_add_ps(a, b);
}

int main() {
    __m512 a = _mm512_setzero_ps();
    __m512 b = _mm512_setzero_ps();
    __m512 result = test_avx512_addps(a, b);
    (void)result;
    return 0;
}
