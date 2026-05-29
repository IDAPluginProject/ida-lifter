/*
 * AVX-512 broadcast & 128-bit lane-shuffle coverage.
 *
 * Exercises instruction forms that route through dedicated lifter handlers:
 *   - vpbroadcastd / vpbroadcastq from a general-purpose register
 *   - vshuff32x4 / vshuff64x2 / vshufi32x4 / vshufi64x2 (128-bit lane shuffles)
 *   - masked vbroadcastss / vbroadcastsd (EVEX merge & zero masking)
 *
 * Each instruction lives in its own non-inlined exported function so the
 * disassembly/pseudocode is easy to inspect with:
 *   idump --plugin lifter --pseudo test_avx512_bcast_shuffle
 *
 * Built by the test CMake (see test/CMakeLists.txt) and exercised by the
 * `run_idump_smoke` target in test/Makefile.
 */
#include <immintrin.h>
#include <stdint.h>

#define NOINLINE __attribute__((noinline))

/* Result sink so the compiler keeps every call above. */
static __attribute__((aligned(64))) int sink[16];

/* ---- vpbroadcastd zmm, r32 -------------------------------------------- */
NOINLINE __m512i bcast_d_gpr(int n) {
    return _mm512_set1_epi32(n);          /* vpbroadcastd zmm, edi */
}

/* ---- vpbroadcastq zmm, r64 -------------------------------------------- */
NOINLINE __m512i bcast_q_gpr(long long n) {
    return _mm512_set1_epi64(n);          /* vpbroadcastq zmm, rdi */
}

/* ---- vshuff32x4 zmm, zmm, zmm, imm8 ----------------------------------- */
NOINLINE __m512 shuf_f32x4(__m512 a, __m512 b) {
    return _mm512_shuffle_f32x4(a, b, 0x4E);
}

/* ---- vshuff64x2 zmm, zmm, zmm, imm8 ----------------------------------- */
NOINLINE __m512d shuf_f64x2(__m512d a, __m512d b) {
    return _mm512_shuffle_f64x2(a, b, 0xEE);
}

/* ---- vshufi32x4 zmm, zmm, zmm, imm8 ----------------------------------- */
NOINLINE __m512i shuf_i32x4(__m512i a, __m512i b) {
    return _mm512_shuffle_i32x4(a, b, 0x1B);
}

/* ---- vshufi64x2 zmm, zmm, zmm, imm8 ----------------------------------- */
NOINLINE __m512i shuf_i64x2(__m512i a, __m512i b) {
    return _mm512_shuffle_i64x2(a, b, 0x39);
}

/* ---- vbroadcastss zmm{k}, xmm  (merge masking) ------------------------ */
NOINLINE __m512 bcast_ss_mask(__m512 src, __mmask16 k, __m128 a) {
    return _mm512_mask_broadcastss_ps(src, k, a);
}

/* ---- vbroadcastss zmm{k}{z}, xmm  (zero masking) ---------------------- */
NOINLINE __m512 bcast_ss_maskz(__mmask16 k, __m128 a) {
    return _mm512_maskz_broadcastss_ps(k, a);
}

/* ---- vbroadcastsd zmm{k}, xmm  (merge masking) ------------------------ */
NOINLINE __m512d bcast_sd_mask(__m512d src, __mmask8 k, __m128d a) {
    return _mm512_mask_broadcastsd_pd(src, k, a);
}

/* ---- vbroadcastsd zmm{k}{z}, xmm  (zero masking) ---------------------- */
NOINLINE __m512d bcast_sd_maskz(__mmask8 k, __m128d a) {
    return _mm512_maskz_broadcastsd_pd(k, a);
}

int main(int argc, char **argv) {
    (void)argv;
    int s = argc;
    __m512i di = bcast_d_gpr(s);
    __m512i qi = bcast_q_gpr((long long)s);

    __m512  fa = _mm512_set1_ps((float)argc);
    __m512  fb = _mm512_set1_ps((float)argc + 1.0f);
    __m512d da = _mm512_set1_pd((double)argc);
    __m512d db = _mm512_set1_pd((double)argc + 1.0);

    __m512  sf = shuf_f32x4(fa, fb);
    __m512d sd = shuf_f64x2(da, db);
    __m512i sii = shuf_i32x4(di, qi);
    __m512i siq = shuf_i64x2(di, qi);

    __m128  x  = _mm_set1_ps((float)argc);
    __m128d xd = _mm_set1_pd((double)argc);
    __mmask16 k16 = (__mmask16)argc;
    __mmask8  k8  = (__mmask8)argc;

    __m512  m1 = bcast_ss_mask(fa, k16, x);
    __m512  m2 = bcast_ss_maskz(k16, x);
    __m512d m3 = bcast_sd_mask(da, k8, xd);
    __m512d m4 = bcast_sd_maskz(k8, xd);

    /* Combine results with plain vector ops (no _mm512_reduce_*, which would
     * pull vextractf64x4 glue into this file) and store to a sink so nothing is
     * dead-code eliminated. */
    __m512  accf = _mm512_add_ps(_mm512_add_ps(sf, m1), m2);
    __m512d accd = _mm512_add_pd(_mm512_add_pd(sd, m3), m4);
    __m512i acci = _mm512_add_epi32(_mm512_add_epi32(sii, siq),
                                    _mm512_castpd_si512(accd));
    acci = _mm512_xor_si512(acci, _mm512_castps_si512(accf));
    _mm512_storeu_si512((void *)sink, acci);
    return (int)sink[0];
}
