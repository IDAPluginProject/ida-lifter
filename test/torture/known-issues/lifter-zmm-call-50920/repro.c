#include <immintrin.h>
#include <stdint.h>

/* ABI/calling-convention torture: funcs=40 seed=1 */
/* compiles on gcc(Linux) and clang --target=x86_64-pc-windows-gnu */

/* ------------------------------------------------------------------------- */
/* Portable calling-convention shim.                                         */
/*   CONV_MS / CONV_SYSV  -> always real (gcc + clang both support these).   */
/*   CONV_VEC / CONV_REG  -> __vectorcall / __regcall where the compiler     */
/*       supports them (clang, incl. the windows-gnu target); otherwise they */
/*       fall back to a genuine, distinct ABI attribute so the cross-        */
/*       convention call chains remain legal and warning-clean everywhere.   */
/* ------------------------------------------------------------------------- */
#define CONV_MS    __attribute__((ms_abi))
#define CONV_SYSV  __attribute__((sysv_abi))

#if defined(__clang__)
  /* clang accepts the attribute spelling on every x86-64 target it targets. */
  #define CONV_VEC  __attribute__((vectorcall))
  #define CONV_REG  __attribute__((regcall))
#elif defined(_MSC_VER)
  #define CONV_VEC  __vectorcall
  #define CONV_REG  __vectorcall
#else
  /* gcc SysV target: no vectorcall/regcall -> keep the chain legal with two
     real, distinct ABI attributes (still forces cross-ABI spills). */
  #define CONV_VEC  __attribute__((sysv_abi))
  #define CONV_REG  __attribute__((ms_abi))
#endif

/* cross-function vector/mask state pools (force loads/stores) */
__m512 g_v512f[8];
__m512d g_v512d[8];
__m512i g_v512i[8];
__m256 g_v256f[8];
__m256d g_v256d[8];
__m256i g_v256i[8];
__m128 g_v128f[8];
__m128d g_v128d[8];
__m128i g_v128i[8];
__mmask8 g_m8[8];
__mmask16 g_m16[8];
__mmask32 g_m32[8];
__mmask64 g_m64[8];

/* forward declarations (call chains across conventions) */
CONV_SYSV __mmask8 abi_sysv_0(__mmask64 p0, __m512d p1, __m128 p2);
CONV_REG __m128 abi_reg_1(__mmask16 p0, __m512d p1, __m128 p2, __m512 p3, __m256i p4, __m256i p5, __mmask8 p6, __m512 p7);
CONV_REG __mmask64 abi_reg_2(__mmask8 p0, __m512d p1, __m256 p2, __m512 p3, __m512 p4);
CONV_MS __mmask32 abi_ms_3(__m512 p0, __m256i p1, __mmask32 p2, __mmask16 p3, __m256i p4, __m512 p5, __m128i p6, __mmask16 p7, __m128 p8, __m128 p9);
CONV_SYSV __m256 abi_sysv_4(__mmask32 p0, __mmask16 p1, __m128 p2, __mmask64 p3, __m512 p4);
CONV_REG __m128i abi_reg_5(__m512i p0, __mmask32 p1, __mmask64 p2);
CONV_MS __m256 abi_ms_6(__m256i p0, __m128i p1, __mmask32 p2, __mmask16 p3, __mmask64 p4, __mmask64 p5, __mmask8 p6, __m128 p7, __m128i p8, __m256i p9);
CONV_MS __m128 abi_ms_7(__m256i p0, __m256i p1, __mmask32 p2, __m512i p3, __m256 p4);
CONV_VEC __m512d abi_vec_8(__mmask32 p0, __m128i p1, __m512d p2, __m512i p3, __m128i p4, __m256i p5, __m256 p6, __m128 p7, __m512 p8);
CONV_REG __m512 abi_reg_9(__mmask8 p0, __mmask8 p1, __mmask8 p2, __m256i p3, __mmask32 p4, __m512i p5);
CONV_SYSV __m128i abi_sysv_10(__m512 p0, __mmask16 p1, __m128i p2, __m128i p3, __mmask16 p4);
CONV_REG __m128i abi_reg_11(__mmask8 p0, __m256 p1, __m128 p2, __mmask64 p3, __mmask32 p4, __m128i p5, __mmask8 p6);
CONV_MS __m256i abi_ms_12(__m512i p0, __m128i p1, __m128i p2, __mmask16 p3, __m256i p4, __m512 p5, __m128 p6, __m256 p7, __mmask8 p8, __m128i p9);
CONV_SYSV __m128i abi_sysv_13(__m128 p0, __m256 p1, __m256i p2, __m256 p3, __m512 p4, __m128i p5, __m128i p6, __mmask8 p7);
CONV_VEC __m128 abi_vec_14(__mmask16 p0, __mmask32 p1);
CONV_SYSV __m128i abi_sysv_15(__m512d p0, __m128i p1, __mmask64 p2, __m512 p3);
CONV_MS __m512d abi_ms_16(__m128 p0, __m512 p1);
CONV_VEC __mmask16 abi_vec_17(__m512d p0, __mmask8 p1, __m512i p2, __m256 p3, __mmask64 p4, __m512d p5);
CONV_SYSV __m512i abi_sysv_18(__m128i p0, __m512i p1, __mmask32 p2, __mmask64 p3, __mmask32 p4, __mmask64 p5);
CONV_REG __m256 abi_reg_19(__m128 p0, __m512d p1, __m512 p2, __mmask64 p3, __m256i p4, __m256 p5, __m256i p6, __mmask16 p7, __mmask64 p8);
CONV_MS __mmask64 abi_ms_20(__mmask16 p0, __mmask8 p1, __m256i p2, __m512 p3, __mmask16 p4, __m512 p5, __m256i p6, __m512i p7, __m512 p8, __m512i p9);
CONV_REG __m128i abi_reg_21(__m128i p0, __mmask16 p1, __mmask32 p2, __m128i p3, __m128 p4, __mmask16 p5, __m128i p6, __mmask32 p7);
CONV_MS __m256i abi_ms_22(__mmask32 p0, __mmask32 p1, __m256i p2, __m512 p3, __mmask64 p4, __m512i p5, __mmask16 p6);
CONV_MS __mmask64 abi_ms_23(__m512d p0, __mmask64 p1, __mmask64 p2);
CONV_SYSV __m256i abi_sysv_24(__m512i p0, __m512 p1, __m128i p2, __m512 p3, __mmask8 p4, __mmask16 p5);
CONV_REG __m512i abi_reg_25(__m512 p0, __m256i p1, __mmask16 p2, __m256 p3, __m512d p4, __mmask16 p5, __mmask8 p6, __mmask32 p7, __m256i p8, __mmask8 p9);
CONV_SYSV __m128 abi_sysv_26(__mmask32 p0, __m256i p1, __mmask64 p2);
CONV_REG __m512 abi_reg_27(__mmask8 p0, __m256i p1, __mmask64 p2, __m512 p3, __m512i p4, __mmask16 p5, __m256 p6);
CONV_SYSV __m256 abi_sysv_28(__mmask16 p0, __mmask64 p1, __mmask32 p2, __m512d p3, __m256i p4, __m128i p5, __m256 p6, __mmask32 p7);
CONV_REG __m128i abi_reg_29(__m512d p0, __m512 p1, __m512d p2, __m512i p3, __m512i p4);
CONV_SYSV __m128i abi_sysv_30(__mmask64 p0, __m256 p1, __mmask8 p2, __m128i p3, __mmask64 p4);
CONV_VEC __m256 abi_vec_31(__m512d p0, __mmask64 p1, __mmask16 p2, __mmask8 p3, __m128 p4, __m512i p5, __mmask8 p6);
CONV_MS __m256 abi_ms_32(__m256i p0, __m512d p1);
CONV_REG __m512i abi_reg_33(__m256 p0, __m512d p1, __mmask8 p2, __mmask8 p3);
CONV_REG __m512d abi_reg_34(__mmask16 p0, __mmask8 p1, __m512d p2, __mmask64 p3, __m256 p4, __mmask64 p5, __mmask8 p6, __m128i p7, __m512d p8, __m128 p9);
CONV_VEC __m512d abi_vec_35(__mmask64 p0, __m512 p1);
CONV_MS __m512d abi_ms_36(__m512d p0, __m512 p1, __mmask16 p2, __mmask16 p3, __mmask8 p4, __m256i p5, __m512i p6, __m512d p7);
CONV_REG __m512i abi_reg_37(__m512i p0, __m512d p1, __m256i p2, __m256i p3, __m128i p4);
CONV_VEC __m128i abi_vec_38(__m128 p0, __m256 p1, __m512d p2, __mmask16 p3, __mmask32 p4, __m256 p5);
CONV_MS __m512 abi_ms_39(__mmask64 p0, __mmask8 p1);

CONV_SYSV __mmask8 abi_sysv_0(__mmask64 p0, __m512d p1, __m128 p2) {
    __m128d sysv0_v128d_1 = g_v128d[6];
    __m128d sysv0_v128d_2 = _mm_max_pd(sysv0_v128d_1, sysv0_v128d_1);
    __m512d sysv0_v512d_3 = _mm512_add_pd(p1, p1);
    __m512i sysv0_v512i_4 = g_v512i[7];
    __m512i sysv0_v512i_5 = _mm512_add_epi32(sysv0_v512i_4, sysv0_v512i_4);
    __m256d sysv0_v256d_6 = g_v256d[3];
    __m256d sysv0_v256d_7 = _mm256_mul_pd(sysv0_v256d_6, sysv0_v256d_6);
    __m256 sysv0_v256f_8 = g_v256f[1];
    __m256 sysv0_v256f_9 = _mm256_mul_ps(sysv0_v256f_8, sysv0_v256f_8);
    __m512d sysv0_v512d_10 = _mm512_max_pd(sysv0_v512d_3, p1);
    __mmask8 sysv0_m8_11 = _mm512_cmp_pd_mask(sysv0_v512d_3, p1, _CMP_GE_OQ);
    __m128 sysv0_v128f_12 = _mm_min_ps(p2, p2);
    __m256i sysv0_v256i_13 = g_v256i[4];
    __m256i sysv0_v256i_14 = _mm256_mullo_epi32(sysv0_v256i_13, sysv0_v256i_13);
    __m256i sysv0_v256i_15 = _mm256_and_si256(sysv0_v256i_13, sysv0_v256i_13);
    __m256d sysv0_v256d_16 = _mm256_add_pd(sysv0_v256d_6, sysv0_v256d_6);
    __m512i sysv0_v512i_17 = _mm512_broadcastmb_epi64(sysv0_m8_11);
    for (int sysv0_i17 = 0; sysv0_i17 < (int)((unsigned)sysv0_m8_11 & 7); sysv0_i17++) {
    __mmask32 sysv0_m32_19 = g_m32[0];
    __mmask32 sysv0_m32_20 = _kxor_mask32(sysv0_m32_19, sysv0_m32_19);
    __mmask16 sysv0_m16_21 = _mm512_kunpackb(sysv0_m8_11, sysv0_m8_11);
    }
    __m512i sysv0_v512i_22 = _mm512_and_si512(sysv0_v512i_4, sysv0_v512i_17);
    __m512d sysv0_v512d_23 = _mm512_mask_blend_pd(sysv0_m8_11, p1, sysv0_v512d_10);
    __mmask32 sysv0_m32_24 = _mm512_cmp_epi16_mask(sysv0_v512i_5, sysv0_v512i_5, 2);
    __m512d sysv0_v512d_25 = _mm512_mask_blend_pd(sysv0_m8_11, sysv0_v512d_3, sysv0_v512d_10);
    __m256i sysv0_v256i_26 = _mm256_sub_epi32(sysv0_v256i_13, sysv0_v256i_15);
    g_v512d[3] = sysv0_v512d_25;
    g_v256i[6] = sysv0_v256i_14;
    g_v128f[3] = p2;
    g_m64[5] = p0;
    return sysv0_m8_11;
}

CONV_REG __m128 abi_reg_1(__mmask16 p0, __m512d p1, __m128 p2, __m512 p3, __m256i p4, __m256i p5, __mmask8 p6, __m512 p7) {
    __m128i reg1_v128i_1 = g_v128i[6];
    __m128i reg1_v128i_2 = _mm_sub_epi32(reg1_v128i_1, reg1_v128i_1);
    __m512 reg1_v512f_3 = _mm512_unpacklo_ps(p7, p3);
    __m128 reg1_v128f_4 = _mm512_extractf32x4_ps(p3, 2);
    __m512i reg1_v512i_5 = g_v512i[4];
    __m512i reg1_v512i_6 = _mm512_sub_epi64(reg1_v512i_5, reg1_v512i_5);
    __m512 reg1_v512f_7 = _mm512_scalef_ps(p7, p3);
    __m512 reg1_v512f_8 = _mm512_cvtepi32_ps(reg1_v512i_5);
    __mmask64 reg1_m64_9 = g_m64[5];
    __mmask8 reg1_m8_10 = abi_sysv_0(reg1_m64_9, p1, p2);
    __mmask8 reg1_m8_11 = abi_sysv_0(reg1_m64_9, p1, reg1_v128f_4);
    __m512 reg1_v512f_12 = _mm512_cvtepi32_ps(reg1_v512i_6);
    __mmask8 reg1_m8_13 = _mm512_cmp_pd_mask(p1, p1, _CMP_GE_OQ);
    __m512d reg1_v512d_14 = _mm512_sub_pd(p1, p1);
    __m128i reg1_v128i_15 = _mm_and_si128(reg1_v128i_1, reg1_v128i_2);
    __m256i reg1_v256i_16 = _mm256_add_epi64(p5, p5);
    __m512i reg1_v512i_17 = _mm512_slli_epi32(reg1_v512i_6, 3);
    for (int reg1_i17 = 0; reg1_i17 < (int)((unsigned)p0 & 7); reg1_i17++) {
    __m512 reg1_v512f_19 = _mm512_rcp14_ps(reg1_v512f_7);
    __m512i reg1_v512i_20 = _mm512_unpacklo_epi32(reg1_v512i_5, reg1_v512i_6);
    __m128d reg1_v128d_21 = g_v128d[0];
    __m128d reg1_v128d_22 = _mm_add_pd(reg1_v128d_21, reg1_v128d_21);
    __mmask64 reg1_m64_23 = _mm512_movepi8_mask(reg1_v512i_20);
    }
    __m128 reg1_v128f_24 = _mm_min_ps(p2, p2);
    __m128i reg1_v128i_25 = _mm_slli_epi32(reg1_v128i_1, 3);
    __m512i reg1_v512i_26 = _mm512_broadcastmb_epi64(reg1_m8_11);
    __m512i reg1_v512i_27 = _mm512_broadcast_i32x4(reg1_v128i_1);
    for (int reg1_i27 = 0; reg1_i27 < (int)((unsigned)p0 & 7); reg1_i27++) {
    __m512d reg1_v512d_29 = _mm512_mask_blend_pd(p6, reg1_v512d_14, p1);
    __m128 reg1_v128f_30 = _mm_mul_ps(reg1_v128f_4, reg1_v128f_4);
    }
    g_v512d[7] = reg1_v512d_14;
    g_v512i[6] = reg1_v512i_6;
    g_v128f[0] = reg1_v128f_4;
    g_m8[3] = p6;
    g_m16[4] = p0;
    return reg1_v128f_4;
}

CONV_REG __mmask64 abi_reg_2(__mmask8 p0, __m512d p1, __m256 p2, __m512 p3, __m512 p4) {
    __m256d reg2_v256d_1 = _mm512_extractf64x4_pd(p1, 1);
    __m512i reg2_v512i_2 = g_v512i[2];
    __mmask64 reg2_m64_3 = _mm512_movepi8_mask(reg2_v512i_2);
    __m512i reg2_v512i_4 = _mm512_ternarylogic_epi32(reg2_v512i_2, reg2_v512i_2, reg2_v512i_2, 0x96);
    __m512i reg2_v512i_5 = _mm512_add_epi64(reg2_v512i_2, reg2_v512i_4);
    __m256 reg2_v256f_6 = _mm256_min_ps(p2, p2);
    __m512i reg2_v512i_7 = _mm512_broadcastmb_epi64(p0);
    __m128 reg2_v128f_8 = g_v128f[5];
    __mmask8 reg2_m8_9 = abi_sysv_0(reg2_m64_3, p1, reg2_v128f_8);
    __mmask8 reg2_m8_10 = abi_sysv_0(reg2_m64_3, p1, reg2_v128f_8);
    __mmask16 reg2_m16_11 = g_m16[4];
    __mmask16 reg2_m16_12 = _knot_mask16(reg2_m16_11);
    __m512i reg2_v512i_13 = _mm512_ternarylogic_epi32(reg2_v512i_7, reg2_v512i_5, reg2_v512i_7, 0x96);
    __m512i reg2_v512i_14 = _mm512_add_epi32(reg2_v512i_7, reg2_v512i_7);
    __m256 reg2_v256f_15 = _mm256_min_ps(p2, reg2_v256f_6);
    __mmask16 reg2_m16_16 = _mm512_cmp_ps_mask(p3, p4, _CMP_LT_OQ);
    if ((unsigned long long)reg2_m16_16 & 1ULL) {
    __mmask16 reg2_m16_17 = _kor_mask16(reg2_m16_11, reg2_m16_12);
    __m256i reg2_v256i_18 = _mm512_cvtepi64_epi32(reg2_v512i_14);
    __m128i reg2_v128i_19 = g_v128i[0];
    __m128i reg2_v128i_20 = _mm_slli_epi32(reg2_v128i_19, 3);
    __mmask16 reg2_m16_21 = _mm512_movepi32_mask(reg2_v512i_7);
    }
    __mmask8 reg2_m8_22 = abi_sysv_0(reg2_m64_3, p1, reg2_v128f_8);
    __m128i reg2_v128i_23 = g_v128i[2];
    __m512i reg2_v512i_24 = _mm512_broadcast_i32x4(reg2_v128i_23);
    __mmask16 reg2_m16_25 = _mm512_cmp_ps_mask(p4, p4, _CMP_LT_OQ);
    __m512i reg2_v512i_26 = _mm512_unpacklo_epi32(reg2_v512i_24, reg2_v512i_2);
    __m512i reg2_v512i_27 = _mm512_broadcastmw_epi32(reg2_m16_11);
    for (int reg2_i27 = 0; reg2_i27 < (int)((unsigned)reg2_m16_16 & 7); reg2_i27++) {
    __mmask64 reg2_m64_29 = _mm512_movepi8_mask(reg2_v512i_27);
    __m256i reg2_v256i_30 = g_v256i[4];
    __m256i reg2_v256i_31 = _mm256_add_epi64(reg2_v256i_30, reg2_v256i_30);
    __m512i reg2_v512i_32 = _mm512_slli_epi32(reg2_v512i_26, 3);
    }
    g_v512f[1] = p3;
    g_v512d[1] = p1;
    g_v512i[2] = reg2_v512i_26;
    g_v256f[4] = reg2_v256f_15;
    g_v256d[3] = reg2_v256d_1;
    g_v128i[5] = reg2_v128i_23;
    g_m16[4] = reg2_m16_11;
    return reg2_m64_3;
}

CONV_MS __mmask32 abi_ms_3(__m512 p0, __m256i p1, __mmask32 p2, __mmask16 p3, __m256i p4, __m512 p5, __m128i p6, __mmask16 p7, __m128 p8, __m128 p9) {
    __m512i ms3_v512i_1 = g_v512i[7];
    __m512i ms3_v512i_2 = _mm512_rol_epi32(ms3_v512i_1, 5);
    __m512 ms3_v512f_3 = _mm512_maskz_mul_ps(p7, p5, p0);
    __m256d ms3_v256d_4 = _mm256_set1_pd(2.0);
    __m256d ms3_v256d_5 = _mm256_mul_pd(ms3_v256d_4, ms3_v256d_4);
    __mmask16 ms3_m16_6 = _mm512_cmp_epi32_mask(ms3_v512i_1, ms3_v512i_2, 2);
    __m256i ms3_v256i_7 = _mm256_sub_epi64(p4, p1);
    __m512 ms3_v512f_8 = _mm512_mask_add_ps(p5, p3, p0, ms3_v512f_3);
    __mmask32 ms3_m32_9 = _kxor_mask32(p2, p2);
    __mmask8 ms3_m8_10 = g_m8[4];
    __m512d ms3_v512d_11 = g_v512d[2];
    __m256 ms3_v256f_12 = g_v256f[2];
    __mmask64 ms3_m64_13 = abi_reg_2(ms3_m8_10, ms3_v512d_11, ms3_v256f_12, ms3_v512f_8, ms3_v512f_3);
    __m256 ms3_v256f_14 = _mm256_unpacklo_ps(ms3_v256f_12, ms3_v256f_12);
    __mmask16 ms3_m16_15 = _knot_mask16(ms3_m16_6);
    __m256i ms3_v256i_16 = _mm256_sub_epi32(p1, p1);
    __m256 ms3_v256f_17 = _mm256_sqrt_ps(ms3_v256f_14);
    __m256i ms3_v256i_18 = _mm256_mullo_epi32(ms3_v256i_16, p1);
    __m512 ms3_v512f_19 = _mm512_mul_ps(p5, p0);
    __m128i ms3_v128i_20 = _mm_sub_epi64(p6, p6);
    __mmask16 ms3_m16_21 = _mm512_cmp_epi32_mask(ms3_v512i_2, ms3_v512i_1, 2);
    __m512 ms3_v512f_22 = _mm512_cvtepi32_ps(ms3_v512i_1);
    __m512d ms3_v512d_23 = _mm512_fmadd_pd(ms3_v512d_11, ms3_v512d_11, ms3_v512d_11);
    __m128i ms3_v128i_24 = _mm_sub_epi64(ms3_v128i_20, ms3_v128i_20);
    g_v512f[4] = p5;
    g_v512i[4] = ms3_v512i_2;
    g_v256i[0] = p1;
    g_v128f[6] = p9;
    g_v128i[3] = ms3_v128i_24;
    g_m8[2] = ms3_m8_10;
    g_m16[2] = p3;
    g_m64[7] = ms3_m64_13;
    return ms3_m32_9;
}

CONV_SYSV __m256 abi_sysv_4(__mmask32 p0, __mmask16 p1, __m128 p2, __mmask64 p3, __m512 p4) {
    __m512 sysv4_v512f_1 = _mm512_max_ps(p4, p4);
    __m512 sysv4_v512f_2 = _mm512_fmadd_ps(sysv4_v512f_1, p4, sysv4_v512f_1);
    __m512d sysv4_v512d_3 = g_v512d[7];
    __m512d sysv4_v512d_4 = _mm512_max_pd(sysv4_v512d_3, sysv4_v512d_3);
    __mmask32 sysv4_m32_5 = _mm512_kunpackw(p1, p1);
    __mmask8 sysv4_m8_6 = g_m8[4];
    __m256 sysv4_v256f_7 = _mm256_set1_ps(1.0f);
    __mmask64 sysv4_m64_8 = abi_reg_2(sysv4_m8_6, sysv4_v512d_4, sysv4_v256f_7, p4, p4);
    __m512i sysv4_v512i_9 = g_v512i[0];
    __m512i sysv4_v512i_10 = _mm512_unpacklo_epi32(sysv4_v512i_9, sysv4_v512i_9);
    __m512i sysv4_v512i_11 = _mm512_permutexvar_epi32(sysv4_v512i_10, sysv4_v512i_9);
    __mmask16 sysv4_m16_12 = _knot_mask16(p1);
    __mmask16 sysv4_m16_13 = _mm512_cmp_epi32_mask(sysv4_v512i_9, sysv4_v512i_11, 2);
    __m256d sysv4_v256d_14 = _mm256_set1_pd(2.0);
    __m256d sysv4_v256d_15 = _mm256_max_pd(sysv4_v256d_14, sysv4_v256d_14);
    __m256d sysv4_v256d_16 = _mm256_sub_pd(sysv4_v256d_14, sysv4_v256d_14);
    for (int sysv4_i16 = 0; sysv4_i16 < (int)((unsigned)p1 & 7); sysv4_i16++) {
    __m512 sysv4_v512f_18 = _mm512_sqrt_ps(sysv4_v512f_2);
    __m512d sysv4_v512d_19 = _mm512_max_pd(sysv4_v512d_4, sysv4_v512d_4);
    }
    __m512 sysv4_v512f_20 = _mm512_fmadd_ps(p4, sysv4_v512f_2, p4);
    __m128d sysv4_v128d_21 = _mm_set1_pd(2.0);
    __m128d sysv4_v128d_22 = _mm_mul_pd(sysv4_v128d_21, sysv4_v128d_21);
    __m512 sysv4_v512f_23 = _mm512_mask_blend_ps(sysv4_m16_13, sysv4_v512f_20, p4);
    __mmask64 sysv4_m64_24 = _kor_mask64(sysv4_m64_8, p3);
    if ((unsigned long long)sysv4_m16_12 & 1ULL) {
    __m128 sysv4_v128f_25 = _mm_min_ps(p2, p2);
    __m256i sysv4_v256i_26 = _mm256_set1_epi32(0x33);
    __m256i sysv4_v256i_27 = _mm256_sub_epi64(sysv4_v256i_26, sysv4_v256i_26);
    }
    g_v512d[3] = sysv4_v512d_4;
    g_v128f[7] = p2;
    g_m16[5] = p1;
    return sysv4_v256f_7;
}

CONV_REG __m128i abi_reg_5(__m512i p0, __mmask32 p1, __mmask64 p2) {
    __mmask16 reg5_m16_1 = (__mmask16)0x5A5A;
    __m512i reg5_v512i_2 = _mm512_broadcastmw_epi32(reg5_m16_1);
    __m128i reg5_v128i_3 = g_v128i[4];
    __m128i reg5_v128i_4 = _mm_unpacklo_epi32(reg5_v128i_3, reg5_v128i_3);
    __m256i reg5_v256i_5 = g_v256i[0];
    __m256i reg5_v256i_6 = _mm256_unpacklo_epi32(reg5_v256i_5, reg5_v256i_5);
    __m512i reg5_v512i_7 = _mm512_and_si512(reg5_v512i_2, reg5_v512i_2);
    __m512 reg5_v512f_8 = g_v512f[1];
    __m512 reg5_v512f_9 = _mm512_mask_add_ps(reg5_v512f_8, reg5_m16_1, reg5_v512f_8, reg5_v512f_8);
    __m128d reg5_v128d_10 = _mm_set1_pd(2.0);
    __m128d reg5_v128d_11 = _mm_fmadd_pd(reg5_v128d_10, reg5_v128d_10, reg5_v128d_10);
    __m512d reg5_v512d_12 = g_v512d[7];
    __m128 reg5_v128f_13 = _mm_set1_ps(1.0f);
    __mmask8 reg5_m8_14 = abi_sysv_0(p2, reg5_v512d_12, reg5_v128f_13);
    __mmask16 reg5_m16_15 = _kor_mask16(reg5_m16_1, reg5_m16_1);
    __m512i reg5_v512i_16 = _mm512_mask_add_epi32(reg5_v512i_7, reg5_m16_15, reg5_v512i_2, reg5_v512i_2);
    __m128d reg5_v128d_17 = _mm_sqrt_pd(reg5_v128d_11);
    if ((unsigned long long)reg5_m16_1 & 1ULL) {
    __mmask64 reg5_m64_18 = _mm512_movepi8_mask(reg5_v512i_2);
    __m512 reg5_v512f_19 = _mm512_scalef_ps(reg5_v512f_8, reg5_v512f_8);
    __m256 reg5_v256f_20 = _mm256_set1_ps(1.0f);
    __m256 reg5_v256f_21 = _mm256_add_ps(reg5_v256f_20, reg5_v256f_20);
    __m512i reg5_v512i_22 = _mm512_broadcastmb_epi64(reg5_m8_14);
    }
    __mmask16 reg5_m16_23 = _knot_mask16(reg5_m16_1);
    __m256 reg5_v256f_24 = g_v256f[1];
    __m256 reg5_v256f_25 = _mm256_min_ps(reg5_v256f_24, reg5_v256f_24);
    __m512 reg5_v512f_26 = _mm512_maskz_mul_ps(reg5_m16_1, reg5_v512f_8, reg5_v512f_9);
    __m512 reg5_v512f_27 = _mm512_mul_ps(reg5_v512f_26, reg5_v512f_9);
    g_v512d[3] = reg5_v512d_12;
    g_v128d[4] = reg5_v128d_17;
    g_v128i[6] = reg5_v128i_4;
    g_m8[6] = reg5_m8_14;
    g_m16[7] = reg5_m16_1;
    g_m32[4] = p1;
    return reg5_v128i_4;
}

CONV_MS __m256 abi_ms_6(__m256i p0, __m128i p1, __mmask32 p2, __mmask16 p3, __mmask64 p4, __mmask64 p5, __mmask8 p6, __m128 p7, __m128i p8, __m256i p9) {
    __mmask32 ms6_m32_1 = _mm512_kunpackw(p3, p3);
    __m128i ms6_v128i_2 = _mm_slli_epi32(p8, 3);
    __m512i ms6_v512i_3 = _mm512_set1_epi32(0x55);
    __m512i ms6_v512i_4 = _mm512_ternarylogic_epi32(ms6_v512i_3, ms6_v512i_3, ms6_v512i_3, 0x96);
    __m128d ms6_v128d_5 = g_v128d[7];
    __m128d ms6_v128d_6 = _mm_max_pd(ms6_v128d_5, ms6_v128d_5);
    __m512i ms6_v512i_7 = _mm512_permutexvar_epi32(ms6_v512i_3, ms6_v512i_4);
    __m512 ms6_v512f_8 = g_v512f[6];
    __mmask16 ms6_m16_9 = _mm512_cmp_ps_mask(ms6_v512f_8, ms6_v512f_8, _CMP_LT_OQ);
    __m128 ms6_v128f_10 = _mm_sub_ps(p7, p7);
    __m256d ms6_v256d_11 = _mm256_set1_pd(2.0);
    __m256d ms6_v256d_12 = _mm256_sqrt_pd(ms6_v256d_11);
    __m128i ms6_v128i_13 = abi_reg_5(ms6_v512i_3, p2, p4);
    __m256d ms6_v256d_14 = _mm256_max_pd(ms6_v256d_12, ms6_v256d_12);
    __m512 ms6_v512f_15 = _mm512_maskz_mul_ps(ms6_m16_9, ms6_v512f_8, ms6_v512f_8);
    __m128 ms6_v128f_16 = _mm_min_ps(ms6_v128f_10, p7);
    __m128i ms6_v128i_17 = _mm_add_epi64(ms6_v128i_2, p8);
    __m128 ms6_v128f_18 = _mm_unpacklo_ps(p7, p7);
    __m512i ms6_v512i_19 = _mm512_sub_epi32(ms6_v512i_7, ms6_v512i_4);
    __m128 ms6_v128f_20 = _mm_sqrt_ps(ms6_v128f_16);
    __m128i ms6_v128i_21 = _mm_and_si128(ms6_v128i_2, ms6_v128i_13);
    __mmask16 ms6_m16_22 = _kand_mask16(ms6_m16_9, ms6_m16_9);
    if ((unsigned long long)ms6_m16_9 & 1ULL) {
    __m128 ms6_v128f_23 = _mm_fmadd_ps(ms6_v128f_18, p7, p7);
    __m512i ms6_v512i_24 = _mm512_sub_epi32(ms6_v512i_4, ms6_v512i_4);
    __m256 ms6_v256f_25 = _mm256_set1_ps(1.0f);
    __m256 ms6_v256f_26 = _mm256_sqrt_ps(ms6_v256f_25);
    __m512d ms6_v512d_27 = g_v512d[7];
    __m512d ms6_v512d_28 = _mm512_sqrt_pd(ms6_v512d_27);
    __m512d ms6_v512d_29 = _mm512_mask_blend_pd(p6, ms6_v512d_28, ms6_v512d_27);
    }
    g_v512f[0] = ms6_v512f_8;
    g_v256d[1] = ms6_v256d_12;
    g_v128d[0] = ms6_v128d_6;
    __m256 ms6_v256f_30 = g_v256f[7];
    return ms6_v256f_30;
}

CONV_MS __m128 abi_ms_7(__m256i p0, __m256i p1, __mmask32 p2, __m512i p3, __m256 p4) {
    __m512i ms7_v512i_1 = _mm512_and_si512(p3, p3);
    __m512i ms7_v512i_2 = _mm512_slli_epi32(ms7_v512i_1, 3);
    __m128 ms7_v128f_3 = _mm_set1_ps(1.0f);
    __m128 ms7_v128f_4 = _mm_unpacklo_ps(ms7_v128f_3, ms7_v128f_3);
    __m128i ms7_v128i_5 = _mm_set1_epi32(0x11);
    __m128i ms7_v128i_6 = _mm_sub_epi64(ms7_v128i_5, ms7_v128i_5);
    __mmask64 ms7_m64_7 = g_m64[5];
    __m128i ms7_v128i_8 = abi_reg_5(ms7_v512i_1, p2, ms7_m64_7);
    __m512 ms7_v512f_9 = _mm512_set1_ps(1.0f);
    __m512 ms7_v512f_10 = _mm512_unpacklo_ps(ms7_v512f_9, ms7_v512f_9);
    __m256i ms7_v256i_11 = _mm256_add_epi64(p0, p0);
    __m128i ms7_v128i_12 = _mm_and_si128(ms7_v128i_6, ms7_v128i_6);
    __mmask16 ms7_m16_13 = g_m16[2];
    __mmask16 ms7_m16_14 = _kxor_mask16(ms7_m16_13, ms7_m16_13);
    for (int ms7_i14 = 0; ms7_i14 < (int)((unsigned)ms7_m16_14 & 7); ms7_i14++) {
    __m512 ms7_v512f_16 = _mm512_unpacklo_ps(ms7_v512f_9, ms7_v512f_10);
    __m256d ms7_v256d_17 = g_v256d[4];
    __m256d ms7_v256d_18 = _mm256_max_pd(ms7_v256d_17, ms7_v256d_17);
    __m128i ms7_v128i_19 = _mm_sub_epi32(ms7_v128i_6, ms7_v128i_12);
    __m128 ms7_v128f_20 = _mm_sub_ps(ms7_v128f_4, ms7_v128f_3);
    }
    __m512 ms7_v512f_21 = _mm512_mask_blend_ps(ms7_m16_13, ms7_v512f_10, ms7_v512f_9);
    __m512 ms7_v512f_22 = _mm512_unpacklo_ps(ms7_v512f_21, ms7_v512f_10);
    __m128i ms7_v128i_23 = _mm_slli_epi32(ms7_v128i_6, 3);
    __m512d ms7_v512d_24 = _mm512_set1_pd(2.0);
    __mmask8 ms7_m8_25 = _mm512_cmp_pd_mask(ms7_v512d_24, ms7_v512d_24, _CMP_GE_OQ);
    __m256i ms7_v256i_26 = _mm256_sub_epi64(ms7_v256i_11, p1);
    __m128i ms7_v128i_27 = _mm_add_epi64(ms7_v128i_8, ms7_v128i_5);
    for (int ms7_i27 = 0; ms7_i27 < (int)((unsigned)ms7_m16_13 & 7); ms7_i27++) {
    __m512d ms7_v512d_29 = _mm512_max_pd(ms7_v512d_24, ms7_v512d_24);
    __mmask64 ms7_m64_30 = _mm512_movepi8_mask(p3);
    }
    g_v256i[3] = p1;
    g_v128i[6] = ms7_v128i_8;
    g_m8[5] = ms7_m8_25;
    g_m16[2] = ms7_m16_13;
    g_m32[1] = p2;
    g_m64[3] = ms7_m64_7;
    return ms7_v128f_4;
}

CONV_VEC __m512d abi_vec_8(__mmask32 p0, __m128i p1, __m512d p2, __m512i p3, __m128i p4, __m256i p5, __m256 p6, __m128 p7, __m512 p8) {
    __m512 vec8_v512f_1 = _mm512_sqrt_ps(p8);
    __m512 vec8_v512f_2 = _mm512_min_ps(p8, p8);
    __m128 vec8_v128f_3 = _mm_fmadd_ps(p7, p7, p7);
    __mmask16 vec8_m16_4 = _mm512_movepi32_mask(p3);
    __m256d vec8_v256d_5 = g_v256d[5];
    __m256d vec8_v256d_6 = _mm256_fmadd_pd(vec8_v256d_5, vec8_v256d_5, vec8_v256d_5);
    __m512i vec8_v512i_7 = _mm512_mask_add_epi32(p3, vec8_m16_4, p3, p3);
    __mmask16 vec8_m16_8 = _mm512_movepi32_mask(vec8_v512i_7);
    __m512 vec8_v512f_9 = _mm512_scalef_ps(vec8_v512f_1, vec8_v512f_2);
    __mmask64 vec8_m64_10 = g_m64[1];
    __m256 vec8_v256f_11 = abi_sysv_4(p0, vec8_m16_8, p7, vec8_m64_10, vec8_v512f_1);
    __m256 vec8_v256f_12 = abi_sysv_4(p0, vec8_m16_8, vec8_v128f_3, vec8_m64_10, vec8_v512f_9);
    __m128 vec8_v128f_13 = _mm_fmadd_ps(p7, vec8_v128f_3, p7);
    __mmask64 vec8_m64_14 = _mm512_cmp_epi8_mask(vec8_v512i_7, p3, 1);
    __m256i vec8_v256i_15 = _mm512_cvtepi64_epi32(vec8_v512i_7);
    if ((unsigned long long)vec8_m16_4 & 1ULL) {
    __mmask16 vec8_m16_16 = _mm512_movepi32_mask(p3);
    __m128i vec8_v128i_17 = _mm_add_epi64(p4, p4);
    __m256i vec8_v256i_18 = _mm256_add_epi32(p5, p5);
    }
    __m128d vec8_v128d_19 = g_v128d[2];
    __m128d vec8_v128d_20 = _mm_mul_pd(vec8_v128d_19, vec8_v128d_19);
    __m256i vec8_v256i_21 = _mm256_add_epi32(p5, p5);
    __m512i vec8_v512i_22 = _mm512_broadcastmw_epi32(vec8_m16_4);
    __m128 vec8_v128f_23 = _mm_mul_ps(vec8_v128f_13, vec8_v128f_13);
    __m512d vec8_v512d_24 = _mm512_fmadd_pd(p2, p2, p2);
    g_v512f[5] = vec8_v512f_9;
    g_v512i[7] = vec8_v512i_7;
    g_m16[5] = vec8_m16_4;
    g_m32[0] = p0;
    return vec8_v512d_24;
}

CONV_REG __m512 abi_reg_9(__mmask8 p0, __mmask8 p1, __mmask8 p2, __m256i p3, __mmask32 p4, __m512i p5) {
    __mmask16 reg9_m16_1 = _mm512_cmp_epi32_mask(p5, p5, 2);
    __m128 reg9_v128f_2 = g_v128f[4];
    __m128 reg9_v128f_3 = _mm_unpacklo_ps(reg9_v128f_2, reg9_v128f_2);
    __m128d reg9_v128d_4 = _mm_set1_pd(2.0);
    __m128d reg9_v128d_5 = _mm_mul_pd(reg9_v128d_4, reg9_v128d_4);
    __m512 reg9_v512f_6 = _mm512_set1_ps(1.0f);
    __mmask16 reg9_m16_7 = _mm512_cmp_ps_mask(reg9_v512f_6, reg9_v512f_6, _CMP_LT_OQ);
    __m512 reg9_v512f_8 = _mm512_unpacklo_ps(reg9_v512f_6, reg9_v512f_6);
    __m128i reg9_v128i_9 = g_v128i[7];
    __m128i reg9_v128i_10 = _mm_and_si128(reg9_v128i_9, reg9_v128i_9);
    __mmask64 reg9_m64_11 = (__mmask64)0x5A5A5A5A5A5A5A5AULL;
    __m512d reg9_v512d_12 = g_v512d[3];
    __mmask8 reg9_m8_13 = abi_sysv_0(reg9_m64_11, reg9_v512d_12, reg9_v128f_3);
    __m128 reg9_v128f_14 = _mm_max_ps(reg9_v128f_2, reg9_v128f_2);
    __m256d reg9_v256d_15 = g_v256d[5];
    __m256d reg9_v256d_16 = _mm256_add_pd(reg9_v256d_15, reg9_v256d_15);
    __m512 reg9_v512f_17 = _mm512_mask_add_ps(reg9_v512f_8, reg9_m16_7, reg9_v512f_8, reg9_v512f_8);
    if ((unsigned long long)reg9_m16_1 & 1ULL) {
    __m512i reg9_v512i_18 = _mm512_ternarylogic_epi32(p5, p5, p5, 0x96);
    __m512i reg9_v512i_19 = _mm512_sub_epi32(p5, p5);
    __m128 reg9_v128f_20 = _mm_mul_ps(reg9_v128f_3, reg9_v128f_2);
    __m512i reg9_v512i_21 = _mm512_mask_add_epi32(p5, reg9_m16_1, p5, reg9_v512i_18);
    __m512i reg9_v512i_22 = _mm512_mask_blend_epi32(reg9_m16_7, reg9_v512i_19, reg9_v512i_18);
    }
    __mmask8 reg9_m8_23 = abi_sysv_0(reg9_m64_11, reg9_v512d_12, reg9_v128f_3);
    __m512i reg9_v512i_24 = _mm512_broadcastmb_epi64(p1);
    __m128i reg9_v128i_25 = _mm_unpacklo_epi32(reg9_v128i_10, reg9_v128i_9);
    __m512i reg9_v512i_26 = _mm512_ternarylogic_epi32(p5, reg9_v512i_24, p5, 0x96);
    __mmask16 reg9_m16_27 = _mm512_cmp_epi32_mask(p5, reg9_v512i_24, 2);
    __mmask16 reg9_m16_28 = _mm512_cmp_epi32_mask(p5, p5, 2);
    g_v512d[0] = reg9_v512d_12;
    g_v512i[3] = reg9_v512i_26;
    g_v128d[7] = reg9_v128d_4;
    g_m8[1] = p0;
    g_m16[3] = reg9_m16_28;
    g_m32[4] = p4;
    g_m64[6] = reg9_m64_11;
    return reg9_v512f_6;
}

CONV_SYSV __m128i abi_sysv_10(__m512 p0, __mmask16 p1, __m128i p2, __m128i p3, __mmask16 p4) {
    __m512i sysv10_v512i_1 = _mm512_set1_epi32(0x55);
    __m512i sysv10_v512i_2 = _mm512_sub_epi64(sysv10_v512i_1, sysv10_v512i_1);
    __m512i sysv10_v512i_3 = _mm512_mask_blend_epi32(p4, sysv10_v512i_2, sysv10_v512i_1);
    __mmask16 sysv10_m16_4 = _kor_mask16(p4, p4);
    __mmask8 sysv10_m8_5 = g_m8[4];
    __m512i sysv10_v512i_6 = _mm512_broadcastmb_epi64(sysv10_m8_5);
    __m128 sysv10_v128f_7 = _mm_set1_ps(1.0f);
    __m128 sysv10_v128f_8 = _mm_add_ps(sysv10_v128f_7, sysv10_v128f_7);
    __m512 sysv10_v512f_9 = _mm512_scalef_ps(p0, p0);
    __m512d sysv10_v512d_10 = _mm512_set1_pd(2.0);
    __m512d sysv10_v512d_11 = _mm512_mask_blend_pd(sysv10_m8_5, sysv10_v512d_10, sysv10_v512d_10);
    __m512i sysv10_v512i_12 = _mm512_mullo_epi32(sysv10_v512i_3, sysv10_v512i_3);
    __mmask32 sysv10_m32_13 = g_m32[7];
    __m256i sysv10_v256i_14 = g_v256i[1];
    __m256 sysv10_v256f_15 = _mm256_set1_ps(1.0f);
    __m512d sysv10_v512d_16 = abi_vec_8(sysv10_m32_13, p2, sysv10_v512d_10, sysv10_v512i_2, p3, sysv10_v256i_14, sysv10_v256f_15, sysv10_v128f_7, sysv10_v512f_9);
    __m512 sysv10_v512f_17 = _mm512_rcp14_ps(p0);
    __m512i sysv10_v512i_18 = _mm512_permutexvar_epi32(sysv10_v512i_6, sysv10_v512i_6);
    __mmask16 sysv10_m16_19 = _knot_mask16(p1);
    if ((unsigned long long)p1 & 1ULL) {
    __m512 sysv10_v512f_20 = _mm512_min_ps(sysv10_v512f_17, sysv10_v512f_9);
    __m128 sysv10_v128f_21 = _mm_mul_ps(sysv10_v128f_8, sysv10_v128f_7);
    }
    __m512d sysv10_v512d_22 = abi_vec_8(sysv10_m32_13, p3, sysv10_v512d_10, sysv10_v512i_3, p2, sysv10_v256i_14, sysv10_v256f_15, sysv10_v128f_7, p0);
    __mmask16 sysv10_m16_23 = _kxor_mask16(p4, sysv10_m16_4);
    __m128i sysv10_v128i_24 = _mm_sub_epi32(p3, p2);
    __m128i sysv10_v128i_25 = _mm_and_si128(p3, p2);
    __mmask16 sysv10_m16_26 = _kxor_mask16(sysv10_m16_19, p1);
    for (int sysv10_i26 = 0; sysv10_i26 < (int)((unsigned)sysv10_m16_23 & 7); sysv10_i26++) {
    __m128 sysv10_v128f_28 = _mm_sub_ps(sysv10_v128f_8, sysv10_v128f_8);
    __m256i sysv10_v256i_29 = _mm512_cvtepi64_epi32(sysv10_v512i_1);
    __m512 sysv10_v512f_30 = _mm512_sqrt_ps(sysv10_v512f_9);
    __m512i sysv10_v512i_31 = _mm512_xor_si512(sysv10_v512i_2, sysv10_v512i_18);
    }
    g_v512f[6] = p0;
    g_m16[3] = sysv10_m16_26;
    return p3;
}

CONV_REG __m128i abi_reg_11(__mmask8 p0, __m256 p1, __m128 p2, __mmask64 p3, __mmask32 p4, __m128i p5, __mmask8 p6) {
    __m512 reg11_v512f_1 = _mm512_set1_ps(1.0f);
    __m512 reg11_v512f_2 = _mm512_max_ps(reg11_v512f_1, reg11_v512f_1);
    __m128 reg11_v128f_3 = _mm_sqrt_ps(p2);
    __mmask16 reg11_m16_4 = (__mmask16)0x5A5A;
    __m512i reg11_v512i_5 = g_v512i[7];
    __m512i reg11_v512i_6 = _mm512_mask_blend_epi32(reg11_m16_4, reg11_v512i_5, reg11_v512i_5);
    __mmask32 reg11_m32_7 = _mm512_cmp_epi16_mask(reg11_v512i_5, reg11_v512i_5, 2);
    __m512i reg11_v512i_8 = _mm512_rol_epi32(reg11_v512i_5, 5);
    __m128d reg11_v128d_9 = _mm_set1_pd(2.0);
    __m128d reg11_v128d_10 = _mm_fmadd_pd(reg11_v128d_9, reg11_v128d_9, reg11_v128d_9);
    __m512d reg11_v512d_11 = g_v512d[0];
    __m256i reg11_v256i_12 = g_v256i[2];
    __m512d reg11_v512d_13 = abi_vec_8(reg11_m32_7, p5, reg11_v512d_11, reg11_v512i_6, p5, reg11_v256i_12, p1, reg11_v128f_3, reg11_v512f_2);
    __mmask8 reg11_m8_14 = abi_sysv_0(p3, reg11_v512d_11, reg11_v128f_3);
    __m512 reg11_v512f_15 = _mm512_cvtepi32_ps(reg11_v512i_5);
    __m512i reg11_v512i_16 = _mm512_mullo_epi32(reg11_v512i_6, reg11_v512i_6);
    for (int reg11_i16 = 0; reg11_i16 < (int)((unsigned)reg11_m16_4 & 7); reg11_i16++) {
    __m128 reg11_v128f_18 = _mm_max_ps(p2, reg11_v128f_3);
    __m512 reg11_v512f_19 = _mm512_maskz_mul_ps(reg11_m16_4, reg11_v512f_2, reg11_v512f_2);
    __m256i reg11_v256i_20 = _mm256_unpacklo_epi32(reg11_v256i_12, reg11_v256i_12);
    __m512i reg11_v512i_21 = _mm512_permutexvar_epi32(reg11_v512i_16, reg11_v512i_5);
    }
    __m512d reg11_v512d_22 = abi_vec_8(p4, p5, reg11_v512d_11, reg11_v512i_8, p5, reg11_v256i_12, p1, p2, reg11_v512f_1);
    __m128i reg11_v128i_23 = _mm_slli_epi32(p5, 3);
    __m512d reg11_v512d_24 = _mm512_fmadd_pd(reg11_v512d_13, reg11_v512d_13, reg11_v512d_13);
    __mmask16 reg11_m16_25 = _mm512_kunpackb(reg11_m8_14, p0);
    __mmask16 reg11_m16_26 = _mm512_cmp_ps_mask(reg11_v512f_2, reg11_v512f_15, _CMP_LT_OQ);
    __m512d reg11_v512d_27 = _mm512_sub_pd(reg11_v512d_24, reg11_v512d_11);
    __m512i reg11_v512i_28 = _mm512_ternarylogic_epi32(reg11_v512i_16, reg11_v512i_8, reg11_v512i_5, 0x96);
    g_v512f[2] = reg11_v512f_2;
    g_v512d[1] = reg11_v512d_27;
    g_v512i[6] = reg11_v512i_16;
    g_m32[2] = p4;
    return p5;
}

CONV_MS __m256i abi_ms_12(__m512i p0, __m128i p1, __m128i p2, __mmask16 p3, __m256i p4, __m512 p5, __m128 p6, __m256 p7, __mmask8 p8, __m128i p9) {
    __m512i ms12_v512i_1 = _mm512_permutexvar_epi32(p0, p0);
    __m256 ms12_v256f_2 = _mm256_min_ps(p7, p7);
    __mmask8 ms12_m8_3 = _kand_mask8(p8, p8);
    __mmask64 ms12_m64_4 = (__mmask64)0x5A5A5A5A5A5A5A5AULL;
    __mmask64 ms12_m64_5 = _kor_mask64(ms12_m64_4, ms12_m64_4);
    __m512i ms12_v512i_6 = _mm512_ternarylogic_epi32(p0, ms12_v512i_1, ms12_v512i_1, 0x96);
    __m256i ms12_v256i_7 = _mm256_add_epi32(p4, p4);
    __mmask32 ms12_m32_8 = (__mmask32)0x5A5A5A5A;
    __m512d ms12_v512d_9 = g_v512d[5];
    __m512d ms12_v512d_10 = abi_vec_8(ms12_m32_8, p1, ms12_v512d_9, ms12_v512i_1, p9, p4, p7, p6, p5);
    __m256 ms12_v256f_11 = _mm256_sqrt_ps(p7);
    __m128 ms12_v128f_12 = _mm512_extractf32x4_ps(p5, 2);
    __m128 ms12_v128f_13 = _mm512_extractf32x4_ps(p5, 2);
    if ((unsigned long long)p3 & 1ULL) {
    __m128 ms12_v128f_14 = _mm_add_ps(ms12_v128f_12, ms12_v128f_12);
    __m512 ms12_v512f_15 = _mm512_cvtepi32_ps(ms12_v512i_6);
    }
    __mmask8 ms12_m8_16 = abi_sysv_0(ms12_m64_4, ms12_v512d_10, p6);
    __m256d ms12_v256d_17 = _mm256_set1_pd(2.0);
    __m256d ms12_v256d_18 = _mm256_sub_pd(ms12_v256d_17, ms12_v256d_17);
    __m512 ms12_v512f_19 = _mm512_sub_ps(p5, p5);
    __m128i ms12_v128i_20 = _mm_slli_epi32(p1, 3);
    if ((unsigned long long)p3 & 1ULL) {
    __m256d ms12_v256d_21 = _mm256_add_pd(ms12_v256d_18, ms12_v256d_18);
    __m256i ms12_v256i_22 = _mm256_mullo_epi32(ms12_v256i_7, p4);
    __m256 ms12_v256f_23 = _mm256_sub_ps(ms12_v256f_11, p7);
    __m256 ms12_v256f_24 = _mm256_sub_ps(ms12_v256f_11, ms12_v256f_23);
    }
    g_v512f[7] = ms12_v512f_19;
    g_v256i[6] = ms12_v256i_7;
    return ms12_v256i_7;
}

CONV_SYSV __m128i abi_sysv_13(__m128 p0, __m256 p1, __m256i p2, __m256 p3, __m512 p4, __m128i p5, __m128i p6, __mmask8 p7) {
    __mmask16 sysv13_m16_1 = (__mmask16)0x5A5A;
    __m512 sysv13_v512f_2 = _mm512_mask_blend_ps(sysv13_m16_1, p4, p4);
    __m256i sysv13_v256i_3 = _mm256_sub_epi32(p2, p2);
    __m512 sysv13_v512f_4 = _mm512_mask_add_ps(sysv13_v512f_2, sysv13_m16_1, sysv13_v512f_2, p4);
    __m128i sysv13_v128i_5 = _mm_and_si128(p6, p5);
    __mmask16 sysv13_m16_6 = _kor_mask16(sysv13_m16_1, sysv13_m16_1);
    __mmask16 sysv13_m16_7 = _kor_mask16(sysv13_m16_1, sysv13_m16_6);
    __m128d sysv13_v128d_8 = g_v128d[1];
    __m128d sysv13_v128d_9 = _mm_sub_pd(sysv13_v128d_8, sysv13_v128d_8);
    __m512 sysv13_v512f_10 = _mm512_mask_add_ps(p4, sysv13_m16_6, sysv13_v512f_4, sysv13_v512f_4);
    __mmask32 sysv13_m32_11 = g_m32[5];
    __m512d sysv13_v512d_12 = _mm512_set1_pd(2.0);
    __m512i sysv13_v512i_13 = _mm512_set1_epi32(0x55);
    __m512d sysv13_v512d_14 = abi_vec_8(sysv13_m32_11, p6, sysv13_v512d_12, sysv13_v512i_13, p5, p2, p1, p0, p4);
    __m512i sysv13_v512i_15 = _mm512_sub_epi32(sysv13_v512i_13, sysv13_v512i_13);
    __m512i sysv13_v512i_16 = _mm512_cvtps_epi32(sysv13_v512f_4);
    for (int sysv13_i16 = 0; sysv13_i16 < (int)((unsigned)sysv13_m16_7 & 7); sysv13_i16++) {
    __mmask16 sysv13_m16_18 = _mm512_kunpackb(p7, p7);
    __m256i sysv13_v256i_19 = _mm256_add_epi32(sysv13_v256i_3, sysv13_v256i_3);
    }
    __m512i sysv13_v512i_20 = _mm512_broadcastmb_epi64(p7);
    __m256 sysv13_v256f_21 = _mm256_mul_ps(p3, p1);
    __m512i sysv13_v512i_22 = _mm512_cvtps_epi32(p4);
    __m256i sysv13_v256i_23 = _mm512_cvtepi64_epi32(sysv13_v512i_15);
    __m512i sysv13_v512i_24 = _mm512_broadcast_i32x4(p5);
    if ((unsigned long long)sysv13_m16_6 & 1ULL) {
    __mmask8 sysv13_m8_25 = _kand_mask8(p7, p7);
    __m512 sysv13_v512f_26 = _mm512_add_ps(sysv13_v512f_4, sysv13_v512f_10);
    __m128i sysv13_v128i_27 = _mm_sub_epi64(p5, sysv13_v128i_5);
    }
    g_v512f[5] = sysv13_v512f_2;
    g_v256f[4] = sysv13_v256f_21;
    g_v128f[4] = p0;
    g_m16[2] = sysv13_m16_1;
    g_m32[0] = sysv13_m32_11;
    return p5;
}

CONV_VEC __m128 abi_vec_14(__mmask16 p0, __mmask32 p1) {
    __m512 vec14_v512f_1 = _mm512_set1_ps(1.0f);
    __m512 vec14_v512f_2 = _mm512_add_ps(vec14_v512f_1, vec14_v512f_1);
    __mmask32 vec14_m32_3 = _kxor_mask32(p1, p1);
    __m512 vec14_v512f_4 = _mm512_unpacklo_ps(vec14_v512f_1, vec14_v512f_1);
    __m128i vec14_v128i_5 = g_v128i[0];
    __m128i vec14_v128i_6 = _mm_unpacklo_epi32(vec14_v128i_5, vec14_v128i_5);
    __m512i vec14_v512i_7 = g_v512i[3];
    __m512i vec14_v512i_8 = _mm512_rol_epi32(vec14_v512i_7, 5);
    __m128i vec14_v128i_9 = abi_sysv_10(vec14_v512f_4, p0, vec14_v128i_5, vec14_v128i_5, p0);
    __m128 vec14_v128f_10 = g_v128f[5];
    __m256 vec14_v256f_11 = g_v256f[0];
    __m256i vec14_v256i_12 = _mm256_set1_epi32(0x33);
    __mmask8 vec14_m8_13 = g_m8[1];
    __m128i vec14_v128i_14 = abi_sysv_13(vec14_v128f_10, vec14_v256f_11, vec14_v256i_12, vec14_v256f_11, vec14_v512f_1, vec14_v128i_5, vec14_v128i_5, vec14_m8_13);
    __m512 vec14_v512f_15 = _mm512_maskz_mul_ps(p0, vec14_v512f_4, vec14_v512f_1);
    __m512d vec14_v512d_16 = _mm512_set1_pd(2.0);
    __m512d vec14_v512d_17 = _mm512_add_pd(vec14_v512d_16, vec14_v512d_16);
    for (int vec14_i17 = 0; vec14_i17 < (int)((unsigned)p0 & 7); vec14_i17++) {
    __m128 vec14_v128f_19 = _mm512_extractf32x4_ps(vec14_v512f_1, 2);
    __m128d vec14_v128d_20 = g_v128d[3];
    __m128d vec14_v128d_21 = _mm_mul_pd(vec14_v128d_20, vec14_v128d_20);
    }
    __mmask64 vec14_m64_22 = g_m64[3];
    __mmask8 vec14_m8_23 = abi_sysv_0(vec14_m64_22, vec14_v512d_17, vec14_v128f_10);
    __m128d vec14_v128d_24 = _mm_set1_pd(2.0);
    __m128d vec14_v128d_25 = _mm_add_pd(vec14_v128d_24, vec14_v128d_24);
    __mmask8 vec14_m8_26 = _kand_mask8(vec14_m8_23, vec14_m8_13);
    __m256i vec14_v256i_27 = _mm256_mullo_epi32(vec14_v256i_12, vec14_v256i_12);
    __m128d vec14_v128d_28 = _mm_fmadd_pd(vec14_v128d_25, vec14_v128d_25, vec14_v128d_24);
    __m128d vec14_v128d_29 = _mm_add_pd(vec14_v128d_24, vec14_v128d_25);
    __m256i vec14_v256i_30 = _mm256_xor_si256(vec14_v256i_12, vec14_v256i_12);
    for (int vec14_i30 = 0; vec14_i30 < (int)((unsigned)p0 & 7); vec14_i30++) {
    __m256i vec14_v256i_32 = _mm256_add_epi64(vec14_v256i_27, vec14_v256i_27);
    __m128 vec14_v128f_33 = _mm_sqrt_ps(vec14_v128f_10);
    __m128 vec14_v128f_34 = _mm_add_ps(vec14_v128f_33, vec14_v128f_33);
    __m128 vec14_v128f_35 = _mm_max_ps(vec14_v128f_33, vec14_v128f_34);
    }
    g_v512f[4] = vec14_v512f_1;
    g_v512d[2] = vec14_v512d_16;
    g_v512i[2] = vec14_v512i_7;
    g_v256f[4] = vec14_v256f_11;
    g_v256i[5] = vec14_v256i_12;
    g_v128f[7] = vec14_v128f_10;
    g_v128d[2] = vec14_v128d_28;
    g_m16[2] = p0;
    return vec14_v128f_10;
}

CONV_SYSV __m128i abi_sysv_15(__m512d p0, __m128i p1, __mmask64 p2, __m512 p3) {
    __m256i sysv15_v256i_1 = _mm256_set1_epi32(0x33);
    __m256i sysv15_v256i_2 = _mm256_unpacklo_epi32(sysv15_v256i_1, sysv15_v256i_1);
    __m128i sysv15_v128i_3 = _mm_mullo_epi32(p1, p1);
    __m256i sysv15_v256i_4 = _mm256_xor_si256(sysv15_v256i_1, sysv15_v256i_1);
    __m128 sysv15_v128f_5 = g_v128f[4];
    __m128 sysv15_v128f_6 = _mm_mul_ps(sysv15_v128f_5, sysv15_v128f_5);
    __m512 sysv15_v512f_7 = _mm512_unpacklo_ps(p3, p3);
    __m256i sysv15_v256i_8 = _mm256_mullo_epi32(sysv15_v256i_4, sysv15_v256i_2);
    __m256d sysv15_v256d_9 = g_v256d[0];
    __m256d sysv15_v256d_10 = _mm256_add_pd(sysv15_v256d_9, sysv15_v256d_9);
    __m256i sysv15_v256i_11 = _mm256_add_epi32(sysv15_v256i_1, sysv15_v256i_2);
    __m512d sysv15_v512d_12 = _mm512_fmadd_pd(p0, p0, p0);
    __mmask32 sysv15_m32_13 = g_m32[4];
    __mmask16 sysv15_m16_14 = (__mmask16)0x5A5A;
    __mmask8 sysv15_m8_15 = (__mmask8)0xA5;
    __m256 sysv15_v256f_16 = abi_ms_6(sysv15_v256i_2, p1, sysv15_m32_13, sysv15_m16_14, p2, p2, sysv15_m8_15, sysv15_v128f_6, p1, sysv15_v256i_11);
    __m128d sysv15_v128d_17 = _mm_set1_pd(2.0);
    __m128d sysv15_v128d_18 = _mm_sub_pd(sysv15_v128d_17, sysv15_v128d_17);
    __m256 sysv15_v256f_19 = _mm256_min_ps(sysv15_v256f_16, sysv15_v256f_16);
    __m512i sysv15_v512i_20 = _mm512_set1_epi32(0x55);
    __m512i sysv15_v512i_21 = _mm512_sub_epi64(sysv15_v512i_20, sysv15_v512i_20);
    if ((unsigned long long)sysv15_m16_14 & 1ULL) {
    __m512d sysv15_v512d_22 = _mm512_mul_pd(p0, p0);
    __m512d sysv15_v512d_23 = _mm512_max_pd(sysv15_v512d_12, sysv15_v512d_22);
    __mmask64 sysv15_m64_24 = _mm512_cmp_epi8_mask(sysv15_v512i_20, sysv15_v512i_21, 1);
    __m512i sysv15_v512i_25 = _mm512_slli_epi32(sysv15_v512i_20, 3);
    }
    __m512i sysv15_v512i_26 = _mm512_permutexvar_epi32(sysv15_v512i_20, sysv15_v512i_20);
    __m128 sysv15_v128f_27 = _mm_add_ps(sysv15_v128f_6, sysv15_v128f_6);
    __m512i sysv15_v512i_28 = _mm512_sub_epi64(sysv15_v512i_21, sysv15_v512i_20);
    __m128i sysv15_v128i_29 = _mm_and_si128(sysv15_v128i_3, sysv15_v128i_3);
    __m512i sysv15_v512i_30 = _mm512_cvtps_epi32(sysv15_v512f_7);
    g_v512i[5] = sysv15_v512i_26;
    g_v256d[4] = sysv15_v256d_10;
    g_v128d[4] = sysv15_v128d_17;
    g_v128i[6] = p1;
    g_m8[7] = sysv15_m8_15;
    g_m32[5] = sysv15_m32_13;
    g_m64[3] = p2;
    return sysv15_v128i_3;
}

CONV_MS __m512d abi_ms_16(__m128 p0, __m512 p1) {
    __m512i ms16_v512i_1 = _mm512_set1_epi32(0x55);
    __m512i ms16_v512i_2 = _mm512_and_si512(ms16_v512i_1, ms16_v512i_1);
    __m512 ms16_v512f_3 = _mm512_max_ps(p1, p1);
    __mmask64 ms16_m64_4 = g_m64[7];
    __mmask64 ms16_m64_5 = _kor_mask64(ms16_m64_4, ms16_m64_4);
    __mmask16 ms16_m16_6 = g_m16[2];
    __mmask16 ms16_m16_7 = _kor_mask16(ms16_m16_6, ms16_m16_6);
    __m128 ms16_v128f_8 = _mm_max_ps(p0, p0);
    __m512i ms16_v512i_9 = _mm512_permutexvar_epi32(ms16_v512i_2, ms16_v512i_1);
    __m128 ms16_v128f_10 = _mm512_extractf32x4_ps(ms16_v512f_3, 2);
    __m512d ms16_v512d_11 = g_v512d[5];
    __m128i ms16_v128i_12 = g_v128i[4];
    __m128i ms16_v128i_13 = abi_sysv_15(ms16_v512d_11, ms16_v128i_12, ms16_m64_4, ms16_v512f_3);
    __m256 ms16_v256f_14 = g_v256f[7];
    __m256 ms16_v256f_15 = _mm256_sub_ps(ms16_v256f_14, ms16_v256f_14);
    __m128 ms16_v128f_16 = _mm_add_ps(ms16_v128f_10, ms16_v128f_8);
    __mmask8 ms16_m8_17 = g_m8[7];
    __m512i ms16_v512i_18 = _mm512_broadcastmb_epi64(ms16_m8_17);
    __m128i ms16_v128i_19 = _mm_add_epi64(ms16_v128i_12, ms16_v128i_12);
    __mmask32 ms16_m32_20 = (__mmask32)0x5A5A5A5A;
    __m256 ms16_v256f_21 = abi_sysv_4(ms16_m32_20, ms16_m16_7, ms16_v128f_16, ms16_m64_4, ms16_v512f_3);
    __m512i ms16_v512i_22 = _mm512_sub_epi64(ms16_v512i_2, ms16_v512i_9);
    __m512i ms16_v512i_23 = _mm512_sub_epi32(ms16_v512i_2, ms16_v512i_22);
    __m128i ms16_v128i_24 = _mm_slli_epi32(ms16_v128i_13, 3);
    __m256 ms16_v256f_25 = _mm256_sqrt_ps(ms16_v256f_21);
    g_v512d[4] = ms16_v512d_11;
    g_v256f[2] = ms16_v256f_15;
    g_v128i[6] = ms16_v128i_19;
    g_m32[3] = ms16_m32_20;
    g_m64[1] = ms16_m64_4;
    return ms16_v512d_11;
}

CONV_VEC __mmask16 abi_vec_17(__m512d p0, __mmask8 p1, __m512i p2, __m256 p3, __mmask64 p4, __m512d p5) {
    __m128 vec17_v128f_1 = g_v128f[6];
    __m128 vec17_v128f_2 = _mm_min_ps(vec17_v128f_1, vec17_v128f_1);
    __m128i vec17_v128i_3 = g_v128i[0];
    __m128i vec17_v128i_4 = _mm_add_epi32(vec17_v128i_3, vec17_v128i_3);
    __m512 vec17_v512f_5 = g_v512f[6];
    __m128 vec17_v128f_6 = _mm512_extractf32x4_ps(vec17_v512f_5, 2);
    __m256i vec17_v256i_7 = _mm256_set1_epi32(0x33);
    __m256i vec17_v256i_8 = _mm256_add_epi32(vec17_v256i_7, vec17_v256i_7);
    __m512 vec17_v512f_9 = _mm512_broadcast_f32x4(vec17_v128f_2);
    __m256 vec17_v256f_10 = _mm256_add_ps(p3, p3);
    __mmask16 vec17_m16_11 = (__mmask16)0x5A5A;
    __m512i vec17_v512i_12 = _mm512_mask_add_epi32(p2, vec17_m16_11, p2, p2);
    __m256i vec17_v256i_13 = _mm256_sub_epi64(vec17_v256i_8, vec17_v256i_8);
    __mmask32 vec17_m32_14 = (__mmask32)0x5A5A5A5A;
    __mmask32 vec17_m32_15 = abi_ms_3(vec17_v512f_9, vec17_v256i_7, vec17_m32_14, vec17_m16_11, vec17_v256i_8, vec17_v512f_9, vec17_v128i_4, vec17_m16_11, vec17_v128f_1, vec17_v128f_2);
    __m512 vec17_v512f_16 = _mm512_cvtepi32_ps(vec17_v512i_12);
    __m128i vec17_v128i_17 = _mm_mullo_epi32(vec17_v128i_3, vec17_v128i_4);
    __m512i vec17_v512i_18 = _mm512_mullo_epi32(vec17_v512i_12, p2);
    __m512 vec17_v512f_19 = _mm512_cvtepi32_ps(p2);
    __m256 vec17_v256f_20 = _mm256_fmadd_ps(p3, vec17_v256f_10, p3);
    if ((unsigned long long)vec17_m16_11 & 1ULL) {
    __m512i vec17_v512i_21 = _mm512_and_si512(vec17_v512i_12, vec17_v512i_18);
    __m256i vec17_v256i_22 = _mm256_mullo_epi32(vec17_v256i_8, vec17_v256i_8);
    __m512 vec17_v512f_23 = _mm512_sub_ps(vec17_v512f_19, vec17_v512f_19);
    __mmask16 vec17_m16_24 = _knot_mask16(vec17_m16_11);
    __m256i vec17_v256i_25 = _mm256_add_epi64(vec17_v256i_13, vec17_v256i_22);
    }
    __m512i vec17_v512i_26 = _mm512_unpacklo_epi32(p2, vec17_v512i_12);
    __m256 vec17_v256f_27 = _mm256_add_ps(vec17_v256f_10, p3);
    __m128d vec17_v128d_28 = g_v128d[1];
    __m128d vec17_v128d_29 = _mm_fmadd_pd(vec17_v128d_28, vec17_v128d_28, vec17_v128d_28);
    __m128i vec17_v128i_30 = _mm_sub_epi64(vec17_v128i_17, vec17_v128i_17);
    g_v512i[6] = vec17_v512i_26;
    g_v256f[3] = vec17_v256f_10;
    g_v128d[0] = vec17_v128d_28;
    g_v128i[6] = vec17_v128i_17;
    g_m32[6] = vec17_m32_15;
    g_m64[5] = p4;
    return vec17_m16_11;
}

CONV_SYSV __m512i abi_sysv_18(__m128i p0, __m512i p1, __mmask32 p2, __mmask64 p3, __mmask32 p4, __mmask64 p5) {
    __m512d sysv18_v512d_1 = _mm512_set1_pd(2.0);
    __m512d sysv18_v512d_2 = _mm512_add_pd(sysv18_v512d_1, sysv18_v512d_1);
    __m128 sysv18_v128f_3 = g_v128f[4];
    __m128 sysv18_v128f_4 = _mm_min_ps(sysv18_v128f_3, sysv18_v128f_3);
    __m128d sysv18_v128d_5 = g_v128d[5];
    __m128d sysv18_v128d_6 = _mm_sub_pd(sysv18_v128d_5, sysv18_v128d_5);
    __mmask8 sysv18_m8_7 = g_m8[2];
    __m512i sysv18_v512i_8 = _mm512_broadcastmb_epi64(sysv18_m8_7);
    __m512i sysv18_v512i_9 = _mm512_add_epi32(p1, sysv18_v512i_8);
    __m256i sysv18_v256i_10 = g_v256i[2];
    __m256 sysv18_v256f_11 = _mm256_set1_ps(1.0f);
    __m512 sysv18_v512f_12 = g_v512f[2];
    __m512d sysv18_v512d_13 = abi_vec_8(p4, p0, sysv18_v512d_1, p1, p0, sysv18_v256i_10, sysv18_v256f_11, sysv18_v128f_3, sysv18_v512f_12);
    __mmask16 sysv18_m16_14 = abi_vec_17(sysv18_v512d_13, sysv18_m8_7, sysv18_v512i_9, sysv18_v256f_11, p3, sysv18_v512d_2);
    __m256 sysv18_v256f_15 = _mm256_fmadd_ps(sysv18_v256f_11, sysv18_v256f_11, sysv18_v256f_11);
    __m512 sysv18_v512f_16 = _mm512_max_ps(sysv18_v512f_12, sysv18_v512f_12);
    __m128i sysv18_v128i_17 = _mm_unpacklo_epi32(p0, p0);
    if ((unsigned long long)sysv18_m16_14 & 1ULL) {
    __m512d sysv18_v512d_18 = _mm512_add_pd(sysv18_v512d_2, sysv18_v512d_2);
    __m512 sysv18_v512f_19 = _mm512_min_ps(sysv18_v512f_16, sysv18_v512f_12);
    __mmask8 sysv18_m8_20 = _kand_mask8(sysv18_m8_7, sysv18_m8_7);
    __mmask16 sysv18_m16_21 = _kand_mask16(sysv18_m16_14, sysv18_m16_14);
    }
    __m512d sysv18_v512d_22 = abi_vec_8(p2, p0, sysv18_v512d_1, sysv18_v512i_8, sysv18_v128i_17, sysv18_v256i_10, sysv18_v256f_15, sysv18_v128f_4, sysv18_v512f_16);
    __m128d sysv18_v128d_23 = _mm_sub_pd(sysv18_v128d_6, sysv18_v128d_6);
    __mmask16 sysv18_m16_24 = _mm512_kunpackb(sysv18_m8_7, sysv18_m8_7);
    __m128d sysv18_v128d_25 = _mm_sub_pd(sysv18_v128d_6, sysv18_v128d_23);
    g_v512f[3] = sysv18_v512f_12;
    g_v512d[6] = sysv18_v512d_2;
    g_v256i[0] = sysv18_v256i_10;
    return sysv18_v512i_9;
}

CONV_REG __m256 abi_reg_19(__m128 p0, __m512d p1, __m512 p2, __mmask64 p3, __m256i p4, __m256 p5, __m256i p6, __mmask16 p7, __mmask64 p8) {
    __m256d reg19_v256d_1 = _mm512_extractf64x4_pd(p1, 1);
    __m256i reg19_v256i_2 = _mm256_slli_epi32(p4, 3);
    __m256d reg19_v256d_3 = _mm256_add_pd(reg19_v256d_1, reg19_v256d_1);
    __m512i reg19_v512i_4 = _mm512_set1_epi32(0x55);
    __m512i reg19_v512i_5 = _mm512_and_si512(reg19_v512i_4, reg19_v512i_4);
    __m512i reg19_v512i_6 = _mm512_sub_epi32(reg19_v512i_4, reg19_v512i_5);
    __m512 reg19_v512f_7 = _mm512_min_ps(p2, p2);
    __mmask8 reg19_m8_8 = g_m8[1];
    __mmask16 reg19_m16_9 = abi_vec_17(p1, reg19_m8_8, reg19_v512i_4, p5, p8, p1);
    __m128 reg19_v128f_10 = _mm_mul_ps(p0, p0);
    __m128d reg19_v128d_11 = g_v128d[3];
    __m128d reg19_v128d_12 = _mm_mul_pd(reg19_v128d_11, reg19_v128d_11);
    __m512d reg19_v512d_13 = _mm512_sqrt_pd(p1);
    __m256i reg19_v256i_14 = _mm256_unpacklo_epi32(p4, p4);
    __m512i reg19_v512i_15 = _mm512_add_epi32(reg19_v512i_6, reg19_v512i_6);
    __m128 reg19_v128f_16 = _mm512_extractf32x4_ps(p2, 2);
    for (int reg19_i16 = 0; reg19_i16 < (int)((unsigned)reg19_m16_9 & 7); reg19_i16++) {
    __m512d reg19_v512d_18 = _mm512_max_pd(reg19_v512d_13, reg19_v512d_13);
    __m512 reg19_v512f_19 = _mm512_cvtepi32_ps(reg19_v512i_4);
    }
    __m128i reg19_v128i_20 = _mm_set1_epi32(0x11);
    __m128i reg19_v128i_21 = abi_sysv_15(p1, reg19_v128i_20, p3, reg19_v512f_7);
    __m128i reg19_v128i_22 = _mm_add_epi64(reg19_v128i_21, reg19_v128i_20);
    __m256d reg19_v256d_23 = _mm256_max_pd(reg19_v256d_3, reg19_v256d_3);
    __m128d reg19_v128d_24 = _mm_mul_pd(reg19_v128d_11, reg19_v128d_12);
    __m128i reg19_v128i_25 = _mm_and_si128(reg19_v128i_20, reg19_v128i_22);
    __mmask16 reg19_m16_26 = _kxor_mask16(reg19_m16_9, reg19_m16_9);
    __m256i reg19_v256i_27 = _mm256_add_epi32(p6, reg19_v256i_2);
    g_v256f[1] = p5;
    g_v128i[2] = reg19_v128i_22;
    return p5;
}

CONV_MS __mmask64 abi_ms_20(__mmask16 p0, __mmask8 p1, __m256i p2, __m512 p3, __mmask16 p4, __m512 p5, __m256i p6, __m512i p7, __m512 p8, __m512i p9) {
    __m512 ms20_v512f_1 = _mm512_sub_ps(p8, p3);
    __m512i ms20_v512i_2 = _mm512_mask_blend_epi32(p4, p7, p9);
    __m256d ms20_v256d_3 = g_v256d[4];
    __m256d ms20_v256d_4 = _mm256_sqrt_pd(ms20_v256d_3);
    __mmask8 ms20_m8_5 = _kand_mask8(p1, p1);
    __m512i ms20_v512i_6 = _mm512_add_epi32(p7, ms20_v512i_2);
    __m256i ms20_v256i_7 = _mm256_mullo_epi32(p6, p2);
    __m512 ms20_v512f_8 = _mm512_cvtepi32_ps(ms20_v512i_6);
    __m128i ms20_v128i_9 = g_v128i[0];
    __m128i ms20_v128i_10 = _mm_slli_epi32(ms20_v128i_9, 3);
    __m256 ms20_v256f_11 = g_v256f[1];
    __m256 ms20_v256f_12 = _mm256_sub_ps(ms20_v256f_11, ms20_v256f_11);
    __m128 ms20_v128f_13 = g_v128f[2];
    __mmask64 ms20_m64_14 = g_m64[3];
    __mmask32 ms20_m32_15 = g_m32[5];
    __m128i ms20_v128i_16 = abi_reg_11(ms20_m8_5, ms20_v256f_11, ms20_v128f_13, ms20_m64_14, ms20_m32_15, ms20_v128i_10, ms20_m8_5);
    __m512 ms20_v512f_17 = _mm512_rcp14_ps(p5);
    __m512d ms20_v512d_18 = _mm512_set1_pd(2.0);
    __m512d ms20_v512d_19 = _mm512_add_pd(ms20_v512d_18, ms20_v512d_18);
    __m512 ms20_v512f_20 = _mm512_unpacklo_ps(p3, ms20_v512f_1);
    __m512 ms20_v512f_21 = _mm512_cvtepi32_ps(ms20_v512i_2);
    __m256 ms20_v256f_22 = _mm256_add_ps(ms20_v256f_11, ms20_v256f_11);
    __mmask16 ms20_m16_23 = _mm512_cmp_ps_mask(ms20_v512f_20, ms20_v512f_20, _CMP_LT_OQ);
    for (int ms20_i23 = 0; ms20_i23 < (int)((unsigned)p4 & 7); ms20_i23++) {
    __m512i ms20_v512i_25 = _mm512_unpacklo_epi32(p9, p9);
    __m512i ms20_v512i_26 = _mm512_broadcastmb_epi64(p1);
    }
    g_v512f[7] = ms20_v512f_17;
    g_v512i[6] = p9;
    g_m32[7] = ms20_m32_15;
    return ms20_m64_14;
}

CONV_REG __m128i abi_reg_21(__m128i p0, __mmask16 p1, __mmask32 p2, __m128i p3, __m128 p4, __mmask16 p5, __m128i p6, __mmask32 p7) {
    __m128i reg21_v128i_1 = _mm_xor_si128(p3, p0);
    __m256d reg21_v256d_2 = g_v256d[1];
    __m256d reg21_v256d_3 = _mm256_add_pd(reg21_v256d_2, reg21_v256d_2);
    __m128i reg21_v128i_4 = _mm_sub_epi64(p6, reg21_v128i_1);
    __m128 reg21_v128f_5 = _mm_fmadd_ps(p4, p4, p4);
    __m512i reg21_v512i_6 = g_v512i[0];
    __m512i reg21_v512i_7 = _mm512_mullo_epi32(reg21_v512i_6, reg21_v512i_6);
    __m512 reg21_v512f_8 = _mm512_set1_ps(1.0f);
    __m128i reg21_v128i_9 = abi_sysv_10(reg21_v512f_8, p1, reg21_v128i_4, p6, p5);
    __mmask64 reg21_m64_10 = (__mmask64)0x5A5A5A5A5A5A5A5AULL;
    __m256 reg21_v256f_11 = abi_sysv_4(p7, p1, reg21_v128f_5, reg21_m64_10, reg21_v512f_8);
    __mmask16 reg21_m16_12 = _mm512_cmp_epi32_mask(reg21_v512i_6, reg21_v512i_7, 2);
    __m128d reg21_v128d_13 = g_v128d[1];
    __m128d reg21_v128d_14 = _mm_fmadd_pd(reg21_v128d_13, reg21_v128d_13, reg21_v128d_13);
    __m512i reg21_v512i_15 = _mm512_mask_add_epi32(reg21_v512i_6, p5, reg21_v512i_7, reg21_v512i_6);
    __m128i reg21_v128i_16 = _mm_add_epi64(reg21_v128i_4, reg21_v128i_9);
    __m512i reg21_v512i_17 = _mm512_sub_epi32(reg21_v512i_15, reg21_v512i_7);
    __m128i reg21_v128i_18 = _mm_sub_epi32(reg21_v128i_16, reg21_v128i_16);
    if ((unsigned long long)reg21_m16_12 & 1ULL) {
    __m128 reg21_v128f_19 = _mm_sqrt_ps(p4);
    __m256d reg21_v256d_20 = _mm256_add_pd(reg21_v256d_2, reg21_v256d_3);
    }
    __m128 reg21_v128f_21 = abi_vec_14(p1, p2);
    __m512 reg21_v512f_22 = _mm512_fmadd_ps(reg21_v512f_8, reg21_v512f_8, reg21_v512f_8);
    __m512d reg21_v512d_23 = _mm512_set1_pd(2.0);
    __m512d reg21_v512d_24 = _mm512_max_pd(reg21_v512d_23, reg21_v512d_23);
    g_v512i[3] = reg21_v512i_7;
    g_v256d[4] = reg21_v256d_2;
    g_v128f[4] = reg21_v128f_21;
    g_m16[3] = p1;
    g_m32[6] = p2;
    g_m64[1] = reg21_m64_10;
    return reg21_v128i_9;
}

CONV_MS __m256i abi_ms_22(__mmask32 p0, __mmask32 p1, __m256i p2, __m512 p3, __mmask64 p4, __m512i p5, __mmask16 p6) {
    __mmask16 ms22_m16_1 = _kxor_mask16(p6, p6);
    __m512 ms22_v512f_2 = _mm512_max_ps(p3, p3);
    __m128 ms22_v128f_3 = _mm_set1_ps(1.0f);
    __m128 ms22_v128f_4 = _mm_fmadd_ps(ms22_v128f_3, ms22_v128f_3, ms22_v128f_3);
    __mmask8 ms22_m8_5 = (__mmask8)0xA5;
    __mmask8 ms22_m8_6 = _kand_mask8(ms22_m8_5, ms22_m8_5);
    __m256i ms22_v256i_7 = _mm256_xor_si256(p2, p2);
    __m512d ms22_v512d_8 = _mm512_set1_pd(2.0);
    __m512d ms22_v512d_9 = _mm512_sub_pd(ms22_v512d_8, ms22_v512d_8);
    __m128i ms22_v128i_10 = g_v128i[7];
    __m128i ms22_v128i_11 = _mm_slli_epi32(ms22_v128i_10, 3);
    __m256 ms22_v256f_12 = g_v256f[4];
    __m512d ms22_v512d_13 = abi_vec_8(p1, ms22_v128i_11, ms22_v512d_8, p5, ms22_v128i_10, p2, ms22_v256f_12, ms22_v128f_4, ms22_v512f_2);
    __m256d ms22_v256d_14 = g_v256d[0];
    __m256d ms22_v256d_15 = _mm256_sqrt_pd(ms22_v256d_14);
    __m512i ms22_v512i_16 = _mm512_broadcast_i32x4(ms22_v128i_11);
    __m512 ms22_v512f_17 = _mm512_fmadd_ps(p3, p3, ms22_v512f_2);
    __m128d ms22_v128d_18 = _mm_set1_pd(2.0);
    __m128d ms22_v128d_19 = _mm_max_pd(ms22_v128d_18, ms22_v128d_18);
    if ((unsigned long long)p6 & 1ULL) {
    __mmask16 ms22_m16_20 = _mm512_cmp_ps_mask(p3, ms22_v512f_17, _CMP_LT_OQ);
    __m512 ms22_v512f_21 = _mm512_broadcast_f32x4(ms22_v128f_3);
    __m128d ms22_v128d_22 = _mm_mul_pd(ms22_v128d_18, ms22_v128d_18);
    }
    __m128i ms22_v128i_23 = _mm_unpacklo_epi32(ms22_v128i_10, ms22_v128i_10);
    __m512i ms22_v512i_24 = _mm512_sub_epi64(p5, p5);
    __m512i ms22_v512i_25 = _mm512_sub_epi64(ms22_v512i_16, ms22_v512i_24);
    if ((unsigned long long)ms22_m16_1 & 1ULL) {
    __m512i ms22_v512i_26 = _mm512_ternarylogic_epi32(ms22_v512i_24, ms22_v512i_24, ms22_v512i_24, 0x96);
    __m512i ms22_v512i_27 = _mm512_ternarylogic_epi32(ms22_v512i_25, ms22_v512i_25, ms22_v512i_26, 0x96);
    }
    g_v512d[7] = ms22_v512d_9;
    g_v512i[3] = ms22_v512i_25;
    g_v256d[5] = ms22_v256d_15;
    g_v256i[6] = ms22_v256i_7;
    g_v128i[6] = ms22_v128i_10;
    g_m16[4] = p6;
    return ms22_v256i_7;
}

CONV_MS __mmask64 abi_ms_23(__m512d p0, __mmask64 p1, __mmask64 p2) {
    __m128i ms23_v128i_1 = _mm_set1_epi32(0x11);
    __m128i ms23_v128i_2 = _mm_unpacklo_epi32(ms23_v128i_1, ms23_v128i_1);
    __m256 ms23_v256f_3 = g_v256f[7];
    __m256 ms23_v256f_4 = _mm256_sqrt_ps(ms23_v256f_3);
    __m512i ms23_v512i_5 = g_v512i[6];
    __m512i ms23_v512i_6 = _mm512_slli_epi32(ms23_v512i_5, 3);
    __m256d ms23_v256d_7 = g_v256d[1];
    __m256d ms23_v256d_8 = _mm256_sqrt_pd(ms23_v256d_7);
    __m512 ms23_v512f_9 = _mm512_set1_ps(1.0f);
    __m512 ms23_v512f_10 = _mm512_fmadd_ps(ms23_v512f_9, ms23_v512f_9, ms23_v512f_9);
    __m256i ms23_v256i_11 = _mm256_set1_epi32(0x33);
    __mmask32 ms23_m32_12 = (__mmask32)0x5A5A5A5A;
    __mmask16 ms23_m16_13 = (__mmask16)0x5A5A;
    __mmask8 ms23_m8_14 = g_m8[7];
    __m128 ms23_v128f_15 = g_v128f[5];
    __m256 ms23_v256f_16 = abi_ms_6(ms23_v256i_11, ms23_v128i_1, ms23_m32_12, ms23_m16_13, p1, p1, ms23_m8_14, ms23_v128f_15, ms23_v128i_2, ms23_v256i_11);
    __m256d ms23_v256d_17 = _mm256_mul_pd(ms23_v256d_8, ms23_v256d_8);
    __m128 ms23_v128f_18 = _mm_min_ps(ms23_v128f_15, ms23_v128f_15);
    __m256i ms23_v256i_19 = _mm512_cvtepi64_epi32(ms23_v512i_5);
    __m256 ms23_v256f_20 = _mm256_add_ps(ms23_v256f_16, ms23_v256f_16);
    __m512i ms23_v512i_21 = _mm512_broadcast_i32x4(ms23_v128i_2);
    for (int ms23_i21 = 0; ms23_i21 < (int)((unsigned)ms23_m16_13 & 7); ms23_i21++) {
    __m512 ms23_v512f_23 = _mm512_unpacklo_ps(ms23_v512f_10, ms23_v512f_9);
    __m512i ms23_v512i_24 = _mm512_broadcast_i32x4(ms23_v128i_1);
    __m128d ms23_v128d_25 = g_v128d[1];
    __m128d ms23_v128d_26 = _mm_fmadd_pd(ms23_v128d_25, ms23_v128d_25, ms23_v128d_25);
    }
    __m128i ms23_v128i_27 = _mm_slli_epi32(ms23_v128i_2, 3);
    __m512i ms23_v512i_28 = _mm512_sub_epi32(ms23_v512i_21, ms23_v512i_5);
    __m128d ms23_v128d_29 = _mm_set1_pd(2.0);
    __m128d ms23_v128d_30 = _mm_sub_pd(ms23_v128d_29, ms23_v128d_29);
    __mmask32 ms23_m32_31 = _kxor_mask32(ms23_m32_12, ms23_m32_12);
    __mmask32 ms23_m32_32 = _mm512_kunpackw(ms23_m16_13, ms23_m16_13);
    __m128i ms23_v128i_33 = _mm_mullo_epi32(ms23_v128i_2, ms23_v128i_1);
    for (int ms23_i33 = 0; ms23_i33 < (int)((unsigned)ms23_m16_13 & 7); ms23_i33++) {
    __m512i ms23_v512i_35 = _mm512_broadcast_i32x4(ms23_v128i_2);
    __m256 ms23_v256f_36 = _mm256_fmadd_ps(ms23_v256f_16, ms23_v256f_3, ms23_v256f_4);
    __m128d ms23_v128d_37 = _mm_fmadd_pd(ms23_v128d_30, ms23_v128d_30, ms23_v128d_30);
    }
    g_v256f[4] = ms23_v256f_3;
    g_v256d[2] = ms23_v256d_7;
    g_v128f[5] = ms23_v128f_18;
    g_v128i[0] = ms23_v128i_2;
    g_m16[2] = ms23_m16_13;
    g_m32[0] = ms23_m32_12;
    g_m64[7] = p1;
    return p1;
}

CONV_SYSV __m256i abi_sysv_24(__m512i p0, __m512 p1, __m128i p2, __m512 p3, __mmask8 p4, __mmask16 p5) {
    __m128d sysv24_v128d_1 = g_v128d[4];
    __m128d sysv24_v128d_2 = _mm_add_pd(sysv24_v128d_1, sysv24_v128d_1);
    __m256 sysv24_v256f_3 = _mm256_set1_ps(1.0f);
    __m256 sysv24_v256f_4 = _mm256_sub_ps(sysv24_v256f_3, sysv24_v256f_3);
    __m128 sysv24_v128f_5 = g_v128f[6];
    __m128 sysv24_v128f_6 = _mm_sqrt_ps(sysv24_v128f_5);
    __m256d sysv24_v256d_7 = g_v256d[0];
    __m256d sysv24_v256d_8 = _mm256_max_pd(sysv24_v256d_7, sysv24_v256d_7);
    __m512d sysv24_v512d_9 = _mm512_set1_pd(2.0);
    __mmask64 sysv24_m64_10 = (__mmask64)0x5A5A5A5A5A5A5A5AULL;
    __m256i sysv24_v256i_11 = _mm256_set1_epi32(0x33);
    __m256 sysv24_v256f_12 = abi_reg_19(sysv24_v128f_5, sysv24_v512d_9, p1, sysv24_m64_10, sysv24_v256i_11, sysv24_v256f_3, sysv24_v256i_11, p5, sysv24_m64_10);
    __mmask16 sysv24_m16_13 = _kxor_mask16(p5, p5);
    __m512i sysv24_v512i_14 = _mm512_permutexvar_epi32(p0, p0);
    __m256 sysv24_v256f_15 = _mm256_unpacklo_ps(sysv24_v256f_4, sysv24_v256f_12);
    __m512i sysv24_v512i_16 = _mm512_ternarylogic_epi32(p0, sysv24_v512i_14, p0, 0x96);
    __m512i sysv24_v512i_17 = _mm512_xor_si512(sysv24_v512i_14, sysv24_v512i_16);
    __m256d sysv24_v256d_18 = _mm256_sub_pd(sysv24_v256d_8, sysv24_v256d_7);
    __m512i sysv24_v512i_19 = _mm512_xor_si512(sysv24_v512i_17, p0);
    __m256 sysv24_v256f_20 = _mm256_fmadd_ps(sysv24_v256f_4, sysv24_v256f_3, sysv24_v256f_15);
    __m512 sysv24_v512f_21 = _mm512_unpacklo_ps(p1, p1);
    for (int sysv24_i21 = 0; sysv24_i21 < (int)((unsigned)sysv24_m16_13 & 7); sysv24_i21++) {
    __m256d sysv24_v256d_23 = _mm256_max_pd(sysv24_v256d_7, sysv24_v256d_18);
    __m128i sysv24_v128i_24 = _mm_add_epi32(p2, p2);
    }
    g_v256f[3] = sysv24_v256f_15;
    g_v128i[0] = p2;
    g_m16[5] = sysv24_m16_13;
    g_m64[1] = sysv24_m64_10;
    return sysv24_v256i_11;
}

CONV_REG __m512i abi_reg_25(__m512 p0, __m256i p1, __mmask16 p2, __m256 p3, __m512d p4, __mmask16 p5, __mmask8 p6, __mmask32 p7, __m256i p8, __mmask8 p9) {
    __mmask16 reg25_m16_1 = _kxor_mask16(p2, p5);
    __m512i reg25_v512i_2 = _mm512_set1_epi32(0x55);
    __m512 reg25_v512f_3 = _mm512_cvtepi32_ps(reg25_v512i_2);
    __m128i reg25_v128i_4 = _mm_set1_epi32(0x11);
    __m512i reg25_v512i_5 = _mm512_broadcast_i32x4(reg25_v128i_4);
    __m512i reg25_v512i_6 = _mm512_mask_blend_epi32(p2, reg25_v512i_5, reg25_v512i_2);
    __mmask32 reg25_m32_7 = _mm512_kunpackw(p2, reg25_m16_1);
    __m128 reg25_v128f_8 = g_v128f[5];
    __m128 reg25_v128f_9 = _mm_fmadd_ps(reg25_v128f_8, reg25_v128f_8, reg25_v128f_8);
    __m512 reg25_v512f_10 = _mm512_min_ps(reg25_v512f_3, p0);
    __mmask64 reg25_m64_11 = (__mmask64)0x5A5A5A5A5A5A5A5AULL;
    __mmask16 reg25_m16_12 = abi_vec_17(p4, p9, reg25_v512i_5, p3, reg25_m64_11, p4);
    __m512d reg25_v512d_13 = abi_vec_8(reg25_m32_7, reg25_v128i_4, p4, reg25_v512i_2, reg25_v128i_4, p1, p3, reg25_v128f_9, p0);
    __m128 reg25_v128f_14 = _mm_fmadd_ps(reg25_v128f_9, reg25_v128f_9, reg25_v128f_9);
    __m512d reg25_v512d_15 = _mm512_sqrt_pd(reg25_v512d_13);
    __m128d reg25_v128d_16 = g_v128d[5];
    __m128d reg25_v128d_17 = _mm_sqrt_pd(reg25_v128d_16);
    __m512i reg25_v512i_18 = _mm512_sub_epi64(reg25_v512i_6, reg25_v512i_6);
    __m512 reg25_v512f_19 = _mm512_min_ps(reg25_v512f_10, reg25_v512f_10);
    if ((unsigned long long)p2 & 1ULL) {
    __mmask16 reg25_m16_20 = _kand_mask16(reg25_m16_12, reg25_m16_1);
    __m256d reg25_v256d_21 = _mm512_extractf64x4_pd(p4, 1);
    }
    __mmask32 reg25_m32_22 = _kxor_mask32(reg25_m32_7, p7);
    __m256d reg25_v256d_23 = _mm512_extractf64x4_pd(reg25_v512d_15, 1);
    __m128d reg25_v128d_24 = _mm_max_pd(reg25_v128d_16, reg25_v128d_17);
    __m128 reg25_v128f_25 = _mm_fmadd_ps(reg25_v128f_14, reg25_v128f_8, reg25_v128f_14);
    g_v512f[6] = p0;
    g_v512d[0] = reg25_v512d_13;
    g_m64[7] = reg25_m64_11;
    return reg25_v512i_18;
}

CONV_SYSV __m128 abi_sysv_26(__mmask32 p0, __m256i p1, __mmask64 p2) {
    __m512i sysv26_v512i_1 = _mm512_set1_epi32(0x55);
    __m512i sysv26_v512i_2 = _mm512_rol_epi32(sysv26_v512i_1, 5);
    __mmask16 sysv26_m16_3 = _mm512_cmp_epi32_mask(sysv26_v512i_1, sysv26_v512i_1, 2);
    __m512 sysv26_v512f_4 = _mm512_set1_ps(1.0f);
    __m512i sysv26_v512i_5 = _mm512_cvtps_epi32(sysv26_v512f_4);
    __m512 sysv26_v512f_6 = _mm512_cvtepi32_ps(sysv26_v512i_2);
    __m256d sysv26_v256d_7 = g_v256d[0];
    __m256d sysv26_v256d_8 = _mm256_mul_pd(sysv26_v256d_7, sysv26_v256d_7);
    __m512i sysv26_v512i_9 = _mm512_xor_si512(sysv26_v512i_5, sysv26_v512i_2);
    __m128 sysv26_v128f_10 = g_v128f[4];
    __m128 sysv26_v128f_11 = _mm_min_ps(sysv26_v128f_10, sysv26_v128f_10);
    __m512i sysv26_v512i_12 = _mm512_add_epi64(sysv26_v512i_1, sysv26_v512i_9);
    __m128 sysv26_v128f_13 = abi_vec_14(sysv26_m16_3, p0);
    __m128i sysv26_v128i_14 = _mm_set1_epi32(0x11);
    __m512d sysv26_v512d_15 = g_v512d[4];
    __m256 sysv26_v256f_16 = g_v256f[6];
    __m512d sysv26_v512d_17 = abi_vec_8(p0, sysv26_v128i_14, sysv26_v512d_15, sysv26_v512i_9, sysv26_v128i_14, p1, sysv26_v256f_16, sysv26_v128f_13, sysv26_v512f_4);
    __m128i sysv26_v128i_18 = _mm_unpacklo_epi32(sysv26_v128i_14, sysv26_v128i_14);
    __m256i sysv26_v256i_19 = _mm256_unpacklo_epi32(p1, p1);
    __m256i sysv26_v256i_20 = _mm256_mullo_epi32(p1, p1);
    __m512d sysv26_v512d_21 = _mm512_fmadd_pd(sysv26_v512d_17, sysv26_v512d_15, sysv26_v512d_15);
    __mmask16 sysv26_m16_22 = _mm512_cmp_epi32_mask(sysv26_v512i_1, sysv26_v512i_5, 2);
    __m512i sysv26_v512i_23 = _mm512_permutexvar_epi32(sysv26_v512i_5, sysv26_v512i_12);
    __m256i sysv26_v256i_24 = _mm512_cvtepi64_epi32(sysv26_v512i_5);
    for (int sysv26_i24 = 0; sysv26_i24 < (int)((unsigned)sysv26_m16_3 & 7); sysv26_i24++) {
    __m256 sysv26_v256f_26 = _mm256_fmadd_ps(sysv26_v256f_16, sysv26_v256f_16, sysv26_v256f_16);
    __m128 sysv26_v128f_27 = _mm_add_ps(sysv26_v128f_13, sysv26_v128f_11);
    __m512d sysv26_v512d_28 = _mm512_fmadd_pd(sysv26_v512d_17, sysv26_v512d_21, sysv26_v512d_15);
    __m256d sysv26_v256d_29 = _mm256_fmadd_pd(sysv26_v256d_7, sysv26_v256d_8, sysv26_v256d_8);
    }
    g_v512f[2] = sysv26_v512f_4;
    g_v512d[0] = sysv26_v512d_21;
    g_v256f[2] = sysv26_v256f_16;
    g_v128f[0] = sysv26_v128f_11;
    g_m16[3] = sysv26_m16_22;
    g_m64[4] = p2;
    return sysv26_v128f_10;
}

CONV_REG __m512 abi_reg_27(__mmask8 p0, __m256i p1, __mmask64 p2, __m512 p3, __m512i p4, __mmask16 p5, __m256 p6) {
    __m512i reg27_v512i_1 = _mm512_mask_blend_epi32(p5, p4, p4);
    __m512d reg27_v512d_2 = _mm512_set1_pd(2.0);
    __m512d reg27_v512d_3 = _mm512_max_pd(reg27_v512d_2, reg27_v512d_2);
    __m512d reg27_v512d_4 = _mm512_fmadd_pd(reg27_v512d_2, reg27_v512d_2, reg27_v512d_3);
    __m128 reg27_v128f_5 = g_v128f[7];
    __m128 reg27_v128f_6 = _mm_fmadd_ps(reg27_v128f_5, reg27_v128f_5, reg27_v128f_5);
    __m256 reg27_v256f_7 = _mm256_max_ps(p6, p6);
    __mmask16 reg27_m16_8 = _kxor_mask16(p5, p5);
    __mmask32 reg27_m32_9 = _mm512_kunpackw(reg27_m16_8, p5);
    __m128i reg27_v128i_10 = _mm_set1_epi32(0x11);
    __m128i reg27_v128i_11 = abi_sysv_13(reg27_v128f_6, p6, p1, p6, p3, reg27_v128i_10, reg27_v128i_10, p0);
    __m128 reg27_v128f_12 = abi_vec_14(p5, reg27_m32_9);
    __m128 reg27_v128f_13 = _mm_max_ps(reg27_v128f_6, reg27_v128f_12);
    __m256i reg27_v256i_14 = _mm256_and_si256(p1, p1);
    __m128 reg27_v128f_15 = _mm_fmadd_ps(reg27_v128f_12, reg27_v128f_12, reg27_v128f_5);
    __m512i reg27_v512i_16 = _mm512_mask_add_epi32(reg27_v512i_1, p5, reg27_v512i_1, reg27_v512i_1);
    __m512i reg27_v512i_17 = _mm512_mask_blend_epi32(reg27_m16_8, reg27_v512i_16, p4);
    __m512d reg27_v512d_18 = _mm512_sqrt_pd(reg27_v512d_4);
    g_v512f[3] = p3;
    g_v512i[0] = reg27_v512i_17;
    g_v256f[3] = p6;
    g_m16[4] = reg27_m16_8;
    return p3;
}

CONV_SYSV __m256 abi_sysv_28(__mmask16 p0, __mmask64 p1, __mmask32 p2, __m512d p3, __m256i p4, __m128i p5, __m256 p6, __mmask32 p7) {
    __mmask8 sysv28_m8_1 = _mm512_cmp_pd_mask(p3, p3, _CMP_GE_OQ);
    __m512i sysv28_v512i_2 = g_v512i[5];
    __m512i sysv28_v512i_3 = _mm512_sub_epi64(sysv28_v512i_2, sysv28_v512i_2);
    __m256d sysv28_v256d_4 = g_v256d[4];
    __m256d sysv28_v256d_5 = _mm256_fmadd_pd(sysv28_v256d_4, sysv28_v256d_4, sysv28_v256d_4);
    __m256d sysv28_v256d_6 = _mm256_mul_pd(sysv28_v256d_4, sysv28_v256d_5);
    __m256d sysv28_v256d_7 = _mm256_mul_pd(sysv28_v256d_5, sysv28_v256d_4);
    __m256d sysv28_v256d_8 = _mm256_mul_pd(sysv28_v256d_7, sysv28_v256d_7);
    __mmask16 sysv28_m16_9 = _mm512_cmp_epi32_mask(sysv28_v512i_3, sysv28_v512i_2, 2);
    __m128 sysv28_v128f_10 = g_v128f[3];
    __m128 sysv28_v128f_11 = _mm_min_ps(sysv28_v128f_10, sysv28_v128f_10);
    __m512 sysv28_v512f_12 = _mm512_set1_ps(1.0f);
    __m512 sysv28_v512f_13 = _mm512_sqrt_ps(sysv28_v512f_12);
    __m256i sysv28_v256i_14 = _mm256_sub_epi32(p4, p4);
    __m512 sysv28_v512f_15 = abi_reg_27(sysv28_m8_1, p4, p1, sysv28_v512f_13, sysv28_v512i_2, sysv28_m16_9, p6);
    __m128i sysv28_v128i_16 = _mm_xor_si128(p5, p5);
    __m256i sysv28_v256i_17 = _mm256_and_si256(p4, sysv28_v256i_14);
    __m512 sysv28_v512f_18 = _mm512_min_ps(sysv28_v512f_12, sysv28_v512f_15);
    for (int sysv28_i18 = 0; sysv28_i18 < (int)((unsigned)p0 & 7); sysv28_i18++) {
    __m256i sysv28_v256i_20 = _mm256_add_epi32(sysv28_v256i_17, p4);
    __m512 sysv28_v512f_21 = _mm512_fmadd_ps(sysv28_v512f_12, sysv28_v512f_18, sysv28_v512f_18);
    __m512i sysv28_v512i_22 = _mm512_ternarylogic_epi32(sysv28_v512i_2, sysv28_v512i_3, sysv28_v512i_3, 0x96);
    }
    __m256 sysv28_v256f_23 = _mm256_sqrt_ps(p6);
    __m128 sysv28_v128f_24 = _mm_unpacklo_ps(sysv28_v128f_11, sysv28_v128f_10);
    g_v512f[6] = sysv28_v512f_12;
    g_v256i[0] = p4;
    g_v128f[1] = sysv28_v128f_11;
    g_m16[0] = sysv28_m16_9;
    g_m32[4] = p7;
    g_m64[6] = p1;
    return p6;
}

CONV_REG __m128i abi_reg_29(__m512d p0, __m512 p1, __m512d p2, __m512i p3, __m512i p4) {
    __m128i reg29_v128i_1 = g_v128i[6];
    __m128i reg29_v128i_2 = _mm_mullo_epi32(reg29_v128i_1, reg29_v128i_1);
    __m512d reg29_v512d_3 = _mm512_fmadd_pd(p2, p0, p0);
    __m512d reg29_v512d_4 = _mm512_sub_pd(p0, reg29_v512d_3);
    __mmask64 reg29_m64_5 = _mm512_movepi8_mask(p4);
    __m128i reg29_v128i_6 = _mm_slli_epi32(reg29_v128i_1, 3);
    __m256 reg29_v256f_7 = _mm256_set1_ps(1.0f);
    __m256 reg29_v256f_8 = _mm256_fmadd_ps(reg29_v256f_7, reg29_v256f_7, reg29_v256f_7);
    __m128i reg29_v128i_9 = _mm_slli_epi32(reg29_v128i_2, 3);
    __mmask16 reg29_m16_10 = (__mmask16)0x5A5A;
    __m512 reg29_v512f_11 = _mm512_mask_blend_ps(reg29_m16_10, p1, p1);
    __mmask8 reg29_m8_12 = g_m8[3];
    __mmask16 reg29_m16_13 = abi_vec_17(reg29_v512d_4, reg29_m8_12, p3, reg29_v256f_8, reg29_m64_5, reg29_v512d_3);
    __m128i reg29_v128i_14 = abi_sysv_15(p2, reg29_v128i_6, reg29_m64_5, p1);
    __m128 reg29_v128f_15 = _mm_set1_ps(1.0f);
    __m128 reg29_v128f_16 = _mm_add_ps(reg29_v128f_15, reg29_v128f_15);
    __m128i reg29_v128i_17 = _mm_mullo_epi32(reg29_v128i_2, reg29_v128i_2);
    __m128i reg29_v128i_18 = _mm_unpacklo_epi32(reg29_v128i_17, reg29_v128i_1);
    __m512d reg29_v512d_19 = _mm512_sqrt_pd(p2);
    if ((unsigned long long)reg29_m16_13 & 1ULL) {
    __m512i reg29_v512i_20 = _mm512_cvtps_epi32(p1);
    __m128 reg29_v128f_21 = _mm_min_ps(reg29_v128f_16, reg29_v128f_16);
    }
    __mmask8 reg29_m8_22 = abi_sysv_0(reg29_m64_5, p2, reg29_v128f_16);
    __m256i reg29_v256i_23 = _mm256_set1_epi32(0x33);
    __m256i reg29_v256i_24 = _mm256_add_epi64(reg29_v256i_23, reg29_v256i_23);
    __m128 reg29_v128f_25 = _mm_fmadd_ps(reg29_v128f_16, reg29_v128f_15, reg29_v128f_16);
    __m256d reg29_v256d_26 = _mm256_set1_pd(2.0);
    __m256d reg29_v256d_27 = _mm256_fmadd_pd(reg29_v256d_26, reg29_v256d_26, reg29_v256d_26);
    __m256i reg29_v256i_28 = _mm512_cvtepi64_epi32(p3);
    __m512 reg29_v512f_29 = _mm512_mask_add_ps(reg29_v512f_11, reg29_m16_10, reg29_v512f_11, p1);
    for (int reg29_i29 = 0; reg29_i29 < (int)((unsigned)reg29_m16_10 & 7); reg29_i29++) {
    __m128d reg29_v128d_31 = _mm_set1_pd(2.0);
    __m128d reg29_v128d_32 = _mm_fmadd_pd(reg29_v128d_31, reg29_v128d_31, reg29_v128d_31);
    __m256i reg29_v256i_33 = _mm256_xor_si256(reg29_v256i_23, reg29_v256i_28);
    }
    g_v128f[1] = reg29_v128f_16;
    g_m64[6] = reg29_m64_5;
    return reg29_v128i_1;
}

CONV_SYSV __m128i abi_sysv_30(__mmask64 p0, __m256 p1, __mmask8 p2, __m128i p3, __mmask64 p4) {
    __m512 sysv30_v512f_1 = g_v512f[3];
    __m512i sysv30_v512i_2 = _mm512_cvtps_epi32(sysv30_v512f_1);
    __mmask16 sysv30_m16_3 = g_m16[2];
    __mmask16 sysv30_m16_4 = _kand_mask16(sysv30_m16_3, sysv30_m16_3);
    __m256 sysv30_v256f_5 = _mm256_fmadd_ps(p1, p1, p1);
    __m128i sysv30_v128i_6 = _mm_mullo_epi32(p3, p3);
    __m128 sysv30_v128f_7 = _mm_set1_ps(1.0f);
    __m128 sysv30_v128f_8 = _mm_mul_ps(sysv30_v128f_7, sysv30_v128f_7);
    __m512 sysv30_v512f_9 = _mm512_scalef_ps(sysv30_v512f_1, sysv30_v512f_1);
    __m128i sysv30_v128i_10 = _mm_mullo_epi32(p3, p3);
    __mmask32 sysv30_m32_11 = (__mmask32)0x5A5A5A5A;
    __m128i sysv30_v128i_12 = abi_reg_5(sysv30_v512i_2, sysv30_m32_11, p0);
    __m256 sysv30_v256f_13 = _mm256_max_ps(p1, sysv30_v256f_5);
    __mmask16 sysv30_m16_14 = _mm512_cmp_epi32_mask(sysv30_v512i_2, sysv30_v512i_2, 2);
    __m512d sysv30_v512d_15 = _mm512_set1_pd(2.0);
    __m512d sysv30_v512d_16 = _mm512_add_pd(sysv30_v512d_15, sysv30_v512d_15);
    __m256d sysv30_v256d_17 = _mm256_set1_pd(2.0);
    __m256d sysv30_v256d_18 = _mm256_max_pd(sysv30_v256d_17, sysv30_v256d_17);
    for (int sysv30_i18 = 0; sysv30_i18 < (int)((unsigned)sysv30_m16_3 & 7); sysv30_i18++) {
    __m512i sysv30_v512i_20 = _mm512_cvtps_epi32(sysv30_v512f_9);
    __m256 sysv30_v256f_21 = _mm256_max_ps(sysv30_v256f_5, sysv30_v256f_13);
    }
    __m512i sysv30_v512i_22 = _mm512_cvtps_epi32(sysv30_v512f_1);
    __m512i sysv30_v512i_23 = _mm512_mask_add_epi32(sysv30_v512i_22, sysv30_m16_3, sysv30_v512i_2, sysv30_v512i_2);
    __mmask64 sysv30_m64_24 = _mm512_movepi8_mask(sysv30_v512i_2);
    for (int sysv30_i24 = 0; sysv30_i24 < (int)((unsigned)sysv30_m16_3 & 7); sysv30_i24++) {
    __m512i sysv30_v512i_26 = _mm512_mullo_epi32(sysv30_v512i_23, sysv30_v512i_22);
    __m512d sysv30_v512d_27 = _mm512_add_pd(sysv30_v512d_16, sysv30_v512d_15);
    __m256i sysv30_v256i_28 = _mm256_set1_epi32(0x33);
    __m256i sysv30_v256i_29 = _mm256_unpacklo_epi32(sysv30_v256i_28, sysv30_v256i_28);
    __mmask16 sysv30_m16_30 = _mm512_cmp_epi32_mask(sysv30_v512i_26, sysv30_v512i_26, 2);
    }
    g_v256f[6] = sysv30_v256f_5;
    g_v128f[0] = sysv30_v128f_7;
    g_m8[7] = p2;
    g_m16[2] = sysv30_m16_14;
    g_m32[7] = sysv30_m32_11;
    g_m64[3] = sysv30_m64_24;
    return sysv30_v128i_6;
}

CONV_VEC __m256 abi_vec_31(__m512d p0, __mmask64 p1, __mmask16 p2, __mmask8 p3, __m128 p4, __m512i p5, __mmask8 p6) {
    __m256 vec31_v256f_1 = g_v256f[1];
    __m256 vec31_v256f_2 = _mm256_fmadd_ps(vec31_v256f_1, vec31_v256f_1, vec31_v256f_1);
    __m128i vec31_v128i_3 = g_v128i[4];
    __m128i vec31_v128i_4 = _mm_and_si128(vec31_v128i_3, vec31_v128i_3);
    __mmask16 vec31_m16_5 = _knot_mask16(p2);
    __m128 vec31_v128f_6 = _mm_mul_ps(p4, p4);
    __m256i vec31_v256i_7 = _mm256_set1_epi32(0x33);
    __m256i vec31_v256i_8 = _mm256_sub_epi32(vec31_v256i_7, vec31_v256i_7);
    __m128d vec31_v128d_9 = g_v128d[7];
    __m128d vec31_v128d_10 = _mm_add_pd(vec31_v128d_9, vec31_v128d_9);
    __m128 vec31_v128f_11 = _mm_unpacklo_ps(p4, p4);
    __m512i vec31_v512i_12 = _mm512_slli_epi32(p5, 3);
    __mmask32 vec31_m32_13 = (__mmask32)0x5A5A5A5A;
    __m128i vec31_v128i_14 = abi_reg_5(p5, vec31_m32_13, p1);
    __m512 vec31_v512f_15 = _mm512_set1_ps(1.0f);
    __m512 vec31_v512f_16 = _mm512_min_ps(vec31_v512f_15, vec31_v512f_15);
    __m128 vec31_v128f_17 = _mm_min_ps(p4, vec31_v128f_11);
    __m512 vec31_v512f_18 = _mm512_sub_ps(vec31_v512f_15, vec31_v512f_15);
    __mmask8 vec31_m8_19 = _mm512_cmp_pd_mask(p0, p0, _CMP_GE_OQ);
    __m512i vec31_v512i_20 = _mm512_sub_epi64(vec31_v512i_12, p5);
    if ((unsigned long long)p2 & 1ULL) {
    __mmask16 vec31_m16_21 = _kand_mask16(p2, p2);
    __m512d vec31_v512d_22 = _mm512_sub_pd(p0, p0);
    __m128 vec31_v128f_23 = _mm_add_ps(vec31_v128f_6, p4);
    __m128i vec31_v128i_24 = _mm_slli_epi32(vec31_v128i_4, 3);
    __m256 vec31_v256f_25 = _mm256_fmadd_ps(vec31_v256f_2, vec31_v256f_1, vec31_v256f_1);
    }
    __mmask32 vec31_m32_26 = _mm512_cmp_epi16_mask(vec31_v512i_12, vec31_v512i_20, 2);
    __mmask16 vec31_m16_27 = _kor_mask16(vec31_m16_5, vec31_m16_5);
    __m512i vec31_v512i_28 = _mm512_mask_add_epi32(vec31_v512i_20, vec31_m16_27, vec31_v512i_20, p5);
    __m512 vec31_v512f_29 = _mm512_mask_blend_ps(p2, vec31_v512f_16, vec31_v512f_15);
    __m512i vec31_v512i_30 = _mm512_mask_blend_epi32(vec31_m16_27, p5, p5);
    __m256i vec31_v256i_31 = _mm512_cvtepi64_epi32(vec31_v512i_20);
    for (int vec31_i31 = 0; vec31_i31 < (int)((unsigned)vec31_m16_27 & 7); vec31_i31++) {
    __m512i vec31_v512i_33 = _mm512_ternarylogic_epi32(p5, vec31_v512i_28, vec31_v512i_12, 0x96);
    __m512i vec31_v512i_34 = _mm512_broadcastmw_epi32(vec31_m16_27);
    __m512 vec31_v512f_35 = _mm512_fmadd_ps(vec31_v512f_16, vec31_v512f_18, vec31_v512f_18);
    }
    g_v256i[3] = vec31_v256i_8;
    g_v128f[0] = vec31_v128f_11;
    g_v128i[2] = vec31_v128i_4;
    g_m16[5] = vec31_m16_27;
    g_m32[0] = vec31_m32_26;
    return vec31_v256f_1;
}

CONV_MS __m256 abi_ms_32(__m256i p0, __m512d p1) {
    __m128d ms32_v128d_1 = g_v128d[0];
    __m128d ms32_v128d_2 = _mm_sqrt_pd(ms32_v128d_1);
    __m256d ms32_v256d_3 = _mm256_set1_pd(2.0);
    __m256d ms32_v256d_4 = _mm256_max_pd(ms32_v256d_3, ms32_v256d_3);
    __m512d ms32_v512d_5 = _mm512_add_pd(p1, p1);
    __m512i ms32_v512i_6 = _mm512_set1_epi32(0x55);
    __m512i ms32_v512i_7 = _mm512_sub_epi32(ms32_v512i_6, ms32_v512i_6);
    __m256i ms32_v256i_8 = _mm256_slli_epi32(p0, 3);
    __mmask8 ms32_m8_9 = (__mmask8)0xA5;
    __m512i ms32_v512i_10 = _mm512_broadcastmb_epi64(ms32_m8_9);
    __m128i ms32_v128i_11 = g_v128i[1];
    __m128i ms32_v128i_12 = _mm_sub_epi64(ms32_v128i_11, ms32_v128i_11);
    __mmask16 ms32_m16_13 = g_m16[3];
    __mmask16 ms32_m16_14 = _kand_mask16(ms32_m16_13, ms32_m16_13);
    __mmask32 ms32_m32_15 = g_m32[2];
    __mmask64 ms32_m64_16 = (__mmask64)0x5A5A5A5A5A5A5A5AULL;
    __m512i ms32_v512i_17 = abi_sysv_18(ms32_v128i_12, ms32_v512i_6, ms32_m32_15, ms32_m64_16, ms32_m32_15, ms32_m64_16);
    __m512d ms32_v512d_18 = _mm512_sub_pd(p1, p1);
    __m512i ms32_v512i_19 = _mm512_sub_epi32(ms32_v512i_17, ms32_v512i_7);
    __m512 ms32_v512f_20 = _mm512_cvtepi32_ps(ms32_v512i_10);
    __m256 ms32_v256f_21 = g_v256f[1];
    __m256 ms32_v256f_22 = _mm256_sub_ps(ms32_v256f_21, ms32_v256f_21);
    for (int ms32_i22 = 0; ms32_i22 < (int)((unsigned)ms32_m16_14 & 7); ms32_i22++) {
    __m512 ms32_v512f_24 = _mm512_add_ps(ms32_v512f_20, ms32_v512f_20);
    __m512d ms32_v512d_25 = _mm512_add_pd(ms32_v512d_5, ms32_v512d_18);
    __m128d ms32_v128d_26 = _mm_sub_pd(ms32_v128d_2, ms32_v128d_2);
    }
    __m128i ms32_v128i_27 = abi_sysv_30(ms32_m64_16, ms32_v256f_22, ms32_m8_9, ms32_v128i_12, ms32_m64_16);
    __m256d ms32_v256d_28 = _mm256_max_pd(ms32_v256d_3, ms32_v256d_4);
    __m256 ms32_v256f_29 = _mm256_sqrt_ps(ms32_v256f_21);
    __m128i ms32_v128i_30 = _mm_sub_epi64(ms32_v128i_11, ms32_v128i_11);
    if ((unsigned long long)ms32_m16_14 & 1ULL) {
    __m512i ms32_v512i_31 = _mm512_sub_epi32(ms32_v512i_17, ms32_v512i_19);
    __m128 ms32_v128f_32 = g_v128f[4];
    __m128 ms32_v128f_33 = _mm_min_ps(ms32_v128f_32, ms32_v128f_32);
    __mmask8 ms32_m8_34 = _kand_mask8(ms32_m8_9, ms32_m8_9);
    }
    g_v512i[3] = ms32_v512i_6;
    g_v256f[7] = ms32_v256f_21;
    g_v256d[5] = ms32_v256d_28;
    g_v256i[1] = p0;
    g_m16[4] = ms32_m16_13;
    g_m32[5] = ms32_m32_15;
    return ms32_v256f_29;
}

CONV_REG __m512i abi_reg_33(__m256 p0, __m512d p1, __mmask8 p2, __mmask8 p3) {
    __m512i reg33_v512i_1 = g_v512i[6];
    __mmask64 reg33_m64_2 = _mm512_cmp_epi8_mask(reg33_v512i_1, reg33_v512i_1, 1);
    __m128i reg33_v128i_3 = _mm_set1_epi32(0x11);
    __m128i reg33_v128i_4 = _mm_slli_epi32(reg33_v128i_3, 3);
    __m256i reg33_v256i_5 = _mm256_set1_epi32(0x33);
    __m256i reg33_v256i_6 = _mm256_slli_epi32(reg33_v256i_5, 3);
    __m128d reg33_v128d_7 = g_v128d[3];
    __m128d reg33_v128d_8 = _mm_max_pd(reg33_v128d_7, reg33_v128d_7);
    __m128 reg33_v128f_9 = _mm_set1_ps(1.0f);
    __m128 reg33_v128f_10 = _mm_sqrt_ps(reg33_v128f_9);
    __mmask16 reg33_m16_11 = _mm512_movepi32_mask(reg33_v512i_1);
    __m256 reg33_v256f_12 = _mm256_sub_ps(p0, p0);
    __m512i reg33_v512i_13 = _mm512_xor_si512(reg33_v512i_1, reg33_v512i_1);
    __mmask32 reg33_m32_14 = (__mmask32)0x5A5A5A5A;
    __m128 reg33_v128f_15 = abi_ms_7(reg33_v256i_6, reg33_v256i_5, reg33_m32_14, reg33_v512i_13, p0);
    __m512 reg33_v512f_16 = g_v512f[5];
    __m512 reg33_v512f_17 = _mm512_sqrt_ps(reg33_v512f_16);
    __m128i reg33_v128i_18 = _mm_mullo_epi32(reg33_v128i_4, reg33_v128i_4);
    __m512d reg33_v512d_19 = _mm512_max_pd(p1, p1);
    __mmask64 reg33_m64_20 = _mm512_movepi8_mask(reg33_v512i_1);
    __m128d reg33_v128d_21 = _mm_add_pd(reg33_v128d_8, reg33_v128d_7);
    for (int reg33_i21 = 0; reg33_i21 < (int)((unsigned)reg33_m16_11 & 7); reg33_i21++) {
    __m512 reg33_v512f_23 = _mm512_unpacklo_ps(reg33_v512f_16, reg33_v512f_16);
    __m512 reg33_v512f_24 = _mm512_mul_ps(reg33_v512f_17, reg33_v512f_16);
    }
    __m128i reg33_v128i_25 = _mm_slli_epi32(reg33_v128i_18, 3);
    __m128 reg33_v128f_26 = _mm_sqrt_ps(reg33_v128f_15);
    __m128 reg33_v128f_27 = _mm_unpacklo_ps(reg33_v128f_10, reg33_v128f_10);
    __m512 reg33_v512f_28 = _mm512_add_ps(reg33_v512f_16, reg33_v512f_16);
    __mmask64 reg33_m64_29 = _mm512_cmp_epi8_mask(reg33_v512i_1, reg33_v512i_1, 1);
    __m512i reg33_v512i_30 = _mm512_mask_add_epi32(reg33_v512i_1, reg33_m16_11, reg33_v512i_13, reg33_v512i_1);
    if ((unsigned long long)reg33_m16_11 & 1ULL) {
    __m256i reg33_v256i_31 = _mm256_unpacklo_epi32(reg33_v256i_5, reg33_v256i_5);
    __mmask16 reg33_m16_32 = _mm512_movepi32_mask(reg33_v512i_13);
    __m512 reg33_v512f_33 = _mm512_max_ps(reg33_v512f_28, reg33_v512f_28);
    __m256i reg33_v256i_34 = _mm256_sub_epi64(reg33_v256i_6, reg33_v256i_6);
    __mmask16 reg33_m16_35 = _mm512_movepi32_mask(reg33_v512i_30);
    }
    g_v512f[6] = reg33_v512f_17;
    g_v512d[1] = reg33_v512d_19;
    g_v256i[1] = reg33_v256i_6;
    g_v128f[1] = reg33_v128f_9;
    g_m64[0] = reg33_m64_20;
    return reg33_v512i_1;
}

CONV_REG __m512d abi_reg_34(__mmask16 p0, __mmask8 p1, __m512d p2, __mmask64 p3, __m256 p4, __mmask64 p5, __mmask8 p6, __m128i p7, __m512d p8, __m128 p9) {
    __mmask8 reg34_m8_1 = _kand_mask8(p6, p1);
    __mmask8 reg34_m8_2 = _kand_mask8(p6, p6);
    __m256 reg34_v256f_3 = _mm256_add_ps(p4, p4);
    __mmask32 reg34_m32_4 = g_m32[3];
    __mmask32 reg34_m32_5 = _kxor_mask32(reg34_m32_4, reg34_m32_4);
    __m512 reg34_v512f_6 = g_v512f[7];
    __m512 reg34_v512f_7 = _mm512_sub_ps(reg34_v512f_6, reg34_v512f_6);
    __m256i reg34_v256i_8 = _mm256_set1_epi32(0x33);
    __m256i reg34_v256i_9 = _mm256_unpacklo_epi32(reg34_v256i_8, reg34_v256i_8);
    __m512i reg34_v512i_10 = g_v512i[2];
    __mmask64 reg34_m64_11 = abi_ms_20(p0, p6, reg34_v256i_9, reg34_v512f_6, p0, reg34_v512f_7, reg34_v256i_9, reg34_v512i_10, reg34_v512f_6, reg34_v512i_10);
    __m512 reg34_v512f_12 = _mm512_mask_blend_ps(p0, reg34_v512f_7, reg34_v512f_7);
    __m128i reg34_v128i_13 = _mm_add_epi32(p7, p7);
    __mmask16 reg34_m16_14 = _kxor_mask16(p0, p0);
    __m128d reg34_v128d_15 = _mm_set1_pd(2.0);
    __m128d reg34_v128d_16 = _mm_sub_pd(reg34_v128d_15, reg34_v128d_15);
    __m512i reg34_v512i_17 = _mm512_mask_blend_epi32(reg34_m16_14, reg34_v512i_10, reg34_v512i_10);
    __m512 reg34_v512f_18 = _mm512_rcp14_ps(reg34_v512f_7);
    __m512i reg34_v512i_19 = _mm512_broadcastmw_epi32(reg34_m16_14);
    g_v512f[7] = reg34_v512f_18;
    g_v512d[5] = p2;
    g_v128f[2] = p9;
    g_m16[1] = reg34_m16_14;
    g_m32[6] = reg34_m32_4;
    g_m64[6] = p3;
    return p8;
}

CONV_VEC __m512d abi_vec_35(__mmask64 p0, __m512 p1) {
    __m256d vec35_v256d_1 = _mm256_set1_pd(2.0);
    __m256d vec35_v256d_2 = _mm256_sqrt_pd(vec35_v256d_1);
    __m512i vec35_v512i_3 = _mm512_set1_epi32(0x55);
    __m512i vec35_v512i_4 = _mm512_add_epi64(vec35_v512i_3, vec35_v512i_3);
    __mmask64 vec35_m64_5 = _kor_mask64(p0, p0);
    __m512i vec35_v512i_6 = _mm512_permutexvar_epi32(vec35_v512i_3, vec35_v512i_3);
    __mmask8 vec35_m8_7 = g_m8[7];
    __m256i vec35_v256i_8 = _mm256_set1_epi32(0x33);
    __mmask16 vec35_m16_9 = (__mmask16)0x5A5A;
    __m256 vec35_v256f_10 = g_v256f[4];
    __m512 vec35_v512f_11 = abi_reg_27(vec35_m8_7, vec35_v256i_8, p0, p1, vec35_v512i_4, vec35_m16_9, vec35_v256f_10);
    __m256i vec35_v256i_12 = _mm256_slli_epi32(vec35_v256i_8, 3);
    __m512i vec35_v512i_13 = _mm512_rol_epi32(vec35_v512i_6, 5);
    if ((unsigned long long)vec35_m16_9 & 1ULL) {
    __m128i vec35_v128i_14 = g_v128i[1];
    __m128i vec35_v128i_15 = _mm_and_si128(vec35_v128i_14, vec35_v128i_14);
    __m512i vec35_v512i_16 = _mm512_sub_epi32(vec35_v512i_4, vec35_v512i_13);
    __m256i vec35_v256i_17 = _mm256_mullo_epi32(vec35_v256i_12, vec35_v256i_12);
    __m512i vec35_v512i_18 = _mm512_slli_epi32(vec35_v512i_6, 3);
    }
    __m512 vec35_v512f_19 = _mm512_mul_ps(p1, vec35_v512f_11);
    __m512i vec35_v512i_20 = _mm512_ternarylogic_epi32(vec35_v512i_13, vec35_v512i_3, vec35_v512i_13, 0x96);
    __m512d vec35_v512d_21 = g_v512d[0];
    __m512d vec35_v512d_22 = _mm512_sqrt_pd(vec35_v512d_21);
    __m512 vec35_v512f_23 = _mm512_maskz_mul_ps(vec35_m16_9, p1, vec35_v512f_11);
    g_v512i[0] = vec35_v512i_20;
    g_m64[5] = vec35_m64_5;
    return vec35_v512d_21;
}

CONV_MS __m512d abi_ms_36(__m512d p0, __m512 p1, __mmask16 p2, __mmask16 p3, __mmask8 p4, __m256i p5, __m512i p6, __m512d p7) {
    __m128d ms36_v128d_1 = g_v128d[5];
    __m128d ms36_v128d_2 = _mm_fmadd_pd(ms36_v128d_1, ms36_v128d_1, ms36_v128d_1);
    __m512i ms36_v512i_3 = _mm512_cvtps_epi32(p1);
    __m512i ms36_v512i_4 = _mm512_xor_si512(p6, ms36_v512i_3);
    __m512i ms36_v512i_5 = _mm512_mask_blend_epi32(p2, p6, ms36_v512i_4);
    __m256 ms36_v256f_6 = _mm256_set1_ps(1.0f);
    __mmask32 ms36_m32_7 = (__mmask32)0x5A5A5A5A;
    __m512i ms36_v512i_8 = abi_reg_25(p1, p5, p3, ms36_v256f_6, p7, p3, p4, ms36_m32_7, p5, p4);
    __m512 ms36_v512f_9 = _mm512_add_ps(p1, p1);
    __m512i ms36_v512i_10 = _mm512_unpacklo_epi32(ms36_v512i_3, ms36_v512i_3);
    __m512d ms36_v512d_11 = _mm512_mask_blend_pd(p4, p0, p7);
    for (int ms36_i11 = 0; ms36_i11 < (int)((unsigned)p2 & 7); ms36_i11++) {
    __m512i ms36_v512i_13 = _mm512_permutexvar_epi32(ms36_v512i_5, ms36_v512i_5);
    __m512 ms36_v512f_14 = _mm512_cvtepi32_ps(ms36_v512i_10);
    __mmask32 ms36_m32_15 = _mm512_kunpackw(p3, p3);
    __m128i ms36_v128i_16 = g_v128i[2];
    __m128i ms36_v128i_17 = _mm_add_epi32(ms36_v128i_16, ms36_v128i_16);
    }
    __m512 ms36_v512f_18 = _mm512_cvtepi32_ps(ms36_v512i_3);
    __m128d ms36_v128d_19 = _mm_mul_pd(ms36_v128d_1, ms36_v128d_1);
    __m128 ms36_v128f_20 = g_v128f[0];
    __m128 ms36_v128f_21 = _mm_min_ps(ms36_v128f_20, ms36_v128f_20);
    __mmask16 ms36_m16_22 = _kxor_mask16(p2, p2);
    for (int ms36_i22 = 0; ms36_i22 < (int)((unsigned)ms36_m16_22 & 7); ms36_i22++) {
    __m256i ms36_v256i_24 = _mm256_and_si256(p5, p5);
    __mmask16 ms36_m16_25 = _mm512_cmp_epi32_mask(ms36_v512i_8, ms36_v512i_5, 2);
    __mmask32 ms36_m32_26 = _mm512_cmp_epi16_mask(p6, p6, 2);
    }
    g_v512d[1] = ms36_v512d_11;
    g_v256f[6] = ms36_v256f_6;
    g_v128f[2] = ms36_v128f_21;
    g_v128d[1] = ms36_v128d_2;
    return ms36_v512d_11;
}

CONV_REG __m512i abi_reg_37(__m512i p0, __m512d p1, __m256i p2, __m256i p3, __m128i p4) {
    __m512i reg37_v512i_1 = _mm512_mullo_epi32(p0, p0);
    __m256 reg37_v256f_2 = g_v256f[2];
    __m256 reg37_v256f_3 = _mm256_mul_ps(reg37_v256f_2, reg37_v256f_2);
    __m128i reg37_v128i_4 = _mm_and_si128(p4, p4);
    __mmask16 reg37_m16_5 = (__mmask16)0x5A5A;
    __mmask16 reg37_m16_6 = _kor_mask16(reg37_m16_5, reg37_m16_5);
    __mmask32 reg37_m32_7 = g_m32[2];
    __m128 reg37_v128f_8 = abi_ms_7(p3, p2, reg37_m32_7, p0, reg37_v256f_3);
    __m256d reg37_v256d_9 = g_v256d[4];
    __m256d reg37_v256d_10 = _mm256_sub_pd(reg37_v256d_9, reg37_v256d_9);
    __m128i reg37_v128i_11 = _mm_sub_epi64(p4, p4);
    __m512i reg37_v512i_12 = _mm512_sub_epi32(reg37_v512i_1, p0);
    if ((unsigned long long)reg37_m16_5 & 1ULL) {
    __m256d reg37_v256d_13 = _mm256_add_pd(reg37_v256d_10, reg37_v256d_9);
    __m512 reg37_v512f_14 = _mm512_set1_ps(1.0f);
    __m512 reg37_v512f_15 = _mm512_add_ps(reg37_v512f_14, reg37_v512f_14);
    __m512 reg37_v512f_16 = _mm512_scalef_ps(reg37_v512f_15, reg37_v512f_14);
    }
    __m512d reg37_v512d_17 = _mm512_mul_pd(p1, p1);
    __m128 reg37_v128f_18 = _mm_mul_ps(reg37_v128f_8, reg37_v128f_8);
    g_v512d[1] = p1;
    g_v256d[6] = reg37_v256d_9;
    g_m16[7] = reg37_m16_6;
    return reg37_v512i_12;
}

CONV_VEC __m128i abi_vec_38(__m128 p0, __m256 p1, __m512d p2, __mmask16 p3, __mmask32 p4, __m256 p5) {
    __m256 vec38_v256f_1 = _mm256_fmadd_ps(p1, p5, p1);
    __mmask16 vec38_m16_2 = _knot_mask16(p3);
    __m128i vec38_v128i_3 = g_v128i[3];
    __m128i vec38_v128i_4 = _mm_sub_epi32(vec38_v128i_3, vec38_v128i_3);
    __m128 vec38_v128f_5 = _mm_add_ps(p0, p0);
    __m256 vec38_v256f_6 = _mm256_max_ps(p5, p5);
    __mmask64 vec38_m64_7 = g_m64[7];
    __mmask64 vec38_m64_8 = _kor_mask64(vec38_m64_7, vec38_m64_7);
    __m512i vec38_v512i_9 = _mm512_set1_epi32(0x55);
    __mmask16 vec38_m16_10 = _mm512_cmp_epi32_mask(vec38_v512i_9, vec38_v512i_9, 2);
    __m128 vec38_v128f_11 = _mm_sub_ps(p0, p0);
    __mmask64 vec38_m64_12 = _mm512_movepi8_mask(vec38_v512i_9);
    __m256i vec38_v256i_13 = _mm256_set1_epi32(0x33);
    __mmask8 vec38_m8_14 = (__mmask8)0xA5;
    __m256 vec38_v256f_15 = abi_ms_6(vec38_v256i_13, vec38_v128i_4, p4, vec38_m16_2, vec38_m64_7, vec38_m64_7, vec38_m8_14, vec38_v128f_11, vec38_v128i_3, vec38_v256i_13);
    __m256 vec38_v256f_16 = _mm256_sub_ps(vec38_v256f_1, vec38_v256f_1);
    __mmask16 vec38_m16_17 = _mm512_cmp_epi32_mask(vec38_v512i_9, vec38_v512i_9, 2);
    __m512 vec38_v512f_18 = _mm512_set1_ps(1.0f);
    __m512 vec38_v512f_19 = _mm512_min_ps(vec38_v512f_18, vec38_v512f_18);
    __m512 vec38_v512f_20 = _mm512_broadcast_f32x4(vec38_v128f_5);
    __m256i vec38_v256i_21 = _mm256_sub_epi32(vec38_v256i_13, vec38_v256i_13);
    __mmask16 vec38_m16_22 = _kand_mask16(vec38_m16_17, vec38_m16_17);
    __m128d vec38_v128d_23 = _mm_set1_pd(2.0);
    __m128d vec38_v128d_24 = _mm_fmadd_pd(vec38_v128d_23, vec38_v128d_23, vec38_v128d_23);
    __m512 vec38_v512f_25 = _mm512_broadcast_f32x4(vec38_v128f_5);
    if ((unsigned long long)vec38_m16_2 & 1ULL) {
    __m512i vec38_v512i_26 = _mm512_sub_epi64(vec38_v512i_9, vec38_v512i_9);
    __m128 vec38_v128f_27 = _mm512_extractf32x4_ps(vec38_v512f_20, 2);
    __m512d vec38_v512d_28 = _mm512_mul_pd(p2, p2);
    __m512i vec38_v512i_29 = _mm512_mask_add_epi32(vec38_v512i_26, vec38_m16_10, vec38_v512i_26, vec38_v512i_26);
    __m128 vec38_v128f_30 = _mm_sub_ps(p0, p0);
    }
    g_v512f[4] = vec38_v512f_25;
    g_v512i[7] = vec38_v512i_9;
    g_v128d[0] = vec38_v128d_23;
    g_v128i[5] = vec38_v128i_4;
    g_m16[6] = vec38_m16_22;
    g_m64[3] = vec38_m64_7;
    return vec38_v128i_4;
}

CONV_MS __m512 abi_ms_39(__mmask64 p0, __mmask8 p1) {
    __m256 ms39_v256f_1 = g_v256f[6];
    __m256 ms39_v256f_2 = _mm256_add_ps(ms39_v256f_1, ms39_v256f_1);
    __m512i ms39_v512i_3 = g_v512i[2];
    __m512i ms39_v512i_4 = _mm512_mullo_epi32(ms39_v512i_3, ms39_v512i_3);
    __m128 ms39_v128f_5 = g_v128f[7];
    __m128 ms39_v128f_6 = _mm_sqrt_ps(ms39_v128f_5);
    __m128i ms39_v128i_7 = _mm_set1_epi32(0x11);
    __m128i ms39_v128i_8 = _mm_xor_si128(ms39_v128i_7, ms39_v128i_7);
    __m128 ms39_v128f_9 = _mm_min_ps(ms39_v128f_5, ms39_v128f_6);
    __m128 ms39_v128f_10 = _mm_sqrt_ps(ms39_v128f_6);
    __m512d ms39_v512d_11 = _mm512_set1_pd(2.0);
    __m512d ms39_v512d_12 = _mm512_sqrt_pd(ms39_v512d_11);
    __m128d ms39_v128d_13 = _mm_set1_pd(2.0);
    __m128d ms39_v128d_14 = _mm_add_pd(ms39_v128d_13, ms39_v128d_13);
    __mmask32 ms39_m32_15 = (__mmask32)0x5A5A5A5A;
    __m128i ms39_v128i_16 = abi_reg_11(p1, ms39_v256f_2, ms39_v128f_9, p0, ms39_m32_15, ms39_v128i_7, p1);
    __m256i ms39_v256i_17 = _mm256_set1_epi32(0x33);
    __m256i ms39_v256i_18 = _mm256_sub_epi64(ms39_v256i_17, ms39_v256i_17);
    __m256i ms39_v256i_19 = _mm256_add_epi64(ms39_v256i_18, ms39_v256i_17);
    __m512d ms39_v512d_20 = _mm512_mul_pd(ms39_v512d_12, ms39_v512d_11);
    for (int ms39_i20 = 0; ms39_i20 < (int)((unsigned)p1 & 7); ms39_i20++) {
    __m512 ms39_v512f_22 = _mm512_cvtepi32_ps(ms39_v512i_4);
    __mmask64 ms39_m64_23 = _mm512_movepi8_mask(ms39_v512i_4);
    __m128i ms39_v128i_24 = _mm_sub_epi64(ms39_v128i_16, ms39_v128i_7);
    }
    __m512i ms39_v512i_25 = _mm512_broadcast_i32x4(ms39_v128i_7);
    __m256 ms39_v256f_26 = _mm256_min_ps(ms39_v256f_2, ms39_v256f_2);
    g_v256f[4] = ms39_v256f_1;
    g_v256i[5] = ms39_v256i_19;
    g_v128d[7] = ms39_v128d_14;
    g_v128i[6] = ms39_v128i_8;
    g_m32[4] = ms39_m32_15;
    __m512 ms39_v512f_27 = g_v512f[0];
    return ms39_v512f_27;
}
