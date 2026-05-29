#!/usr/bin/env python3
"""Generate a massive AVX/AVX2/AVX-512/AVX10 torture corpus to flush out lifter
INTERRs and decompilation failures.

The generator emits many C functions that chain intrinsics across every family
the lifter supports, at all vector widths, with masked forms, control flow that
keeps vectors live across basic blocks (the classic trigger for "temporaries
cross block boundaries" INTERRs), high register pressure, memory operands, and
cross-function state via globals. Output is deterministic for a given --seed so
any failure idump finds is reproducible.

Usage:
    gen_torture.py --funcs 400 --seed 1 --out torture_gen.c
"""
from __future__ import annotations

import argparse
import random


# ---- type system -----------------------------------------------------------
# name -> (C type, is it a vector we can store in a global array?)
CTYPE = {
    "v512f": "__m512",  "v512d": "__m512d", "v512i": "__m512i",
    "v256f": "__m256",  "v256d": "__m256d", "v256i": "__m256i",
    "v128f": "__m128",  "v128d": "__m128d", "v128i": "__m128i",
    "v512h": "__m512h", "v256h": "__m256h", "v128h": "__m128h",
    "m8": "__mmask8", "m16": "__mmask16", "m32": "__mmask32", "m64": "__mmask64",
    "s32": "int", "s64": "long long", "u32": "unsigned", "u64": "unsigned long long",
    "f32": "float", "f64": "double",
}
VEC_TYPES = [t for t in CTYPE if t.startswith("v")]
MASK_TYPES = ["m8", "m16", "m32", "m64"]

# ---- op library ------------------------------------------------------------
# Each op: (result_type, [arg_types...], template) where template uses {0},{1}..
OPS: list[tuple[str, list[str], str]] = []

def add(res, args, tmpl):
    OPS.append((res, args, tmpl))

# Packed arithmetic / logic / fma at each width and element type.
for w, fi, di, ii in [(512, "v512f", "v512d", "v512i"),
                      (256, "v256f", "v256d", "v256i"),
                      (128, "v128f", "v128d", "v128i")]:
    p = "" if w == 128 else str(w)
    # float32
    for op in ("add", "sub", "mul", "div", "min", "max"):
        add(fi, [fi, fi], f"_mm{p}_{op}_ps({{0}}, {{1}})")
    add(fi, [fi, fi, fi], f"_mm{p}_fmadd_ps({{0}}, {{1}}, {{2}})")
    add(fi, [fi, fi, fi], f"_mm{p}_fmsub_ps({{0}}, {{1}}, {{2}})")
    add(fi, [fi, fi, fi], f"_mm{p}_fnmadd_ps({{0}}, {{1}}, {{2}})")
    add(fi, [fi], f"_mm{p}_sqrt_ps({{0}})")
    add(fi, [fi, fi], f"_mm{p}_and_ps({{0}}, {{1}})")
    add(fi, [fi, fi], f"_mm{p}_xor_ps({{0}}, {{1}})")
    add(fi, [fi, fi], f"_mm{p}_unpacklo_ps({{0}}, {{1}})")
    add(fi, [fi, fi], f"_mm{p}_shuffle_ps({{0}}, {{1}}, 0x1B)")
    # float64
    for op in ("add", "sub", "mul", "div", "min", "max"):
        add(di, [di, di], f"_mm{p}_{op}_pd({{0}}, {{1}})")
    add(di, [di, di, di], f"_mm{p}_fmadd_pd({{0}}, {{1}}, {{2}})")
    add(di, [di], f"_mm{p}_sqrt_pd({{0}})")
    add(di, [di, di], f"_mm{p}_unpackhi_pd({{0}}, {{1}})")
    # int
    for op in ("add", "sub"):
        add(ii, [ii, ii], f"_mm{p}_{op}_epi32({{0}}, {{1}})")
        add(ii, [ii, ii], f"_mm{p}_{op}_epi64({{0}}, {{1}})")
        add(ii, [ii, ii], f"_mm{p}_{op}_epi8({{0}}, {{1}})")
        add(ii, [ii, ii], f"_mm{p}_{op}_epi16({{0}}, {{1}})")
    add(ii, [ii, ii], f"_mm{p}_mullo_epi32({{0}}, {{1}})")
    add(ii, [ii, ii], f"_mm{p}_and_si{w}({{0}}, {{1}})")
    add(ii, [ii, ii], f"_mm{p}_or_si{w}({{0}}, {{1}})")
    add(ii, [ii, ii], f"_mm{p}_xor_si{w}({{0}}, {{1}})")
    add(ii, [ii], f"_mm{p}_slli_epi32({{0}}, 3)")
    add(ii, [ii, ii], f"_mm{p}_sllv_epi32({{0}}, {{1}})")
    add(ii, [ii, ii], f"_mm{p}_unpacklo_epi32({{0}}, {{1}})")
    add(ii, [ii, ii], f"_mm{p}_shuffle_epi8({{0}}, {{1}})")
    add(ii, [ii], f"_mm{p}_abs_epi32({{0}})")
    # broadcast scalar -> vector
    add(fi, [], f"_mm{p}_set1_ps(1.0f)")
    add(di, [], f"_mm{p}_set1_pd(2.0)")
    add(ii, [], f"_mm{p}_set1_epi32(0x55)")

# AVX-512 specific (512-bit) extras: ternary logic, permutes, lane shuffles, rotate
add("v512i", ["v512i", "v512i", "v512i"], "_mm512_ternarylogic_epi32({0}, {1}, {2}, 0x96)")
add("v512i", ["v512i", "v512i"], "_mm512_permutexvar_epi32({0}, {1})")
add("v512i", ["v512i", "v512i", "v512i"], "_mm512_permutex2var_epi32({0}, {1}, {2})")
add("v512f", ["v512f", "v512f"], "_mm512_shuffle_f32x4({0}, {1}, 0x4E)")
add("v512d", ["v512d", "v512d"], "_mm512_shuffle_f64x2({0}, {1}, 0xEE)")
add("v512i", ["v512i", "v512i"], "_mm512_shuffle_i32x4({0}, {1}, 0x1B)")
add("v512i", ["v512i"], "_mm512_rol_epi32({0}, 5)")
add("v512i", ["v512i", "v512i"], "_mm512_rolv_epi32({0}, {1})")
add("v512i", ["v512i"], "_mm512_conflict_epi32({0})")
add("v512i", ["v512i"], "_mm512_popcnt_epi64({0})")
add("v512i", ["v512i"], "_mm512_lzcnt_epi32({0})")
add("v512f", ["v512f"], "_mm512_getexp_ps({0})")
add("v512f", ["v512f", "v512f"], "_mm512_scalef_ps({0}, {1})")
add("v512f", ["v512f"], "_mm512_rcp14_ps({0})")
add("v512f", ["v512f"], "_mm512_rsqrt14_ps({0})")
add("v512f", ["v512f"], "_mm512_roundscale_ps({0}, 0x10)")
add("v512i", ["v512i", "v512i", "v512i"], "_mm512_dpbusd_epi32({0}, {1}, {2})")
add("v512i", ["v512i", "v512i", "v512i"], "_mm512_dpwssd_epi32({0}, {1}, {2})")
add("v512i", ["v512i", "v512i", "v512i"], "_mm512_madd52lo_epu64({0}, {1}, {2})")

# Extract / insert (cross width)
add("v256d", ["v512d"], "_mm512_extractf64x4_pd({0}, 1)")
add("v128f", ["v512f"], "_mm512_extractf32x4_ps({0}, 2)")
add("v512d", ["v512d", "v128d"], "_mm512_insertf64x2({0}, {1}, 3)")
add("v512f", ["v512f", "v256f"], "_mm512_insertf32x8({0}, {1}, 1)")

# Broadcast vector forms
add("v512f", ["v128f"], "_mm512_broadcast_f32x4({0})")
add("v512i", ["v128i"], "_mm512_broadcast_i32x4({0})")

# Conversions changing type/width (mix the pool)
add("v512i", ["v512f"], "_mm512_cvtps_epi32({0})")
add("v512f", ["v512i"], "_mm512_cvtepi32_ps({0})")
add("v512d", ["v256f"], "_mm512_cvtps_pd({0})")
add("v256f", ["v512d"], "_mm512_cvtpd_ps({0})")
add("v512i", ["v512d"], "_mm512_cvtpd_epi64({0})")
add("v256i", ["v512i"], "_mm512_cvtepi64_epi32({0})")
add("v128i", ["v512i"], "_mm512_cvtepi32_epi8({0})")
add("v512i", ["v128i"], "_mm512_cvtepi8_epi32({0})")

# FP16 family (avx512fp16)
add("v512h", ["v512h", "v512h"], "_mm512_add_ph({0}, {1})")
add("v512h", ["v512h", "v512h"], "_mm512_mul_ph({0}, {1})")
add("v512h", ["v512h", "v512h", "v512h"], "_mm512_fmadd_ph({0}, {1}, {2})")
add("v512h", ["v512h", "v512h", "v512h"], "_mm512_fmsub_ph({0}, {1}, {2})")
add("v512h", ["v512h", "v512h"], "_mm512_fmul_pch({0}, {1})")
add("v512h", ["v512h"], "_mm512_sqrt_ph({0})")
add("v512h", ["v512h"], "_mm512_rcp_ph({0})")
add("v512h", ["v512h"], "_mm512_getexp_ph({0})")
add("v256h", ["v512f"], "_mm512_cvtxps_ph({0})")     # ps(512) -> ph(256) (vcvtps2phx)
add("v512f", ["v256h"], "_mm512_cvtxph_ps({0})")     # ph(256) -> ps(512) (vcvtph2psx)
add("v512i", ["v512h"], "_mm512_cvtph_epi16({0})")
add("v512h", ["v512i"], "_mm512_cvtepi16_ph({0})")
add("v512d", ["v128h"], "_mm512_cvtph_pd({0})")
add("v128h", ["v512d"], "_mm512_cvtpd_ph({0})")

# Mask producers (compare / test / fpclass / movepi)
add("m16", ["v512f", "v512f"], "_mm512_cmp_ps_mask({0}, {1}, _CMP_LT_OQ)")
add("m8",  ["v512d", "v512d"], "_mm512_cmp_pd_mask({0}, {1}, _CMP_GE_OQ)")
add("m64", ["v512i", "v512i"], "_mm512_cmp_epi8_mask({0}, {1}, 1)")
add("m16", ["v512i", "v512i"], "_mm512_cmp_epi32_mask({0}, {1}, 2)")
add("m16", ["v512i", "v512i"], "_mm512_cmp_epu32_mask({0}, {1}, 4)")
add("m16", ["v512i", "v512i"], "_mm512_test_epi32_mask({0}, {1})")
add("m16", ["v512i", "v512i"], "_mm512_testn_epi32_mask({0}, {1})")
add("m16", ["v512f"], "_mm512_fpclass_ps_mask({0}, 0x3)")
add("m64", ["v512i"], "_mm512_movepi8_mask({0})")
add("m16", ["v512i"], "_mm512_movepi32_mask({0})")
add("m32", ["v512h", "v512h"], "_mm512_cmp_ph_mask({0}, {1}, 1)")

# Mask consumers / ALU
add("m16", ["m16", "m16"], "_kand_mask16({0}, {1})")
add("m16", ["m16", "m16"], "_kor_mask16({0}, {1})")
add("m16", ["m16", "m16"], "_kxor_mask16({0}, {1})")
add("m16", ["m16"], "_knot_mask16({0})")
add("m16", ["m16"], "_kshiftli_mask16({0}, 2)")
add("m16", ["m16"], "_kshiftri_mask16({0}, 1)")
add("m8",  ["m8", "m8"], "_kand_mask8({0}, {1})")
add("m32", ["m32", "m32"], "_kxor_mask32({0}, {1})")
add("m64", ["m64", "m64"], "_kor_mask64({0}, {1})")
add("m32", ["m16", "m16"], "_mm512_kunpackw({0}, {1})")
add("m16", ["m8", "m8"], "_mm512_kunpackb({0}, {1})")
add("u32", ["m16"], "_cvtmask16_u32({0})")
add("m16", ["u32"], "_cvtu32_mask16({0})")

# Mask-controlled select / blend (consume a mask + 2 vectors)
add("v512f", ["m16", "v512f", "v512f"], "_mm512_mask_blend_ps({0}, {1}, {2})")
add("v512i", ["m16", "v512i", "v512i"], "_mm512_mask_blend_epi32({0}, {1}, {2})")
add("v512d", ["m8", "v512d", "v512d"], "_mm512_mask_blend_pd({0}, {1}, {2})")

# Masked (merge) compute: src, k, a, b
add("v512f", ["v512f", "m16", "v512f", "v512f"], "_mm512_mask_add_ps({0}, {1}, {2}, {3})")
add("v512f", ["m16", "v512f", "v512f"], "_mm512_maskz_mul_ps({0}, {1}, {2})")
add("v512i", ["v512i", "m16", "v512i", "v512i"], "_mm512_mask_add_epi32({0}, {1}, {2}, {3})")
add("v512f", ["v512f", "v512f", "v512f", "m16"], "_mm512_mask3_fmadd_ps({0}, {1}, {2}, {3})")

# Mask -> vector broadcasts
add("v512i", ["m8"], "_mm512_broadcastmb_epi64({0})")
add("v512i", ["m16"], "_mm512_broadcastmw_epi32({0})")

# index ops over types
by_res: dict[str, list[int]] = {}
for i, (res, _a, _t) in enumerate(OPS):
    by_res.setdefault(res, []).append(i)


# ---- generator -------------------------------------------------------------
GLOBALS_N = 12  # per global pool


def header() -> str:
    g = ["#include <immintrin.h>", "#include <stdint.h>", "",
         "/* cross-function state pools (force loads/stores + dataflow) */"]
    for t in VEC_TYPES + MASK_TYPES:
        g.append(f"{CTYPE[t]} g_{t}[{GLOBALS_N}];")
    g.append("static const float  fbuf[64];")
    g.append("static const double dbuf[32];")
    g.append("static const int    ibuf[64];")
    g.append("")
    return "\n".join(g)


def mem_load(rng, t):
    """A memory-operand load producing type t (or None)."""
    if t == "v512f": return "_mm512_loadu_ps(fbuf)"
    if t == "v256f": return "_mm256_loadu_ps(fbuf)"
    if t == "v128f": return "_mm_loadu_ps(fbuf)"
    if t == "v512d": return "_mm512_loadu_pd(dbuf)"
    if t == "v512i": return "_mm512_loadu_si512((const void*)ibuf)"
    if t == "v256i": return "_mm256_loadu_si256((const void*)ibuf)"
    if t == "v128i": return "_mm_loadu_si128((const void*)ibuf)"
    return None


class Func:
    def __init__(self, rng, idx):
        self.rng = rng
        self.idx = idx
        self.lines: list[str] = []
        self.pool: dict[str, list[str]] = {}
        self.counter = 0

    def fresh(self, t):
        self.counter += 1
        return f"{t}_{self.counter}"

    def have(self, t):
        return self.pool.get(t)

    def put(self, t, name):
        self.pool.setdefault(t, []).append(name)

    def pick(self, t):
        lst = self.pool.get(t)
        return self.rng.choice(lst) if lst else None

    def ensure(self, t):
        """Return a variable of type t, creating a seed if needed."""
        v = self.pick(t)
        if v:
            return v
        name = self.fresh(t)
        ml = mem_load(self.rng, t)
        if ml and self.rng.random() < 0.5:
            self.lines.append(f"    {CTYPE[t]} {name} = {ml};")
        elif t in MASK_TYPES:
            self.lines.append(f"    {CTYPE[t]} {name} = ({CTYPE[t]})(g_{t}[{self.rng.randrange(GLOBALS_N)}] ^ {self.rng.randrange(1,255)});")
        elif t in ("s32", "u32"): self.lines.append(f"    {CTYPE[t]} {name} = {self.rng.randrange(1,99)};")
        elif t in ("s64", "u64"): self.lines.append(f"    {CTYPE[t]} {name} = {self.rng.randrange(1,99)}LL;")
        elif t in ("f32",): self.lines.append(f"    float {name} = 1.5f;")
        elif t in ("f64",): self.lines.append(f"    double {name} = 2.5;")
        else:
            # vector: read a global (cross-function dataflow)
            self.lines.append(f"    {CTYPE[t]} {name} = g_{t}[{self.rng.randrange(GLOBALS_N)}];")
        self.put(t, name)
        return name

    def emit_op(self):
        res, args, tmpl = self.rng.choice(OPS)
        argv = [self.ensure(a) for a in args]
        name = self.fresh(res)
        expr = tmpl.format(*argv)
        self.lines.append(f"    {CTYPE[res]} {name} = {expr};")
        self.put(res, name)
        return res, name

    def snapshot(self):
        return {t: list(v) for t, v in self.pool.items()}

    def restore(self, snap):
        # Discard variables declared inside a nested C scope (they're now out
        # of scope) but keep the pre-block pool intact.
        self.pool = {t: list(v) for t, v in snap.items()}

    def emit_block(self, n):
        for _ in range(n):
            self.emit_op()

    def maybe_branch(self):
        """Wrap some ops in control flow so vectors stay live across blocks."""
        mt = self.pick("m16") or self.pick("m8") or self.pick("m32") or self.pick("m64")
        cond = f"(unsigned long long){mt}" if mt else self.ensure("s32")
        kind = self.rng.random()
        snap = self.snapshot()
        if kind < 0.5:
            self.lines.append(f"    if (({cond}) & 1ULL) {{")
            self.emit_block(self.rng.randint(2, 6))
            self.lines.append("    }")
        else:
            iv = self.fresh("i")
            self.lines.append(f"    for (int {iv} = 0; {iv} < (int)(({cond}) & 7); {iv}++) {{")
            self.emit_block(self.rng.randint(2, 5))
            self.lines.append("    }")
        self.restore(snap)

    def store_some_globals(self):
        for t in VEC_TYPES + MASK_TYPES:
            v = self.pick(t)
            if v and self.rng.random() < 0.4:
                self.lines.append(f"    g_{t}[{self.rng.randrange(GLOBALS_N)}] = {v};")

    def render(self):
        rng = self.rng
        # choose a return type that we will definitely produce
        ret = rng.choice(VEC_TYPES + MASK_TYPES)
        # function signature: a few vector params of mixed types
        nparams = rng.randint(2, 5)
        ptypes = [rng.choice(VEC_TYPES) for _ in range(nparams)]
        params = []
        for i, pt in enumerate(ptypes):
            pn = f"p{i}"
            params.append(f"{CTYPE[pt]} {pn}")
            self.put(pt, pn)
        sig = f"{CTYPE[ret]} torture_{self.idx}({', '.join(params)})"
        self.emit_block(rng.randint(8, 18))
        if rng.random() < 0.8:
            self.maybe_branch()
        self.emit_block(rng.randint(4, 12))
        if rng.random() < 0.6:
            self.maybe_branch()
        self.store_some_globals()
        rv = self.ensure(ret)
        body = "\n".join(self.lines)
        return f"{sig} {{\n{body}\n    return {rv};\n}}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="torture_gen.c")
    ap.add_argument("--avx2", action="store_true",
                    help="restrict to AVX/AVX2 (128/256-bit, no AVX-512/FP16/mask) "
                         "so the corpus builds for 32-bit (ELF32/PE32) targets")
    args = ap.parse_args()

    if args.avx2:
        global VEC_TYPES, MASK_TYPES, OPS, by_res
        VEC_TYPES = ["v256f", "v256d", "v256i", "v128f", "v128d", "v128i"]
        MASK_TYPES = []
        allowed = set(VEC_TYPES) | {"s32", "s64", "u32", "u64", "f32", "f64"}
        OPS = [(res, a, t) for (res, a, t) in OPS
               if res in allowed and all(x in allowed for x in a)]
        by_res = {}
        for i, (res, _a, _t) in enumerate(OPS):
            by_res.setdefault(res, []).append(i)

    rng = random.Random(args.seed)
    parts = [header(),
             f"/* generated: funcs={args.funcs} seed={args.seed} */", ""]
    for i in range(args.funcs):
        parts.append(Func(rng, i).render())
    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {args.out}: {args.funcs} functions, {len(OPS)} op templates, seed {args.seed}")


if __name__ == "__main__":
    main()
