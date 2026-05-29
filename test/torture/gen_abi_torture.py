#!/usr/bin/env python3
"""Generate an intrinsic-based C torture corpus that varies CALLING CONVENTIONS
and ABI to stress how the lifter handles vector registers / masks crossing real
function boundaries -- a prime INTERR suspect in the wild.

Every emitted function is tagged with a randomly chosen attribute drawn from
ms_abi / sysv_abi / vectorcall / regcall.  Functions form CALL CHAINS: a caller
of one convention invokes earlier-defined callees of a *different* convention,
passing and returning __m512/__m512d/__m512i/__mmask16/__mmask64 by value, so
vector args and returns flow across convention boundaries and force the
prologue/epilogue spills that the lifter must model.

The conventions differ in exactly the ways that hurt the lifter:
  * ms_abi   -> xmm6-15 (and zmm6-31) are callee-saved -> the compiler emits
                vector spills around calls and in prologue/epilogue.
  * sysv_abi -> all vector regs caller-saved -> spills land at the call sites.
  * vectorcall / regcall -> __m512/__m256/__m128 and masks travel in vector
                registers, with deep arg lists overflowing into stack slots.

We mix widths (128/256/512), masked intrinsics, deep argument lists (8+ vector
params, which overflow the register-passing budget and spill to the stack), and
a global pool of __m512 / __mmask values written and read across functions.

The corpus compiles freestanding (no libc calls) on BOTH:
  * gcc Linux            (ms_abi is a valid attribute on a SysV target)
  * clang --target=x86_64-pc-windows-gnu  (sysv_abi is valid on a Windows target)

Usage:
    gen_abi_torture.py --funcs 60 --seed 1 --out abitort.c
"""
from __future__ import annotations

import argparse
import random


# ---- type system -----------------------------------------------------------
# key -> C type.  We deliberately only pass/return types that all four
# conventions accept by value at every width.
CTYPE = {
    "v512f": "__m512",  "v512d": "__m512d", "v512i": "__m512i",
    "v256f": "__m256",  "v256d": "__m256d", "v256i": "__m256i",
    "v128f": "__m128",  "v128d": "__m128d", "v128i": "__m128i",
    "m8": "__mmask8", "m16": "__mmask16", "m32": "__mmask32", "m64": "__mmask64",
}
VEC_TYPES = ["v512f", "v512d", "v512i", "v256f", "v256d", "v256i",
             "v128f", "v128d", "v128i"]
MASK_TYPES = ["m8", "m16", "m32", "m64"]
ALL_TYPES = VEC_TYPES + MASK_TYPES

# Types we route across function boundaries (the task names these explicitly).
# All of VEC + MASK are passable; this list just biases toward the wide forms.
BOUNDARY_TYPES = ["v512f", "v512d", "v512i", "m16", "m64",
                  "v256f", "v256i", "v128f", "v128i", "m8", "m32"]


# ---- calling conventions ---------------------------------------------------
# tag used in names -> the CONV_* macro applied to the function.
#
# Portability note (verified against the project toolchain): gcc on its SysV
# x86-64 target accepts ms_abi / sysv_abi but does NOT support vectorcall /
# regcall (it ignores the attribute and silently degrades to the default
# convention).  clang --target=x86_64-pc-windows-gnu accepts ALL FOUR cleanly.
# So we emit a macro shim (CONV_MS/CONV_SYSV/CONV_VEC/CONV_REG): on a compiler
# that supports vectorcall/regcall the VEC/REG functions get the genuine
# vector-register-passing conventions; elsewhere they fall back to a real,
# distinct ABI attribute so the call chains stay legal and meaningfully
# mixed.  The Windows/clang build therefore exercises the full convention
# matrix (idump loads PE natively), and both builds compile warning-clean.
CONV_TAGS = ["ms", "sysv", "vec", "reg"]
CONV_MACRO = {"ms": "CONV_MS", "sysv": "CONV_SYSV",
              "vec": "CONV_VEC", "reg": "CONV_REG"}

CONV_PROLOGUE = r"""
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
"""


# ---- op library: build a result of type `res` from in-scope vars -----------
# Each op: (result_type, [arg_types...], template) using {0},{1},...
OPS: list[tuple[str, list[str], str]] = []


def add(res, args, tmpl):
    OPS.append((res, args, tmpl))


for w, fi, di, ii in [(512, "v512f", "v512d", "v512i"),
                      (256, "v256f", "v256d", "v256i"),
                      (128, "v128f", "v128d", "v128i")]:
    p = "" if w == 128 else str(w)
    for op in ("add", "sub", "mul", "min", "max"):
        add(fi, [fi, fi], f"_mm{p}_{op}_ps({{0}}, {{1}})")
    add(fi, [fi, fi, fi], f"_mm{p}_fmadd_ps({{0}}, {{1}}, {{2}})")
    add(fi, [fi], f"_mm{p}_sqrt_ps({{0}})")
    add(fi, [fi, fi], f"_mm{p}_unpacklo_ps({{0}}, {{1}})")
    for op in ("add", "sub", "mul", "max"):
        add(di, [di, di], f"_mm{p}_{op}_pd({{0}}, {{1}})")
    add(di, [di, di, di], f"_mm{p}_fmadd_pd({{0}}, {{1}}, {{2}})")
    add(di, [di], f"_mm{p}_sqrt_pd({{0}})")
    for op in ("add", "sub"):
        add(ii, [ii, ii], f"_mm{p}_{op}_epi32({{0}}, {{1}})")
        add(ii, [ii, ii], f"_mm{p}_{op}_epi64({{0}}, {{1}})")
    add(ii, [ii, ii], f"_mm{p}_mullo_epi32({{0}}, {{1}})")
    add(ii, [ii, ii], f"_mm{p}_and_si{w}({{0}}, {{1}})")
    add(ii, [ii, ii], f"_mm{p}_xor_si{w}({{0}}, {{1}})")
    add(ii, [ii], f"_mm{p}_slli_epi32({{0}}, 3)")
    add(ii, [ii, ii], f"_mm{p}_unpacklo_epi32({{0}}, {{1}})")

# 512-bit AVX-512 extras
add("v512i", ["v512i", "v512i", "v512i"], "_mm512_ternarylogic_epi32({0}, {1}, {2}, 0x96)")
add("v512i", ["v512i", "v512i"], "_mm512_permutexvar_epi32({0}, {1})")
add("v512i", ["v512i"], "_mm512_rol_epi32({0}, 5)")
add("v512f", ["v512f", "v512f"], "_mm512_scalef_ps({0}, {1})")
add("v512f", ["v512f"], "_mm512_rcp14_ps({0})")

# width crossings
add("v256d", ["v512d"], "_mm512_extractf64x4_pd({0}, 1)")
add("v128f", ["v512f"], "_mm512_extractf32x4_ps({0}, 2)")
add("v512f", ["v128f"], "_mm512_broadcast_f32x4({0})")
add("v512i", ["v512f"], "_mm512_cvtps_epi32({0})")
add("v512f", ["v512i"], "_mm512_cvtepi32_ps({0})")
add("v256i", ["v512i"], "_mm512_cvtepi64_epi32({0})")
add("v512i", ["v128i"], "_mm512_broadcast_i32x4({0})")

# mask producers
add("m16", ["v512f", "v512f"], "_mm512_cmp_ps_mask({0}, {1}, _CMP_LT_OQ)")
add("m8",  ["v512d", "v512d"], "_mm512_cmp_pd_mask({0}, {1}, _CMP_GE_OQ)")
add("m64", ["v512i", "v512i"], "_mm512_cmp_epi8_mask({0}, {1}, 1)")
add("m16", ["v512i", "v512i"], "_mm512_cmp_epi32_mask({0}, {1}, 2)")
add("m32", ["v512i", "v512i"], "_mm512_cmp_epi16_mask({0}, {1}, 2)")
add("m16", ["v512i"], "_mm512_movepi32_mask({0})")
add("m64", ["v512i"], "_mm512_movepi8_mask({0})")

# mask ALU
add("m16", ["m16", "m16"], "_kand_mask16({0}, {1})")
add("m16", ["m16", "m16"], "_kor_mask16({0}, {1})")
add("m16", ["m16", "m16"], "_kxor_mask16({0}, {1})")
add("m16", ["m16"], "_knot_mask16({0})")
add("m8",  ["m8", "m8"], "_kand_mask8({0}, {1})")
add("m32", ["m32", "m32"], "_kxor_mask32({0}, {1})")
add("m64", ["m64", "m64"], "_kor_mask64({0}, {1})")
add("m32", ["m16", "m16"], "_mm512_kunpackw({0}, {1})")
add("m16", ["m8", "m8"], "_mm512_kunpackb({0}, {1})")

# mask-controlled blends / merge-masked compute (vectors + masks together)
add("v512f", ["m16", "v512f", "v512f"], "_mm512_mask_blend_ps({0}, {1}, {2})")
add("v512i", ["m16", "v512i", "v512i"], "_mm512_mask_blend_epi32({0}, {1}, {2})")
add("v512d", ["m8", "v512d", "v512d"], "_mm512_mask_blend_pd({0}, {1}, {2})")
add("v512f", ["v512f", "m16", "v512f", "v512f"], "_mm512_mask_add_ps({0}, {1}, {2}, {3})")
add("v512f", ["m16", "v512f", "v512f"], "_mm512_maskz_mul_ps({0}, {1}, {2})")
add("v512i", ["v512i", "m16", "v512i", "v512i"], "_mm512_mask_add_epi32({0}, {1}, {2}, {3})")

# mask -> vector broadcasts (keep masks live into vector domain)
add("v512i", ["m8"], "_mm512_broadcastmb_epi64({0})")
add("v512i", ["m16"], "_mm512_broadcastmw_epi32({0})")


GLOBALS_N = 8  # entries per global pool


# ---- type-builder helpers (no memory loads => freestanding/no-libc) --------
def zero_expr(t: str) -> str:
    """A constant of type t, built without touching memory or libc."""
    if t == "v512f": return "_mm512_set1_ps(1.0f)"
    if t == "v512d": return "_mm512_set1_pd(2.0)"
    if t == "v512i": return "_mm512_set1_epi32(0x55)"
    if t == "v256f": return "_mm256_set1_ps(1.0f)"
    if t == "v256d": return "_mm256_set1_pd(2.0)"
    if t == "v256i": return "_mm256_set1_epi32(0x33)"
    if t == "v128f": return "_mm_set1_ps(1.0f)"
    if t == "v128d": return "_mm_set1_pd(2.0)"
    if t == "v128i": return "_mm_set1_epi32(0x11)"
    if t == "m8":  return "(__mmask8)0xA5"
    if t == "m16": return "(__mmask16)0x5A5A"
    if t == "m32": return "(__mmask32)0x5A5A5A5A"
    if t == "m64": return "(__mmask64)0x5A5A5A5A5A5A5A5AULL"
    raise KeyError(t)


# index ops by result type
by_res: dict[str, list[int]] = {}
for _i, (_res, _a, _t) in enumerate(OPS):
    by_res.setdefault(_res, []).append(_i)


def header(funcs: int, seed: int) -> str:
    g = ["#include <immintrin.h>", "#include <stdint.h>", "",
         f"/* ABI/calling-convention torture: funcs={funcs} seed={seed} */",
         "/* compiles on gcc(Linux) and clang --target=x86_64-pc-windows-gnu */",
         CONV_PROLOGUE,
         "/* cross-function vector/mask state pools (force loads/stores) */"]
    for t in ALL_TYPES:
        g.append(f"{CTYPE[t]} g_{t}[{GLOBALS_N}];")
    g.append("")
    return "\n".join(g)


class Func:
    """One generated function with a chosen convention and signature."""

    def __init__(self, rng, idx, conv_tag, callees):
        self.rng = rng
        self.idx = idx
        self.conv_macro = CONV_MACRO[conv_tag]
        self.conv_tag = conv_tag
        self.callees = callees   # list of dicts describing earlier functions
        self.lines: list[str] = []
        self.pool: dict[str, list[str]] = {}
        self.counter = 0
        # gcc-safety bookkeeping (see emit_call): on the gcc build the
        # convention macros degrade vec->sysv_abi and reg->ms_abi, so the
        # "ms family" callees are those originally tagged ms or reg.  gcc hits
        # an expand_call ICE (calls.cc:3721 -- a compiler bug, not a lifter
        # issue) in any SysV-side function that makes TWO OR MORE calls when at
        # least one callee is ms-family.  Either call alone is fine.  So the
        # safe, sufficient rule is: an ms-family call must be the ONLY call in
        # the function.  We therefore allow either (a) a single ms-family call,
        # or (b) any number of sysv-family calls -- never both.  This keeps the
        # corpus reliably compilable while still crossing every ABI boundary;
        # the Windows/clang build (full genuine matrix) applies the same rule
        # so both builds stay structurally identical.
        self.n_calls = 0
        self.called_ms_family = False
        # decide signature now so prototypes and definitions agree
        self.ret = rng.choice(BOUNDARY_TYPES)
        # deep arg lists: bias high so register budgets overflow to the stack
        nparams = rng.randint(2, 10)
        self.ptypes = [rng.choice(BOUNDARY_TYPES) for _ in range(nparams)]
        self.name = f"abi_{conv_tag}_{idx}"

    # -- signature -----------------------------------------------------------
    def param_list(self) -> str:
        return ", ".join(f"{CTYPE[t]} p{i}" for i, t in enumerate(self.ptypes))

    def signature(self) -> str:
        return f"{self.conv_macro} {CTYPE[self.ret]} {self.name}({self.param_list()})"

    def prototype(self) -> str:
        return self.signature() + ";"

    # -- value pool ----------------------------------------------------------
    def fresh(self, t):
        self.counter += 1
        return f"{self.conv_tag}{self.idx}_{t}_{self.counter}"

    def put(self, t, name):
        self.pool.setdefault(t, []).append(name)

    def pick(self, t):
        lst = self.pool.get(t)
        return self.rng.choice(lst) if lst else None

    def ensure(self, t):
        v = self.pick(t)
        if v:
            return v
        name = self.fresh(t)
        r = self.rng.random()
        if r < 0.5:
            # read a global -> cross-function dataflow
            self.lines.append(
                f"    {CTYPE[t]} {name} = g_{t}[{self.rng.randrange(GLOBALS_N)}];")
        else:
            self.lines.append(f"    {CTYPE[t]} {name} = {zero_expr(t)};")
        self.put(t, name)
        return name

    def emit_op(self):
        res, args, tmpl = self.rng.choice(OPS)
        argv = [self.ensure(a) for a in args]
        name = self.fresh(res)
        self.lines.append(f"    {CTYPE[res]} {name} = {tmpl.format(*argv)};")
        self.put(res, name)
        return res, name

    def emit_block(self, n):
        for _ in range(n):
            self.emit_op()

    # -- the point of this generator: cross-convention calls -----------------
    def emit_call(self):
        """Call an earlier function (a different convention when possible),
        passing in-scope vectors/masks and capturing the vector/mask result so
        it flows back into the dataflow -> spills across the boundary."""
        if not self.callees:
            return False
        ms_family = {"ms", "reg"}
        # If we've already called an ms-family callee, no further call at all.
        if self.called_ms_family:
            return False
        avail = self.callees
        if self.n_calls >= 1:
            # We've already made a call -> any new call must NOT be ms-family
            # (an ms-family call must be the lone call in the function).
            avail = [c for c in avail if c["tag"] not in ms_family]
            if not avail:
                return False
        # prefer a callee of a *different* convention to force ABI crossing
        diff = [c for c in avail if c["tag"] != self.conv_tag]
        pool = diff if (diff and self.rng.random() < 0.85) else avail
        c = self.rng.choice(pool)
        if c["tag"] in ms_family:
            self.called_ms_family = True
        self.n_calls += 1
        argv = [self.ensure(t) for t in c["ptypes"]]
        name = self.fresh(c["ret"])
        self.lines.append(
            f"    {CTYPE[c['ret']]} {name} = {c['name']}({', '.join(argv)});")
        self.put(c["ret"], name)
        return True

    def maybe_branch(self):
        mt = (self.pick("m16") or self.pick("m8") or self.pick("m32")
              or self.pick("m64"))
        if not mt:
            return
        snap = {t: list(v) for t, v in self.pool.items()}
        if self.rng.random() < 0.5:
            self.lines.append(f"    if ((unsigned long long){mt} & 1ULL) {{")
            # NB: keep cross-convention calls OUT of nested scopes -- a call
            # with deep vector args inside a branch can trip a gcc
            # expand_call ICE (compiler bug, not a lifter issue).  Top-level
            # calls already cross ABI boundaries and force the spills we want.
            self.emit_block(self.rng.randint(2, 5))
            self.lines.append("    }")
        else:
            iv = f"{self.conv_tag}{self.idx}_i{self.counter}"
            self.counter += 1
            self.lines.append(
                f"    for (int {iv} = 0; {iv} < (int)((unsigned){mt} & 7); {iv}++) {{")
            self.emit_block(self.rng.randint(2, 4))
            self.lines.append("    }")
        # variables declared inside the nested scope are now out of scope
        self.pool = {t: list(v) for t, v in snap.items()}

    def store_some_globals(self):
        for t in ALL_TYPES:
            v = self.pick(t)
            if v and self.rng.random() < 0.4:
                self.lines.append(
                    f"    g_{t}[{self.rng.randrange(GLOBALS_N)}] = {v};")

    # -- emit a definition ---------------------------------------------------
    def render(self) -> str:
        rng = self.rng
        # seed the pool with the parameters (vectors/masks live on entry)
        for i, pt in enumerate(self.ptypes):
            self.put(pt, f"p{i}")
        self.emit_block(rng.randint(4, 10))
        # at least one cross-convention call early when callees exist
        if self.callees:
            self.emit_call()
            if rng.random() < 0.6:
                self.emit_call()
        self.emit_block(rng.randint(2, 6))
        if rng.random() < 0.8:
            self.maybe_branch()
        # another call after the branch -> result must survive the merge
        if self.callees and rng.random() < 0.7:
            self.emit_call()
        self.emit_block(rng.randint(2, 6))
        if rng.random() < 0.5:
            self.maybe_branch()
        self.store_some_globals()
        rv = self.ensure(self.ret)
        body = "\n".join(self.lines)
        return f"{self.signature()} {{\n{body}\n    return {rv};\n}}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="abitort.c")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # First decide each function's convention and signature so we can emit a
    # prototypes block before any definition (call chains compile in any order).
    funcs: list[Func] = []
    descs: list[dict] = []
    for i in range(args.funcs):
        tag = rng.choice(CONV_TAGS)
        # callees are strictly earlier functions -> definitions never call a
        # not-yet-declared symbol's *type*, but prototypes cover all anyway.
        f = Func(rng, i, tag, list(descs))
        funcs.append(f)
        descs.append({"name": f.name, "tag": tag,
                      "ret": f.ret, "ptypes": f.ptypes})

    parts = [header(args.funcs, args.seed),
             "/* forward declarations (call chains across conventions) */"]
    parts += [f.prototype() for f in funcs]
    parts.append("")
    parts += [f.render() for f in funcs]

    with open(args.out, "w") as fh:
        fh.write("\n".join(parts))

    convs = {}
    for d in descs:
        convs[d["tag"]] = convs.get(d["tag"], 0) + 1
    print(f"wrote {args.out}: {args.funcs} functions, {len(OPS)} op templates, "
          f"seed {args.seed}, conv mix={convs}")


if __name__ == "__main__":
    main()
