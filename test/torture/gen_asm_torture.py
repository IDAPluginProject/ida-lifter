#!/usr/bin/env python3
"""Generate a raw-assembly torture corpus (AT&T syntax) that pushes the lifter
to the fringes of what's legal.

Unlike the intrinsic corpus, this emits hand-chosen instruction sequences with
*explicit* registers and NO save/restore, so:
  * functions read registers they never wrote (the lifter must synthesize
    __readzmm/__readmask of "cross-function" state),
  * high EVEX registers (zmm16-31, k1-7) appear freely,
  * masked {k}{z}, mixed sizes, memory operands and odd orderings are mixed,
all of which stress the microcode the lifter emits (temporaries crossing block
boundaries, k/zmm modeling) — the classic INTERR triggers. The assembler
validates legality, so every emitted instruction is encodable.

Usage: gen_asm_torture.py --funcs 400 --seed 1 --out atort.s
"""
from __future__ import annotations
import argparse
import random

ZMM = lambda r: f"%zmm{r}"
YMM = lambda r: f"%ymm{r}"
XMM = lambda r: f"%xmm{r}"
K   = lambda r: f"%k{r}"
GPR32 = ["%eax", "%ebx", "%ecx", "%edx", "%esi", "%edi", "%r8d", "%r9d"]
GPR64 = ["%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11"]


def zr(rng): return rng.randrange(32)
def kr(rng): return rng.randrange(1, 8)   # k1-k7 (k0 special)
def kr0(rng): return rng.randrange(8)
def mem(rng): return rng.choice(["(%rdi)", "(%rsi)", "(%rdx)", "0x40(%rdi)"])


# Each template is a function (rng)->str producing one legal AT&T instruction.
# src operands first, dest last (AT&T order).
def TEMPLATES():
    T = []
    A = lambda f: T.append(f)
    # --- packed FP/int ALU at all widths ---
    for reg, suf512 in [(ZMM, "")]:
        pass
    for W, R in [(512, ZMM), (256, YMM), (128, XMM)]:
        A(lambda r, R=R: f"vaddps {R(zr(r))}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R: f"vmulpd {R(zr(r))}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R: f"vsubps {R(zr(r))}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R: f"vpaddd {R(zr(r))}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R: f"vpmulld {R(zr(r))}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R: f"vfmadd213ps {R(zr(r))}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R: f"vsqrtpd {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R: f"vpxorq {R(zr(r))}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R: f"vunpcklps {R(zr(r))}, {R(zr(r))}, {R(zr(r))}")
    # --- masked / zero-masked (fringe) ---
    A(lambda r: f"vaddps {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}")
    A(lambda r: f"vmulpd {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}{{z}}")
    A(lambda r: f"vsqrtpd {ZMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}")
    A(lambda r: f"vpaddd {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}{{z}}")
    # --- compares / tests / fpclass -> mask ---
    A(lambda r: f"vpcmpd ${r.randrange(8)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {K(kr(r))}")
    A(lambda r: f"vcmpps ${r.randrange(32)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {K(kr(r))}")
    A(lambda r: f"vptestmd {ZMM(zr(r))}, {ZMM(zr(r))}, {K(kr(r))}")
    A(lambda r: f"vptestnmq {ZMM(zr(r))}, {ZMM(zr(r))}, {K(kr(r))}")
    A(lambda r: f"vfpclassps ${r.randrange(128)}, {ZMM(zr(r))}, {K(kr(r))}")
    A(lambda r: f"vpcmpeqd {ZMM(zr(r))}, {ZMM(zr(r))}, {K(kr(r))}")
    # --- mask ALU ---
    A(lambda r: f"kandw {K(kr(r))}, {K(kr(r))}, {K(kr(r))}")
    A(lambda r: f"korw {K(kr(r))}, {K(kr(r))}, {K(kr(r))}")
    A(lambda r: f"kxorq {K(kr(r))}, {K(kr(r))}, {K(kr(r))}")
    A(lambda r: f"knotw {K(kr(r))}, {K(kr(r))}")
    A(lambda r: f"kshiftlw ${r.randrange(16)}, {K(kr(r))}, {K(kr(r))}")
    A(lambda r: f"kunpckbw {K(kr(r))}, {K(kr(r))}, {K(kr(r))}")
    A(lambda r: f"kmovw {K(kr(r))}, {r.choice(GPR32)}")
    A(lambda r: f"kmovw {r.choice(GPR32)}, {K(kr(r))}")
    A(lambda r: f"kmovq {K(kr(r))}, {K(kr(r))}")
    # --- mask <-> vector ---
    A(lambda r: f"vpmovd2m {ZMM(zr(r))}, {K(kr(r))}")
    A(lambda r: f"vpmovm2d {K(kr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vpmovb2m {ZMM(zr(r))}, {K(kr(r))}")
    A(lambda r: f"vpbroadcastmw2d {K(kr(r))}, {ZMM(zr(r))}")
    # --- broadcasts (incl GPR source) ---
    A(lambda r: f"vpbroadcastd {r.choice(GPR32)}, {ZMM(zr(r))}")
    A(lambda r: f"vpbroadcastq {r.choice(GPR64)}, {ZMM(zr(r))}")
    A(lambda r: f"vbroadcastss {XMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vbroadcastsd {XMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}")
    # --- permute / shuffle / ternlog / rotate ---
    A(lambda r: f"vpermps {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vpermt2d {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vpermi2ps {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vshuff64x2 $0xEE, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vpternlogd $0x96, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vprold $5, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vplzcntd {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vpconflictd {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vpopcntq {ZMM(zr(r))}, {ZMM(zr(r))}")
    # --- extract / insert (cross width) ---
    A(lambda r: f"vextractf64x4 $1, {ZMM(zr(r))}, {YMM(zr(r))}")
    A(lambda r: f"vextractf32x4 $2, {ZMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vinsertf64x2 $3, {XMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vinserti32x8 $1, {YMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    # --- conversions (size/type changing) ---
    A(lambda r: f"vcvtdq2ps {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vcvtps2dq {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vcvtph2ps {YMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vcvtps2ph $0, {ZMM(zr(r))}, {YMM(zr(r))}")
    A(lambda r: f"vcvtpd2ps {ZMM(zr(r))}, {YMM(zr(r))}")
    A(lambda r: f"vcvtudq2ps {ZMM(zr(r))}, {ZMM(zr(r))}")
    # --- FP16 ---
    A(lambda r: f"vaddph {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vfmadd231ph {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vsqrtph {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vrcpph {ZMM(zr(r))}, {ZMM(zr(r))}")
    # --- memory operands (loads/stores/broadcast-from-mem) ---
    A(lambda r: f"vmovups {mem(r)}, {ZMM(zr(r))}")
    A(lambda r: f"vmovaps {ZMM(zr(r))}, {mem(r)}")
    A(lambda r: f"vmovdqu64 {mem(r)}, {ZMM(zr(r))}")
    A(lambda r: f"vaddps {mem(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vbroadcastss {mem(r)}, {ZMM(zr(r))}")
    # --- gather / scatter (vSIB + mask: classic INTERR territory) ---
    A(lambda r: f"vgatherdps (%rdi,{ZMM(zr(r))},4), {ZMM(zr(r))}{{{K(kr(r))}}}")
    A(lambda r: f"vpgatherdd (%rsi,{ZMM(zr(r))},4), {ZMM(zr(r))}{{{K(kr(r))}}}")
    A(lambda r: f"vgatherqpd (%rdi,{ZMM(zr(r))},8), {ZMM(zr(r))}{{{K(kr(r))}}}")
    A(lambda r: f"vscatterdps {ZMM(zr(r))}, (%rdx,{ZMM(zr(r))},4){{{K(kr(r))}}}")
    A(lambda r: f"vpscatterdd {ZMM(zr(r))}, (%rdi,{ZMM(zr(r))},4){{{K(kr(r))}}}")
    # --- compress / expand (mask + memory) ---
    A(lambda r: f"vpcompressd {ZMM(zr(r))}, {mem(r)}{{{K(kr(r))}}}")
    A(lambda r: f"vpexpandd {mem(r)}, {ZMM(zr(r))}{{{K(kr(r))}}}")
    A(lambda r: f"vcompressps {ZMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}")
    A(lambda r: f"vexpandpd {ZMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}{{z}}")
    # --- scalar forms (mix scalar + packed register state) ---
    A(lambda r: f"vaddss {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vmulsd {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vfmadd213ss {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vsqrtsd {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vcvtsi2sd {r.choice(GPR64)}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vmovd {r.choice(GPR32)}, {XMM(zr(r))}")
    A(lambda r: f"vmovq {XMM(zr(r))}, {r.choice(GPR64)}")
    # --- approximations / scalef / getexp ---
    A(lambda r: f"vrcp14ps {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vrsqrt14ps {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vscalefps {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vgetexpps {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vgetmantps $1, {ZMM(zr(r))}, {ZMM(zr(r))}")
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="atort.s")
    ap.add_argument("--min", type=int, default=6)
    ap.add_argument("--max", type=int, default=22)
    args = ap.parse_args()
    rng = random.Random(args.seed * 7919 + 13)
    T = TEMPLATES()
    lines = ['.intel_syntax noprefix' if False else '.text']
    for i in range(args.funcs):
        name = f"atort_{args.seed}_{i}"
        lines.append(f".globl {name}")
        lines.append(f".type {name}, @function")
        lines.append(f"{name}:")
        n = rng.randint(args.min, args.max)
        for _ in range(n):
            lines.append("    " + rng.choice(T)(rng))
        # add a couple of branches so vector/mask state crosses blocks
        if rng.random() < 0.6:
            lab = f".L{args.seed}_{i}"
            lines.append(f"    kortestw {K(kr(rng))}, {K(kr(rng))}")
            lines.append(f"    jz {lab}")
            for _ in range(rng.randint(2, 6)):
                lines.append("    " + rng.choice(T)(rng))
            lines.append(f"{lab}:")
        lines.append("    vzeroupper")
        lines.append("    ret")
        lines.append("")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {args.funcs} asm functions, {len(T)} templates, seed {args.seed}")


if __name__ == "__main__":
    main()
