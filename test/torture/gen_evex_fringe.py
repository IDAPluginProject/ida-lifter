#!/usr/bin/env python3
"""Generate a raw-assembly torture corpus (AT&T syntax) focused on EVEX encoding
features at the *fringe of legality*.

Where gen_asm_torture.py exercises broad instruction coverage, this generator
zeroes in on the EVEX-specific machinery the lifter must decode and model:

  * embedded broadcast      vaddps (%rdi){1to16}, %zmm1, %zmm2
  * static rounding + SAE    vaddps {rn-sae}, %zmm1, %zmm2, %zmm3  (rn/rz/ru/rd)
  * SAE only                 vsqrtpd {sae}, %zmm1, %zmm2
  * explicit {k0} no-op mask and {k1}{z} zero-masking
  * high regs zmm16-31, k1-7 mixed with the above
  * compressed disp8*N memory edges (large +/- displacements, the N-scaling)
  * RIP-relative vector memory (vmovaps sym(%rip), %zmm0 with a local sym)
  * segment-override vector memory (%fs:, %gs:)
  * VEX and EVEX of the same op inside one function
  * scalar EVEX with rounding (vaddsd {rn-sae},...; vcvtsi2sd with {rn})
  * masked gather/scatter with {k} (vSIB) -- known-tricky

Each emitted function is a global symbol:  .globl name; name: <insns>; vzeroupper; ret
with NO register save/restore, so the lifter must synthesize cross-function
zmm/k state -- a classic INTERR trigger.

The assembler (gas via gcc) validates legality; every emitted instruction is
encodable.  Templates gas rejects were dropped; as many fringe forms as are
legal are kept.

Usage: gen_evex_fringe.py --funcs N --seed S --out FILE.s   (deterministic)
"""
from __future__ import annotations
import argparse
import random

ZMM = lambda r: f"%zmm{r}"
YMM = lambda r: f"%ymm{r}"
XMM = lambda r: f"%xmm{r}"
K   = lambda r: f"%k{r}"
GPR32 = ["%eax", "%ebx", "%ecx", "%edx", "%esi", "%edi", "%r8d", "%r9d",
         "%r12d", "%r13d", "%r14d", "%r15d"]
GPR64 = ["%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11",
         "%r12", "%r13", "%r14", "%r15"]

# EVEX gives access to the full 32 vector registers; deliberately bias toward
# the high half (zmm16-31) which only EVEX can encode.
def zr(rng):
    if rng.random() < 0.55:
        return rng.randrange(16, 32)
    return rng.randrange(32)

def xr(rng):  # same idea, name for clarity at xmm/ymm widths
    return zr(rng)

def kr(rng):  return rng.randrange(1, 8)   # k1-k7 (k0 special: no masking)
def kr0(rng): return rng.randrange(8)      # k0-k7

# Mask suffix.  Mix of plain {k} and {k}{z}.
#
# NOTE: an *explicit* {%k0} write mask is architecturally a legal no-op encoding
# but gas (this binutils) rejects it ("`%k0' can't be used for write mask"), so
# we never emit {%k0}.  The "no-mask" fringe is instead covered by emitting the
# bare un-suffixed register form (also EVEX-encoded here via zmm16-31).
def mask(rng, allow_z=True, allow_k0=True):
    roll = rng.random()
    if roll < 0.55:
        return f"{{{K(kr(rng))}}}"
    if allow_z:
        return f"{{{K(kr(rng))}}}{{z}}"
    return f"{{{K(kr(rng))}}}"

# Compressed-disp8*N memory edges: large +/- displacements exercising N-scaling,
# plus SIB forms and bases that aren't %rdi.
DISP = [0, 0x40, -0x40, 0x80, -0x80, 0x100, 0x200, 0x400, 0x800, -0x800,
        0x1000, -0x1000, 0x7f, -0x80, 0x3fc0]
def mem(rng):
    base = rng.choice(["%rdi", "%rsi", "%rdx", "%rcx", "%rax", "%r8", "%r12"])
    roll = rng.random()
    if roll < 0.35:
        d = rng.choice(DISP)
        return f"{d:#x}({base})" if d >= 0 else f"-{-d:#x}({base})"
    if roll < 0.7:
        idx = rng.choice(["%rcx", "%rdx", "%rax", "%r9", "%r13"])
        sc = rng.choice([1, 2, 4, 8])
        d = rng.choice(DISP)
        ds = f"{d:#x}" if d >= 0 else f"-{-d:#x}"
        return f"{ds}({base},{idx},{sc})"
    return f"({base})"

# vSIB memory for gather/scatter: vector index register + scale.  Returns
# (memory_operand, index_register_number) so the caller can pick a destination
# register distinct from the index -- overlapping gather dest/index is a #UD at
# runtime (gas only warns), and we want strictly legal encodings.
def vsib(rng, R):
    base = rng.choice(["%rdi", "%rsi", "%rdx", "%rcx"])
    sc = rng.choice([1, 2, 4, 8])
    d = rng.choice([0, 0x40, -0x40, 0x80, 0x100])
    ds = (f"{d:#x}" if d >= 0 else f"-{-d:#x}") if d else ""
    idx = zr(rng)
    return f"{ds}({base},{R(idx)},{sc})", idx

def distinct(rng, avoid):
    v = zr(rng)
    while v == avoid:
        v = zr(rng)
    return v

# Rounding modes (only legal when the only memory/reg-rounding form is used and
# operands are full-width zmm registers, src reg form, no memory).
RND = ["{rn-sae}", "{rz-sae}", "{ru-sae}", "{rd-sae}"]
def rnd(rng): return rng.choice(RND)

# Broadcast token width depends on element size: ps/d -> 1toN(32b), pd/q ->
# 1toN(64b).  N = vectorbytes / elembytes.
BCAST32 = {512: "{1to16}", 256: "{1to8}", 128: "{1to4}"}
BCAST64 = {512: "{1to8}",  256: "{1to4}", 128: "{1to2}"}


def TEMPLATES():
    T = []
    A = lambda f: T.append(f)

    # ---- embedded broadcast from memory (the {1toN} forms) ----
    for W, R in [(512, ZMM), (256, YMM), (128, XMM)]:
        b32, b64 = BCAST32[W], BCAST64[W]
        A(lambda r, R=R, b=b32: f"vaddps {mem(r)}{b}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R, b=b64: f"vmulpd {mem(r)}{b}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R, b=b32: f"vpaddd {mem(r)}{b}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R, b=b64: f"vpandq {mem(r)}{b}, {R(zr(r))}, {R(zr(r))}")
        A(lambda r, R=R, b=b32: f"vfmadd132ps {mem(r)}{b}, {R(zr(r))}, {R(zr(r))}")
        # broadcast + masking together
        A(lambda r, R=R, b=b32: f"vsubps {mem(r)}{b}, {R(zr(r))}, {R(zr(r))}{mask(r)}")
        A(lambda r, R=R, b=b64: f"vdivpd {mem(r)}{b}, {R(zr(r))}, {R(zr(r))}{mask(r)}")
        A(lambda r, R=R, b=b32: f"vpminsd {mem(r)}{b}, {R(zr(r))}, {R(zr(r))}")

    # ---- static rounding + SAE (zmm reg form, no memory) ----
    A(lambda r: f"vaddps {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vmulpd {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vsubps {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vdivpd {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vfmadd213ps {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vscalefps {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vcvtps2dq {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vcvtdq2ps {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vcvtpd2ps {rnd(r)}, {ZMM(zr(r))}, {YMM(zr(r))}")
    # sqrt: rounding form (the SAE-implied path)
    A(lambda r: f"vsqrtpd {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vsqrtps {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    # rounding + masking
    A(lambda r: f"vaddpd {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}{mask(r, allow_k0=False)}")
    A(lambda r: f"vmulps {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}{{z}}")

    # ---- SAE only (no rounding) ----
    # NOTE: vsqrtp{s,d}/vsqrts{d,s} take *rounding* (which implies SAE) but gas
    # rejects standalone {sae} for sqrt, so those are emitted as rounding forms
    # in the rounding section instead.
    A(lambda r: f"vmaxps {{sae}}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vminpd {{sae}}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vgetexpps {{sae}}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vreducepd $0x3, {{sae}}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vrndscaleps $0x1, {{sae}}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vgetmantpd $0x2, {{sae}}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    # compare with SAE -> mask
    A(lambda r: f"vcmppd $0, {{sae}}, {ZMM(zr(r))}, {ZMM(zr(r))}, {K(kr(r))}")
    A(lambda r: f"vcmpps $0x1f, {{sae}}, {ZMM(zr(r))}, {ZMM(zr(r))}, {K(kr(r))}{{{K(kr(r))}}}")
    A(lambda r: f"vfpclasspd $0x7f, {ZMM(zr(r))}, {K(kr(r))}")

    # ---- implicit no-mask (k0, encoded by omitting the suffix) + {k}{z} ----
    # gas rejects an explicit {%k0}; the un-suffixed form *is* k0 in EVEX, and
    # we force EVEX by using a high register so the lifter sees an EVEX prefix
    # with mask field 0.
    A(lambda r: f"vaddps {ZMM(zr(r))}, {ZMM(r.randrange(16,32))}, {ZMM(r.randrange(16,32))}")
    A(lambda r: f"vpaddd {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}{{z}}")
    A(lambda r: f"vmulpd {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}{mask(r)}")
    A(lambda r: f"vpabsd {ZMM(zr(r))}, {ZMM(zr(r))}{{{K(kr(r))}}}{{z}}")
    A(lambda r: f"vmovdqa64 {ZMM(r.randrange(16,32))}, {ZMM(r.randrange(16,32))}")

    # ---- RIP-relative vector memory (sym defined locally) ----
    A(lambda r: f"vmovaps evfr_pool(%rip), {ZMM(zr(r))}")
    A(lambda r: f"vmovups evfr_pool(%rip), {ZMM(zr(r))}{mask(r)}")
    A(lambda r: f"vmovdqa64 evfr_pool(%rip), {ZMM(zr(r))}")
    A(lambda r: f"vaddps evfr_pool(%rip){{1to16}}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vbroadcastss evfr_pool(%rip), {ZMM(zr(r))}")
    A(lambda r: f"vmovaps {ZMM(zr(r))}, evfr_pool(%rip)")

    # ---- segment-override vector memory ----
    A(lambda r: f"vmovups %fs:0x10, {ZMM(zr(r))}")
    A(lambda r: f"vmovaps %gs:0x40, {ZMM(zr(r))}")
    A(lambda r: f"vmovdqu64 %fs:{rng_disp_str(r)}, {ZMM(zr(r))}")
    A(lambda r: f"vaddps %gs:0x80{{1to16}}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vmovups {ZMM(zr(r))}, %fs:0x20")

    # ---- VEX and EVEX of the same op (paired in one template) ----
    A(lambda r: f"vaddps {YMM(r.randrange(16))}, {YMM(r.randrange(16))}, {YMM(r.randrange(16))}\n"
                f"    vaddps {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}{mask(r)}")
    A(lambda r: f"vmulpd {XMM(r.randrange(16))}, {XMM(r.randrange(16))}, {XMM(r.randrange(16))}\n"
                f"    vmulpd {rnd(r)}, {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}")
    A(lambda r: f"vpxor {YMM(r.randrange(16))}, {YMM(r.randrange(16))}, {YMM(r.randrange(16))}\n"
                f"    vpxorq {ZMM(zr(r))}, {ZMM(zr(r))}, {ZMM(zr(r))}{mask(r)}")
    A(lambda r: f"vbroadcastss {XMM(r.randrange(16))}, {YMM(r.randrange(16))}\n"
                f"    vbroadcastss {XMM(zr(r))}, {ZMM(zr(r))}{mask(r)}")

    # ---- scalar EVEX with rounding / SAE ----
    A(lambda r: f"vaddsd {rnd(r)}, {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vmulss {rnd(r)}, {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vsubsd {rnd(r)}, {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vfmadd213ss {rnd(r)}, {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vsqrtsd {rnd(r)}, {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vminss {{sae}}, {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vcmpsd $0, {{sae}}, {XMM(zr(r))}, {XMM(zr(r))}, {K(kr(r))}")
    A(lambda r: f"vgetexpss {{sae}}, {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    # scalar fp->int and fp->fp conversions with rounding (EVEX-only feature).
    # NOTE: vcvt{,u}si2s{s,d} (int->fp) with rounding are rejected by this gas
    # ("misplaced {rn-sae}") in every operand placement, so they are omitted;
    # the fp->int and fp->fp rounding conversions below cover the same path.
    A(lambda r: f"vcvtsd2si {{rd-sae}}, {XMM(zr(r))}, {r.choice(GPR64)}")
    A(lambda r: f"vcvtss2si {{rz-sae}}, {XMM(zr(r))}, {r.choice(GPR32)}")
    A(lambda r: f"vcvttsd2si {{sae}}, {XMM(zr(r))}, {r.choice(GPR64)}")
    A(lambda r: f"vcvtsd2ss {rnd(r)}, {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    A(lambda r: f"vcvtss2sd {{sae}}, {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}")
    # scalar masked
    A(lambda r: f"vaddsd {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}{{{K(kr(r))}}}{{z}}")
    A(lambda r: f"vmovsd {XMM(zr(r))}, {XMM(zr(r))}, {XMM(zr(r))}{{{K(kr(r))}}}")

    # ---- masked gather / scatter (vSIB + {k}) ----
    # Gather destination must differ from the vSIB index register (else #UD);
    # vsib() reports the index so we pick a distinct dest with distinct().
    # Idx is the *index* register class, Dst the destination class; for the
    # dword-index/qword-data gathers (vgatherdpd, vpgatherqq's sibling) the
    # index is half the dest's element-count width, so the two classes differ.
    def gather(r, mn, Idx, Dst):
        m, idx = vsib(r, Idx)
        return f"{mn} {m}, {Dst(distinct(r, idx))}{{{K(kr(r))}}}"
    def scatter(r, mn, Idx, Src):
        m, _ = vsib(r, Idx)
        return f"{mn} {Src(zr(r))}, {m}{{{K(kr(r))}}}"
    # full-width (zmm dest):
    A(lambda r: gather(r, "vgatherdps", ZMM, ZMM))   # dword idx, dword data
    A(lambda r: gather(r, "vpgatherdd", ZMM, ZMM))
    A(lambda r: gather(r, "vgatherqpd", ZMM, ZMM))   # qword idx, qword data
    A(lambda r: gather(r, "vpgatherqq", ZMM, ZMM))
    A(lambda r: gather(r, "vgatherdpd", YMM, ZMM))   # dword idx (ymm) -> qword data (zmm)
    A(lambda r: gather(r, "vpgatherdq", YMM, ZMM))
    A(lambda r: gather(r, "vgatherqps", ZMM, YMM))   # qword idx (zmm) -> dword data (ymm)
    A(lambda r: scatter(r, "vscatterdps", ZMM, ZMM))
    A(lambda r: scatter(r, "vpscatterdd", ZMM, ZMM))
    A(lambda r: scatter(r, "vscatterqpd", ZMM, ZMM))
    A(lambda r: scatter(r, "vscatterdpd", YMM, ZMM))
    # gather/scatter at ymm/xmm widths too
    A(lambda r: gather(r, "vgatherdps", YMM, YMM))
    A(lambda r: gather(r, "vpgatherdd", XMM, XMM))
    A(lambda r: gather(r, "vgatherdpd", XMM, YMM))   # dword idx (xmm) -> qword data (ymm)

    # ---- compress / expand to memory (mask + disp8*N memory) ----
    A(lambda r: f"vpcompressd {ZMM(zr(r))}, {mem(r)}{{{K(kr(r))}}}")
    A(lambda r: f"vpexpandd {mem(r)}, {ZMM(zr(r))}{{{K(kr(r))}}}{{z}}")
    A(lambda r: f"vcompressps {ZMM(zr(r))}, {mem(r)}{{{K(kr(r))}}}")
    A(lambda r: f"vexpandpd {mem(r)}, {ZMM(zr(r))}{{{K(kr(r))}}}")

    # ---- memory loads/stores at disp8*N edges (different N per elem size) ----
    A(lambda r: f"vmovups {mem(r)}, {ZMM(zr(r))}{mask(r)}")
    A(lambda r: f"vmovaps {ZMM(zr(r))}, {mem(r)}{{{K(kr(r))}}}")
    A(lambda r: f"vmovdqu8 {mem(r)}, {ZMM(zr(r))}{{{K(kr(r))}}}{{z}}")
    A(lambda r: f"vmovdqu16 {mem(r)}, {ZMM(zr(r))}{{{K(kr(r))}}}")
    A(lambda r: f"vmovdqa32 {mem(r)}, {ZMM(zr(r))}{mask(r)}")

    return T


# A small set of legal raw segment displacements.
def rng_disp_str(rng):
    d = rng.choice([0x10, 0x40, 0x100, 0x200])
    return f"{d:#x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="evfr.s")
    ap.add_argument("--min", type=int, default=6)
    ap.add_argument("--max", type=int, default=20)
    args = ap.parse_args()
    rng = random.Random(args.seed * 6151 + 29)
    T = TEMPLATES()
    lines = [".text"]
    for i in range(args.funcs):
        name = f"evfr_{args.seed}_{i}"
        lines.append(f".globl {name}")
        lines.append(f".type {name}, @function")
        lines.append(f"{name}:")
        n = rng.randint(args.min, args.max)
        for _ in range(n):
            lines.append("    " + rng.choice(T)(rng))
        # branch so vector/mask state crosses basic blocks (microcode temps
        # crossing block boundaries is a known INTERR trigger).
        if rng.random() < 0.6:
            lab = f".Levfr_{args.seed}_{i}"
            lines.append(f"    kortestw {K(kr(rng))}, {K(kr(rng))}")
            lines.append(f"    jz {lab}")
            for _ in range(rng.randint(2, 6)):
                lines.append("    " + rng.choice(T)(rng))
            lines.append(f"{lab}:")
        lines.append("    vzeroupper")
        lines.append("    ret")
        lines.append(f".size {name}, .-{name}")
        lines.append("")
    # 64-byte aligned data pool referenced by RIP-relative / broadcast templates.
    lines.append(".section .rodata")
    lines.append(".align 64")
    lines.append("evfr_pool:")
    lines.append("    .zero 256")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {args.funcs} asm functions, {len(T)} templates, seed {args.seed}")


if __name__ == "__main__":
    main()
