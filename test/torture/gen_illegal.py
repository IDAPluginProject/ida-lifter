#!/usr/bin/env python3
"""Generate a *pathological* raw-assembly torture corpus (AT&T syntax) that
pushes the lifter BEYOND what a sane compiler would ever emit, while staying
DECODABLE by IDA. Two complementary ingredients:

(1) RAW .byte EVEX/VEX sequences the assembler refuses to emit but IDA still
    decodes. We start from a known-good EVEX encoding (62 P0 P1 P2 op modrm ...)
    and flip *fringe* fields that IDA tolerates:
      * apply a k-mask {k} to instructions whose semantics ignore it,
      * set the EVEX broadcast/RC bit b (P2 bit4) on register-form ops — IDA
        re-reads it as embedded-rounding {er}/{sae}, which is architecturally
        nonsensical on integer/move/ternlog ops,
      * set the zeroing bit z (P2 bit7) with no mask (aaa=0) — "{z} without k",
      * set the zeroing bit z on a mask-destination compare (writemask dest
        cannot zero-merge),
      * exercise high vvvv via V' on 2-operand ops that ignore vvvv.
    Each raw blob is a *real-ish* EVEX prefix + opcode + modrm; every blob in the
    library has been verified to disassemble to a real mnemonic (not "(bad)") and
    to decode under IDA. We avoid the truly-undecodable mutations (e.g. L'L=11
    reserved length, clearing the mandatory P1 bit2) which produce "(bad)".

(2) PATHOLOGICAL CONTROL FLOW & LAYOUT around normal vector ops:
      * jumps INTO the middle of a multi-byte EVEX instruction (overlapping code
        — IDA re-decodes the same bytes two different ways),
      * very large straight-line bodies (hundreds of vector insns) to stress the
        microcode optimizer/verifier,
      * deeply nested branches/loops with vector + mask state live across blocks,
      * jump tables / computed jumps among vector blocks,
      * vector ops immediately around call/ret and at function boundaries,
      * misaligned function entries / padding made of vector instruction bytes.

Every function still ASSEMBLES: the raw encodings go through .byte, the CFG
scaffolding uses normal mnemonics. Output is deterministic for a given --seed.

Usage: gen_illegal.py --funcs 40 --seed 1 --out illtort.s
"""
from __future__ import annotations
import argparse
import random

# ---------------------------------------------------------------------------
# Register helpers (mirror gen_asm_torture.py so the corpora look related).
# ---------------------------------------------------------------------------
ZMM = lambda r: f"%zmm{r}"
YMM = lambda r: f"%ymm{r}"
XMM = lambda r: f"%xmm{r}"
K   = lambda r: f"%k{r}"
GPR64 = ["%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9", "%r10", "%r11"]


def zr(rng): return rng.randrange(32)
def kr(rng): return rng.randrange(1, 8)
def mem(rng): return rng.choice(["(%rdi)", "(%rsi)", "(%rdx)", "0x40(%rdi)"])


# ===========================================================================
# (1) RAW EVEX/VEX BYTE LIBRARY
# ===========================================================================
# EVEX is: 62  P0  P1  P2  opcode  modrm  [sib] [disp] [imm]
#   P0 = R  X  B  R' 0 0 m m       (mm = map; bit2 reserved/0)
#   P1 = W  v3 v2 v1 v0 1 p p      (bit2 mandatory 1; vvvv inverted)
#   P2 = z  L' L  b  V' a a a      (z=zeroing, L'L=length, b=bcst/RC,
#                                    V'=high-vvvv inverted, aaa=mask)
# Each entry is (label, comment, [bytes]). The label tags what fringe field we
# perturbed; the comment is what IDA/objdump renders (verified separately).
# All blobs use length L'L=10 (512-bit / zmm) unless noted, and have been
# checked to disassemble to a real mnemonic, never "(bad)".

def _evex(P0, P1, P2, op, modrm, *tail):
    return [0x62, P0, P1, P2, op, modrm, *tail]


# Base (sane) reference encodings we mutate, for documentation:
#   vaddps  %zmm1,%zmm2,%zmm3        = 62 f1 6c 48 58 d9
#   vpaddd  %zmm1,%zmm2,%zmm3        = 62 f1 6d 48 fe d9
#   vmovaps %zmm5,%zmm6              = 62 f1 7c 48 28 f5
#   vpternlogd $0x96,z1,z2,z3        = 62 f3 6d 48 25 d9 96
#   vmulps  %zmm4,%zmm5,%zmm6        = 62 f1 54 48 59 f4
#   vsubps  %zmm4,%zmm5,%zmm6        = 62 f1 54 48 5c f4
#   vsqrtps %zmm1,%zmm2              = 62 f1 7c 48 51 d1
#   vcmpeqps z1,z2,k1                = 62 f1 6c 48 c2 c9 00
#   vmovdqu64 (%rdi),%zmm0           = 62 f1 fe 48 6f 07

# P2 bit layout we will toggle:
B_BCST = 0x10   # broadcast / RC bit (b)
B_Z    = 0x80   # zeroing bit (z)
V_PRIME= 0x08   # V' (high vvvv, inverted)

# A curated list of (name, comment, bytes). Comments describe the *intended*
# pathology; the actual IDA rendering is reported by self-test.
RAW_LIB = [
    # --- k-mask applied to ops where it is semantically meaningless ---------
    ("kmask_on_addps_k3",
     "vaddps with k3 mask (sane) -> still legal but we feed odd aaa",
     _evex(0xf1, 0x6c, 0x48 | 0x03, 0x58, 0xd9)),           # {k3}
    ("kmask_on_movaps",
     "vmovaps reg-reg with a write-mask {k5} applied to a plain move",
     _evex(0xf1, 0x7c, 0x48 | 0x05, 0x28, 0xf5)),           # {k5}

    # --- broadcast/RC bit set on REGISTER-form ops (no memory operand) -------
    # IDA re-reads b as embedded rounding {er}/{sae}; nonsensical on these ops.
    ("bcst_on_movaps_reg",
     "vmovaps reg-reg with EVEX.b set (rounding ctrl on a move)",
     _evex(0xf1, 0x7c, 0x48 | B_BCST, 0x28, 0xf5)),
    ("bcst_on_paddd_reg",
     "vpaddd reg-reg with EVEX.b set (rounding ctrl on integer add)",
     _evex(0xf1, 0x6d, 0x48 | B_BCST, 0xfe, 0xd9)),
    ("bcst_on_ternlogd_reg",
     "vpternlogd reg form with EVEX.b set (rounding ctrl on a logic op)",
     _evex(0xf3, 0x6d, 0x48 | B_BCST, 0x25, 0xd9, 0x96)),
    ("bcst_on_addps_reg_rc",
     "vaddps reg-reg with EVEX.b (embedded rounding) but RC bits=01",
     _evex(0xf1, 0x6c, 0x48 | B_BCST, 0x58, 0xd9)),

    # --- zeroing bit z set with no mask (aaa=0): {z} without {k} -------------
    ("z_no_mask_addps",
     "vaddps {z} with aaa=0 (zeroing requested but no write-mask)",
     _evex(0xf1, 0x6c, 0x48 | B_Z, 0x58, 0xd9)),
    ("z_no_mask_paddd",
     "vpaddd {z} with aaa=0 (zeroing requested but no write-mask)",
     _evex(0xf1, 0x6d, 0x48 | B_Z, 0xfe, 0xd9)),

    # --- z bit set on a mask-destination compare (illegal: k-dest can't zero) -
    ("z_on_cmpps_kdest",
     "vcmpeqps -> k1 with EVEX.z set (zeroing a mask destination)",
     _evex(0xf1, 0x6c, 0x48 | B_Z, 0xc2, 0xc9, 0x00)),
    ("z_on_cmpd_kdest",
     "vpcmpeqd -> k1 with EVEX.z set on the mask destination",
     _evex(0xf1, 0x7d, 0x48 | B_Z, 0x76, 0xc9)),

    # --- V' / high-vvvv perturbation on 2-operand ops that ignore vvvv -------
    ("vprime_on_sqrtps",
     "vsqrtps with V' cleared (forces a high non-1111 vvvv on a 2-op insn)",
     _evex(0xf1, 0x7c, 0x40, 0x51, 0xd1)),                  # V'=0, vvvv=1111
    ("vprime_on_rcp14ps",
     "vrcp14ps with V' cleared (high vvvv source on a unary op)",
     _evex(0xf2, 0x7d, 0x40, 0x4c, 0xd1)),

    # --- broadcast bit on a memory-LESS movdqu / load-form opcode ------------
    ("bcst_on_movdqu64_reg",
     "vmovdqu64 reg-reg with EVEX.b set",
     _evex(0xf1, 0xfe, 0x48 | B_BCST, 0x6f, 0xd9)),

    # --- mask + broadcast bit together on a register-form op -----------------
    ("kmask_and_bcst_addps",
     "vaddps reg-reg with BOTH a write-mask {k2} and EVEX.b (rounding) set",
     _evex(0xf1, 0x6c, 0x48 | B_BCST | 0x02, 0x58, 0xd9)),

    # --- high EVEX registers (zmm16-31) combined with fringe bits ------------
    ("hi_reg_bcst_addps",
     "vaddps %zmm17,%zmm18,%zmm19 with EVEX.b set (high regs + rounding ctrl)",
     _evex(0xa1, 0x6c, 0x40 | B_BCST, 0x58, 0xd9)),
    ("hi_reg_z_no_mask",
     "vpaddd %zmm17,%zmm18,%zmm19 with {z}/aaa=0",
     _evex(0xa1, 0x6d, 0x40 | B_Z, 0xfe, 0xd9)),
]


def emit_raw(rng, lines, n):
    """Emit n random raw .byte blobs from the library."""
    for _ in range(n):
        name, comment, bs = rng.choice(RAW_LIB)
        lines.append(f"    # raw[{name}]: {comment}")
        lines.append("    .byte " + ",".join(f"0x{b:02x}" for b in bs))


# ===========================================================================
# (1b) NORMAL vector ops (for the CFG scaffolding) — assembler-legal.
# ===========================================================================
def vec_normal(rng):
    R = rng.choice([ZMM, YMM, XMM])
    a, b, c = zr(rng), zr(rng), zr(rng)
    return rng.choice([
        f"vaddps {R(a)}, {R(b)}, {R(c)}",
        f"vmulpd {R(a)}, {R(b)}, {R(c)}",
        f"vpaddd {R(a)}, {R(b)}, {R(c)}",
        f"vpxorq {R(a)}, {R(b)}, {R(c)}",
        f"vfmadd213ps {R(a)}, {R(b)}, {R(c)}",
        f"vpternlogd $0x96, {R(a)}, {R(b)}, {R(c)}",
        f"vsqrtpd {R(a)}, {R(b)}",
        f"vmovups {mem(rng)}, {R(c)}",
        f"vpermps {ZMM(a)}, {ZMM(b)}, {ZMM(c)}",
        f"vcvtdq2ps {ZMM(a)}, {ZMM(b)}",
    ])


def vec_masked(rng):
    a, b, c, k = zr(rng), zr(rng), zr(rng), kr(rng)
    return rng.choice([
        f"vaddps {ZMM(a)}, {ZMM(b)}, {ZMM(c)}{{{K(k)}}}",
        f"vpaddd {ZMM(a)}, {ZMM(b)}, {ZMM(c)}{{{K(k)}}}{{z}}",
        f"vcmpps $0, {ZMM(a)}, {ZMM(b)}, {K(k)}",
        f"vptestmd {ZMM(a)}, {ZMM(b)}, {K(k)}",
        f"vmovdqu64 {mem(rng)}, {ZMM(c)}{{{K(k)}}}{{z}}",
    ])


def mask_op(rng):
    k = kr(rng)
    return rng.choice([
        f"kandw {K(k)}, {K(kr(rng))}, {K(kr(rng))}",
        f"korw {K(k)}, {K(kr(rng))}, {K(kr(rng))}",
        f"kxorq {K(k)}, {K(kr(rng))}, {K(kr(rng))}",
        f"knotw {K(k)}, {K(kr(rng))}",
        f"kshiftlw $3, {K(k)}, {K(kr(rng))}",
    ])


# ===========================================================================
# (2) PATHOLOGICAL FUNCTION SHAPES
# ===========================================================================
# Each shape is shape(rng, name, seed, idx) -> list[str] of asm lines for one
# function (label + body + epilogue). Mixed with raw blobs throughout.

def epilogue(lines):
    lines.append("    vzeroupper")
    lines.append("    ret")


def func_header(lines, name):
    lines.append(f".globl {name}")
    lines.append(f".type {name}, @function")
    lines.append(f"{name}:")


def shape_overlap(rng, name, seed, idx):
    """Jump INTO the middle of a multi-byte EVEX instruction: IDA re-decodes
    the same bytes two different ways (overlapping code)."""
    lines = []
    func_header(lines, name)
    Llab = f".Lov_{seed}_{idx}"
    Alab = f".Lblob_{seed}_{idx}"
    lines.append("    xorl %eax, %eax")
    lines.append("    testq %rdi, %rdi")
    lines.append(f"    jne {Llab}")
    # primary straight-line path with the long blob
    for _ in range(rng.randint(2, 5)):
        lines.append("    " + vec_normal(rng))
    lines.append(f"    # 7-byte vpternlogd blob; alternate entry jumps to interior")
    lines.append(f"{Alab}:")
    lines.append("    .byte 0x62,0xf3,0x6d,0x48,0x25,0xd9,0x96")
    for _ in range(rng.randint(1, 3)):
        lines.append("    " + vec_normal(rng))
    epilogue(lines)
    # alternate entry: jump to an interior byte (+offset) of the blob
    off = rng.choice([2, 3, 4])
    lines.append(f"{Llab}:")
    for _ in range(rng.randint(1, 3)):
        lines.append("    " + vec_normal(rng))
    lines.append(f"    jmp {Alab}+{off}")
    lines.append("")
    return lines


def shape_huge(rng, name, seed, idx):
    """Very large straight-line body: hundreds of vector insns + raw blobs to
    stress the microcode optimizer/verifier."""
    lines = []
    func_header(lines, name)
    total = rng.randint(220, 360)
    for i in range(total):
        roll = rng.random()
        if roll < 0.18:
            emit_raw(rng, lines, 1)
        elif roll < 0.40:
            lines.append("    " + vec_masked(rng))
        elif roll < 0.55:
            lines.append("    " + mask_op(rng))
        else:
            lines.append("    " + vec_normal(rng))
    epilogue(lines)
    lines.append("")
    return lines


def shape_nested(rng, name, seed, idx):
    """Deeply nested branches/loops keeping vector + mask values live across
    many blocks (classic 'temporaries cross block boundaries' trigger)."""
    lines = []
    func_header(lines, name)
    depth = rng.randint(4, 7)
    lines.append("    movq %rdi, %rcx")
    # establish some live vector + mask state before the nest
    for _ in range(rng.randint(2, 4)):
        lines.append("    " + vec_normal(rng))
    lines.append("    " + mask_op(rng))
    labs = [f".Lnz_{seed}_{idx}_{d}" for d in range(depth)]
    for d in range(depth):
        lines.append(f"    subq $1, %rcx")
        lines.append(f"    kortestw {K(kr(rng))}, {K(kr(rng))}")
        lines.append(f"    jz {labs[d]}")
        lines.append("    " + vec_masked(rng))
        if rng.random() < 0.5:
            emit_raw(rng, lines, 1)
    # innermost body
    for _ in range(rng.randint(3, 6)):
        lines.append("    " + vec_normal(rng))
    # close the nest in reverse, each block touching live vector state
    for d in reversed(range(depth)):
        lines.append(f"{labs[d]}:")
        lines.append("    " + vec_masked(rng))
        # backward branch -> loop with vector state live across the back-edge
        if rng.random() < 0.4:
            lines.append(f"    cmpq $0, %rcx")
            lines.append(f"    jg {labs[d]}")
    epilogue(lines)
    lines.append("")
    return lines


def shape_jumptable(rng, name, seed, idx):
    """Computed jump / jump table dispatching among vector blocks."""
    lines = []
    func_header(lines, name)
    ncase = rng.randint(3, 6)
    tbl = f".Ltbl_{seed}_{idx}"
    cases = [f".Lc_{seed}_{idx}_{j}" for j in range(ncase)]
    done = f".Ldone_{seed}_{idx}"
    lines.append("    " + vec_normal(rng))
    lines.append(f"    andl ${ncase - 1}, %edi")
    lines.append(f"    leaq {tbl}(%rip), %rax")
    lines.append("    movslq (%rax,%rdi,4), %rcx")
    lines.append("    addq %rax, %rcx")
    lines.append("    jmp *%rcx")
    lines.append("    .balign 4")
    lines.append(f"{tbl}:")
    for j in range(ncase):
        lines.append(f"    .long {cases[j]} - {tbl}")
    for j in range(ncase):
        lines.append(f"{cases[j]}:")
        for _ in range(rng.randint(2, 5)):
            lines.append("    " + vec_normal(rng))
        if rng.random() < 0.5:
            emit_raw(rng, lines, 1)
        lines.append(f"    jmp {done}")
    lines.append(f"{done}:")
    lines.append("    " + vec_masked(rng))
    epilogue(lines)
    lines.append("")
    return lines


def shape_callret(rng, name, seed, idx, helper):
    """Vector ops immediately around call/ret and at function boundaries:
    a vector op as the very first and very last instruction, with a call to a
    helper sandwiched between live vector state."""
    lines = []
    func_header(lines, name)
    # vector op at the very entry (no prologue)
    lines.append("    " + vec_normal(rng))
    lines.append("    " + vec_masked(rng))
    emit_raw(rng, lines, 1)
    # preserve nothing; call the helper with live zmm state straddling the call
    lines.append(f"    call {helper}")
    lines.append("    " + vec_normal(rng))
    emit_raw(rng, lines, 1)
    lines.append("    vzeroupper")
    # raw blob immediately before ret, then a bare ret (vector op at boundary)
    lines.append("    " + vec_normal(rng))
    lines.append("    ret")
    lines.append("")
    return lines


def shape_misaligned(rng, name, seed, idx):
    """Misaligned function entry / padding made of vector instruction bytes:
    pad with raw EVEX bytes between the global label and a secondary entry, so
    the function 'starts' inside a vector byte stream."""
    lines = []
    func_header(lines, name)
    # padding: a couple of raw blobs that look like vector bytes but precede the
    # 'real' work; a second global entry points past them.
    emit_raw(rng, lines, rng.randint(1, 2))
    inner = f"{name}_mid"
    lines.append(f".globl {inner}")
    lines.append(f".type {inner}, @function")
    lines.append(f"{inner}:")
    for _ in range(rng.randint(3, 6)):
        lines.append("    " + vec_normal(rng))
    emit_raw(rng, lines, 1)
    lines.append("    " + vec_masked(rng))
    epilogue(lines)
    lines.append("")
    return lines


def shape_rawmix(rng, name, seed, idx):
    """A medium body that is mostly raw blobs interleaved with masked vector
    ops — maximal exposure of the fringe encodings to the lifter."""
    lines = []
    func_header(lines, name)
    n = rng.randint(30, 70)
    for i in range(n):
        if rng.random() < 0.55:
            emit_raw(rng, lines, 1)
        elif rng.random() < 0.5:
            lines.append("    " + vec_masked(rng))
        else:
            lines.append("    " + vec_normal(rng))
    # a small branch so the fringe state crosses a block boundary
    lab = f".Lrx_{seed}_{idx}"
    lines.append(f"    kortestw {K(kr(rng))}, {K(kr(rng))}")
    lines.append(f"    jz {lab}")
    emit_raw(rng, lines, rng.randint(2, 4))
    lines.append(f"{lab}:")
    lines.append("    " + vec_normal(rng))
    epilogue(lines)
    lines.append("")
    return lines


SHAPES = [
    shape_overlap,
    shape_huge,
    shape_nested,
    shape_jumptable,
    shape_misaligned,
    shape_rawmix,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="illtort.s")
    args = ap.parse_args()

    rng = random.Random(args.seed * 1000003 + 911)
    lines = [".text"]

    # One shared helper used by the call/ret shape (defined first so it exists).
    helper = f"illhelp_{args.seed}"
    func_header(lines, helper)
    lines.append("    " + vec_normal(rng))
    lines.append("    ret")
    lines.append("")

    n_callret = 0
    for i in range(args.funcs):
        name = f"ill_{args.seed}_{i}"
        # deterministically rotate through shapes, plus the call/ret shape.
        pick = i % (len(SHAPES) + 1)
        if pick == len(SHAPES):
            lines += shape_callret(rng, name, args.seed, i, helper)
            n_callret += 1
        else:
            lines += SHAPES[pick](rng, name, args.seed, i)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {args.funcs} pathological asm functions "
          f"(+1 helper), {len(RAW_LIB)} raw EVEX blobs, "
          f"{len(SHAPES) + 1} shapes, seed {args.seed}")


if __name__ == "__main__":
    main()
