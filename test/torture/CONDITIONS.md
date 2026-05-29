# CONDITIONS.md — Wild-Variation Audit for the AVX Lifter Torture Suite

Completeness spec / audit checklist for the AVX(-512/10) lifter torture suite.
It enumerates how AVX-bearing binaries vary **in the wild**, organized by axis.
For each value:

- **(a) microcode / INTERR risk** — why this could change the microcode the
  lifter (Hex-Rays `microcode_filter_t` / `codegen_t` plugin) sees, or trip a
  decompiler internal error.
- **(b) coverage** — `cover` / `partial` / `TODO` for the *current* suite
  (`gen_torture.py` intrinsic-C corpus + `gen_asm_torture.py` raw-AT&T corpus,
  built per `flags.txt`, dumped with `idump --plugin lifter --pseudo-only`).

The lifter is built/installed separately (`$HOME/.idapro/plugins/lifter.so`); we
only generate **binaries** and scan `idump` output for `INTERR`,
`Success rate: <100`, and `__asm { mnemonic }` (unlifted).

Definitions used below:
- **cover** — an existing generator deterministically emits this and at least one
  built sample exercises it through `idump`.
- **partial** — emitted only incidentally / for a subset (e.g. only zmm0-15, only
  one width, only via compiler discretion), or buildable but not wired into the
  default sweep.
- **TODO** — not produced by any current generator; a known blind spot.

Grounding for the coverage calls below was established by building and dumping
samples (seed 99): the ELF64 C + ELF64 asm corpora build clean and decompile at
100% with 0 INTERR / 0 `__asm`; a mingw `x86_64-w64-mingw32-gcc-13` PE32+ DLL
and an `i686-w64-mingw32-gcc` PE32 DLL build clean from the *same* C source with
the full `flags.txt`; `idump` loads the PE32+ DLL natively. The only `__asm` seen
across targets was a mingw CRT `fninit` stub (x87, not an AVX path) — i.e. the
cross-target path *does* pull in different runtime/codegen idioms.

---

## Axis 1 — Compiler & version (codegen idioms)

The lifter consumes IDA's decoded `insn_t`, so the **encoding** matters more than
the source compiler — but each compiler/version picks different *instruction
selection*, *register allocation*, *spill idioms*, *prologue/epilogue shapes*,
and *which intrinsic lowers to which mnemonic*. Those drive which handlers fire
and in what dataflow context.

| Value | (a) risk | (b) coverage |
|---|---|---|
| **gcc** (15.x here; also 11/12/13 era idioms) | Reference idioms; tends to keep intrinsics 1:1, uses `vmovdqa64`/`vmovaps` spills, `vzeroupper` at returns. Our only build compiler today. | **cover** (C corpus is gcc -O1) |
| **gcc, older majors (8–13)** | Different default tuning, different AVX-512 lowering (e.g. `-mprefer-vector-width`), older `vpternlog` usage. Can emit mnemonic variants/operand orders the handlers were not tuned for. | **TODO** (single gcc on box; no multi-version sweep) |
| **clang/LLVM** (23 here; also targets x86 & win) | Aggressive vectorizer; different shuffle lowering (`vpermil*`, `vpshufb` chains), `vpblendm*` vs `vpternlog`, masked-select idioms, scalarized gather fallback, distinct spill scheduling. New microcode shapes for the same C. | **partial** (clang installed + verified to build, but not wired into a generator/sweep) |
| **MSVC `cl`** | Not on box, but PE corpora can stand in for the idioms: `vmovups [rsp+..]` callee-save of xmm6-15, SEH-driven prologues, `__chkstk`, different intrinsic→mnemonic map (e.g. `_mm512_*` lowering), no `vzeroupper` discipline in some paths. | **TODO** (no MSVC; mingw PE only approximates ABI, not codegen) |
| **Intel `icx` (LLVM-based) / `icc` (classic)** | Heaviest AVX-512 user in the wild: embedded broadcast `{1toN}`, static rounding `{er}`/`{sae}`, `vrcp14`/`vrsqrt14`+`vfmadd` Newton sequences, `vscalef`, mask-heavy loops, `vpcompress`/`vpexpand`, software-pipelined kernels with many live zmm. Most likely to hit unimplemented EVEX features. | **TODO** (not installed; the *features* it emits are partially covered by the asm corpus, but not the icc idiom shapes) |
| **Hand-written asm / inline asm / ISPC / JITs (libxsmm, oneDNN, numpy, ffmpeg)** | Arbitrary legal sequences, unusual orderings, registers read-before-write across calls. | **partial** (the raw-asm corpus is exactly this style — no save/restore, cross-fn reads) |

---

## Axis 2 — Optimization level & flags

| Value | (a) risk | (b) coverage |
|---|---|---|
| **-O0** | Every temp spilled/reloaded; vectors live in memory between ops; lots of `vmovaps [rsp+..]` load/store pairs; address-taken locals. Stresses memory-operand handlers + stack-slot vector typing. | **TODO** (C corpus fixed at -O1) |
| **-O1** | Moderate register pressure, some spills, branches preserved. | **cover** (default) |
| **-O2** | Heavier inlining, CSE merges vector temps, branch folding, more registers live across blocks (the classic "temporary crosses block boundary" INTERR trigger). | **TODO** |
| **-O3** | Full vectorizer: loop bodies become wide AVX-512, gather/scatter, masked remainders, unrolled+interleaved live ranges, `vpternlog` fusion. Highest novel-microcode yield. | **partial** (asm corpus injects gather/scatter/mask remainders by hand; not real -O3 loop shapes) |
| **-Os / -Oz** | Size-tuned selection prefers shorter encodings; may pick VEX over EVEX where legal, fold broadcasts, reuse registers tightly. | **TODO** |
| **-Ofast / -ffast-math** | Reassociation, reciprocal approximations (`vrcp14`/`vrsqrt14` + FMA Newton), `vfmadd` fusion, no-NaN assumptions changing compare/min-max lowering. | **partial** (rcp14/rsqrt14/scalef/getexp in both corpora; not from real fast-math reassociation) |
| **-funroll-loops** | Many simultaneously-live zmm; deep SSA webs; very long basic blocks → microcode list size / live-range stress. | **partial** (asm functions are long & high-pressure, but flat, not unrolled loops) |
| **-flto / -flto=thin** | Cross-module inlining, merged vector temps across TU boundaries, identical-code folding (ICF) → shared/aliased function bodies, odd symbolization. | **TODO** |
| **PGO (`-fprofile-use`) / AutoFDO** | Hot/cold splitting (`.text.hot`/`.text.unlikely`), block reordering, cold-path outlining, partial inlining → fragmented/jumped-into bodies. | **TODO** |
| **-fno-omit-frame-pointer vs -fomit-frame-pointer** | Frame-pointer present changes stack-slot addressing (`[rbp-..]` vs `[rsp+..]`) for spilled vectors; affects how the lifter types/tracks memory vector operands. | **partial** (gcc -O1 omits FP on x86-64 by default; FP-present case TODO) |
| **-mprefer-vector-width=128/256/512** | Forces width selection; 256-pref code still uses zmm for some ops → mixed-width live state. | **TODO** |
| **-fcf-protection (CET) / endbr64** | `endbr64` at entries, notrack jumps — affects CFG recovery, not AVX directly, but changes block boundaries. | **TODO** |
| **-fstack-protector / canaries** | Extra GPR traffic and a tail compare; benign for AVX but changes epilogue shape. | **TODO** |

---

## Axis 3 — Output format / OS

The lifter operates post-decode, so format mostly affects **what IDA loads and
symbolizes** (function discovery, ABI inference, name demangling) and which CRT
idioms appear — all of which change the function *set* and surrounding code.

| Value | (a) risk | (b) coverage |
|---|---|---|
| **ELF64 (x86-64) shared object** | Baseline; exported funcs recognized; SysV ABI inferred. | **cover** (both corpora) |
| **ELF64 object (.o, unlinked)** | No PLT/GOT, relocations unresolved; IDA may type funcs differently. `gcc -c` path. | **TODO** |
| **ELF64 executable (PIE/-no-pie)** | Different entry/CRT, GOT-relative loads of vector globals (RIP-rel). | **partial** (globals→RIP-rel loads appear in -fPIC .so) |
| **ELF32 (i386) .so/.o** | No xmm8-15/zmm8-31, 8 GPRs, different ABI; AVX-512 still encodable but register-starved → heavy spilling. mingw i686 builds confirmed to compile the C corpus. | **partial** (i686-w64-mingw32 builds verified; not wired into sweep) |
| **PE32+ (x86-64 Windows) DLL/EXE** | MS x64 ABI, SEH/`.pdata` unwind, xmm6-15 callee-save spills, `__chkstk`. idump loads natively (verified). Different CRT funcs surface (`fpreset`/`fninit`). | **partial** (mingw PE32+ build + idump verified; not in default sweep) |
| **PE32 (x86 Windows)** | i686 ABI variants (cdecl/stdcall/fastcall), SEH (FS:[0]) frames. Confirmed buildable via i686-w64-mingw32. | **TODO** (buildable, not generated/swept) |
| **Mach-O (x86_64, macOS)** | SysV-like ABI but distinct symbol/section layout, `__stubs`, no cross-compiler on box. | **TODO** (no Mach-O toolchain here) |
| **Stripped vs symbols** | Stripped → IDA invents `sub_*` names, may merge/miss functions, infer types from scratch → different decompiler input. | **TODO** (corpora always export named symbols) |
| **PIC/PIE vs non-PIC** | RIP-relative vs absolute global addressing for vector globals/constant pools (`vbroadcastss .LCPI(%rip)`); changes memory-operand microcode. | **partial** (-fPIC default; non-PIC + absolute-addr TODO) |
| **Debug info (DWARF/CodeView)** | Better typing of vector locals; can change Hex-Rays var modeling. | **TODO** |

---

## Axis 4 — Calling convention / ABI

Critical because it determines **which vector registers are passed in, returned
in, and callee-saved** — i.e. which regs are *live on entry without a defining
instruction* (forcing the lifter to synthesize `__readzmm`/`__readmask`) and
which are spilled around calls.

| Value | (a) risk | (b) coverage |
|---|---|---|
| **System V AMD64** | xmm0-7 args, xmm0/1 return, **no callee-saved vector regs**, all caller-saved. zmm/k clobbered freely across calls. Our baseline. | **cover** |
| **MS x64** | xmm0-3 args (by position, shared with GPR slots), xmm6-15 **callee-saved** (spilled in prologue, restored in epilogue), zmm6-15 low-128 saved / upper volatile, k0-7 volatile. Spill/restore of vectors → memory-vector typing + cross-block live ranges. | **partial** (mingw PE approximates; no generator targets the save/restore idiom directly) |
| **__vectorcall** | Up to 6 vector args in xmm/ymm/zmm0-5, HVA (homogeneous vector aggregate) args/returns passed in multiple vector regs, returns in xmm0-3. Many vectors live on entry. High INTERR potential. | **TODO** |
| **__regcall (Intel)** | Extends vector arg/return registers further (xmm0-15 for args in some forms), aggressive register passing → maximal cross-fn live vector state. | **TODO** |
| **32-bit cdecl / stdcall** | Vectors passed on **stack** (no xmm arg regs in classic 32-bit), returned in xmm0 / ST(0); caller vs callee cleanup differs (ret N). Heavy stack-vector loads. | **TODO** |
| **32-bit fastcall** | ecx/edx GPR args only; vectors still stack. | **TODO** |
| **32-bit thiscall (C++)** | `this` in ecx (MSVC) or stack (gcc); member fns with vector fields. | **TODO** |
| **SysV `regparm`/custom attrs** | GPR-arg overrides; rare but legal. | **TODO** |
| **Naked / no-prologue functions** | No save/restore at all — reads regs never written (the asm corpus's core trick). | **cover** (raw-asm corpus is no-save/restore by design) |
| **Varargs with vector args** | SysV: AL holds xmm-arg count; vectors spilled to register-save area. Unusual stack vector layout. | **TODO** |

---

## Axis 5 — Instruction encoding (highest-yield axis)

Same mnemonic, different encoding → potentially different `insn_t` flags the
handler must inspect. The lifter reads `insn.evex_flags` (`EVEX_z`, etc.), opmask
in Op6, broadcast/rounding decorators. Gaps here are the most likely INTERR
sources.

| Value | (a) risk | (b) coverage |
|---|---|---|
| **VEX (128/256)** | Auto zero-upper of dest on XMM write — lifter must model the implicit upper-clear. | **cover** (both corpora at 128/256) |
| **EVEX (512 + EVEX-128/256)** | Adds mask/broadcast/rounding/disp8*N; high regs. Core target. | **cover** (asm corpus uses EVEX zmm freely) |
| **EVEX-encoded 128/256 (AVX512VL)** | xmm/ymm with mask/{z}, zmm16-31 visible as xmm16-31/ymm16-31. Handlers keyed on width may misclassify. | **partial** (C corpus uses VL via VEC types; asm corpus mostly 512) |
| **Embedded broadcast `{1toN}`** (`vaddps (%rax){1to16}, ...`) | Memory operand semantically broadcasts; a width/element-count mismatch in the handler can mistype the load. | **TODO** (no generator emits `{1toN}`) |
| **Static rounding `{er}`** (`{rn-sae}`/`{ru-sae}`/`{rd-sae}`/`{rz-sae}`) | Only on reg-reg 512/scalar EVEX; sets EVEX.b as rounding-control. Handler must not treat as broadcast; rounding-mode operand may appear. | **TODO** (corpora emit `vgetmantps $imm` and `vroundss imm` but **not** `{er}` decorators) |
| **Suppress-all-exceptions `{sae}`** (e.g. `vcmpps {sae}`, `vcvt* {sae}`) | EVEX.b without rounding bits; compares/converts with SAE. | **TODO** |
| **Opmask `{k1}`–`{k7}` (merge)** | Op6 holds mask; handler must blend with dest. Masked path often unimplemented where unmasked exists. | **cover** (asm corpus emits `{k}`; C corpus emits `_mm512_mask_*`) |
| **Zero-masking `{k}{z}`** | EVEX.z set; zeroes unmasked lanes. Separate code path from merge. | **cover** (asm `{k}{z}`, C `_mm512_maskz_*`) |
| **k0 as mask (= "no mask") vs k0 as operand** | `{k0}` decorator is illegal/means unmasked; but k0 *is* a legal source/dest for `kmov`/`kand`. Confusing the two → wrong modeling. | **partial** (asm uses k1-7 for masks, k0 only via `kr0` in a few spots) |
| **disp8*N compressed displacement** | EVEX scales disp8 by tuple size; IDA decodes the effective disp, but tuple/element-size assumptions in memory handlers can drift. | **partial** (memory operands use small disps like `0x40(%rdi)`; not deliberately at compression boundaries) |
| **RIP-relative operands** | `vbroadcastss .LC(%rip), %zmm` constant pools; PIC globals. Memory-operand microcode differs from base+index. | **partial** (PIC global loads appear; not exhaustively across all mem-op forms) |
| **Segment overrides (`%fs:`/`%gs:`)** | TLS vector access, Windows TEB; segment-prefixed vector loads stress address computation. | **TODO** |
| **Redundant / legacy prefixes** (66/F2/F3/REX combos, redundant 66 on VEX) | Decoder usually normalizes, but malformed/redundant prefixes can yield odd `insn` shapes. (Belongs to the illegal-encoding component.) | **TODO** |
| **High EVEX regs zmm16-31 / xmm16-31 / ymm16-31** | Handlers predating ZMM modeling that call `reg2mreg` directly work for 0-15 but drop 16-31 to `__asm` (per README: `vpermps` etc.) — prime INTERR/`__asm` suspects. | **cover** (asm `zr()` ranges 0-31) |
| **Mask regs k1-k7 as explicit operands** (`kandw`, `kshiftl`, `kunpck`, `kmov`) | Mask-ALU modeling via `__readmask`/`__writemask`. | **cover** (asm corpus + C `_k*_mask*`) |
| **Mask↔GPR / mask↔vector** (`kmovw r,k`; `vpmovm2d`, `vpmovd2m`, `vpbroadcastmw2d`) | Type punning between mask and GPR/vector domains. | **cover** (both corpora) |
| **Scalar EVEX (`vaddss`/`vmulsd` with {er}/{k})** | Scalar register slice of a zmm with masking/rounding; upper-bits preservation semantics. | **partial** (scalar forms in asm; not masked/rounded scalar) |
| **NDD/APX (REX2, EVEX-promoted GPR, new data dest)** | Newest encodings (APX); if IDA decodes them, brand-new `insn` shapes. | **TODO** (likely beyond ISA flags here; future) |
| **Operand-order / commuted forms, `vfmadd132/213/231` variants** | Different operand positions for the same math; handler must map all three FMA forms. | **partial** (213ps/231ph in asm; 132 form TODO) |

---

## Axis 6 — ISA subset mix

Which subsets are *enabled* changes which intrinsics lower to which encodings and
whether VL/embedded features are available.

| Value | (a) risk | (b) coverage |
|---|---|---|
| **Full flags.txt union** (avx2…bf16, gfni/vaes/vpclmul) | Maximal mnemonic set; current build target. | **cover** (default) |
| **AVX2-only / AVX2+FMA (no AVX-512)** | VEX-only world; no mask regs; `vpermps`/`vgather*` in VEX form differ from EVEX. Common in shipped binaries. | **TODO** (never built without AVX-512) |
| **AVX-512F only (no VL/BW/DQ)** | 512-only, no EVEX-128/256, no byte/word masks (k as 16-bit only). Restricts mask widths → different `__mmask` types. | **TODO** |
| **Selective subset combos** (e.g. F+CD only; VBMI2 compress/expand; VNNI dp; IFMA madd52; BITALG; VPOPCNTDQ; BF16 dpbf16) | Each subset has signature instructions whose handler may be isolated; building with only one isolates which handler breaks. | **partial** (all enabled together; not isolated per-subset) |
| **GFNI / VAES / VPCLMULQDQ** | `vgf2p8affineqb`, `vaesenc` (VEX & EVEX 256/512), `vpclmulqdq` 512 — crypto idioms, often hand-written, EVEX-broadcast forms. | **TODO** (enabled in flags but no generator emits these mnemonics) |
| **AVX10.1 / AVX10.2** | Re-packages AVX-512 with a version knob; 256-bit-max variants (AVX10/256) and 512 (AVX10/512). Same encodings, but compilers may prefer 256 EVEX broadly. | **TODO** (toolchain may not target; same encodings as covered EVEX) |
| **Knights legacy: AVX-512ER/PF/4FMAPS/4VNNIW** (`v4fmaddps`, `vexp2ps`, `vrcp28ps`, `vgatherpf*`, `vscatterpf*`) | KNL/KNM-only; very rare; prefetch-gather and 4-source FMA have unusual operand shapes. High chance handler simply absent. | **TODO** (not in flags; would need explicit asm) |
| **AMX adjacency** (`tile*`, `tdpbf16ps`) | Tiles aren't vector regs but appear beside AVX-512 in ML kernels; mixing tile + zmm code in one function. | **TODO** (out of AVX scope; adjacency only) |
| **Plain SSE/SSE2 fallback paths** | Non-VEX legacy SSE mixed with AVX (no `vzeroupper` → SSE/AVX transition penalty code); legacy `movaps` vs `vmovaps`. | **TODO** (corpora are VEX/EVEX only) |

---

## Axis 7 — Control flow & layout

These shape the **CFG and microcode block structure** Hex-Rays builds; the
classic INTERR is "temporary crosses block boundary," so liveness across blocks
is the key stressor.

| Value | (a) risk | (b) coverage |
|---|---|---|
| **Straight-line blocks** | Baseline. | **cover** |
| **Vector/mask live across branches** | Forces cross-block temps; the historical INTERR trigger. | **cover** (C `if`/`for` wrappers, asm `kortestw`+`jz`) |
| **Deep nesting / many blocks** | Large microcode lists, complex phi/merge of vector temps. | **partial** (C nests one level; asm one branch) |
| **Loops with carried vector deps** | Reductions, accumulators live around back-edge; masked remainder loops. | **partial** (C `for` exists but body re-snapshots; no real carried zmm accumulator) |
| **Huge functions (1000s of insns)** | Microcode/list scaling, register-pressure modeling limits. | **partial** (asm max 22 insns + branch; not "huge") |
| **Jump tables / switch** | Indirect branch CFG; vector code in case arms. | **TODO** |
| **Overlapping / jumped-into instructions** | Decoder desyncs, instruction reuse at offset — pathological CFG. | **TODO** (belongs to illegal/raw-byte component) |
| **Tail calls (`jmp` to func)** | No return; vector state handed off; ABI tail-call thunks. | **TODO** |
| **Hot/cold splitting (`.text.unlikely`)** | Function body split across sections; cold part reads regs set in hot part. | **TODO** |
| **Exception handling: SEH (`.pdata`/`.xdata`)** | Windows unwind; xmm6-15 save offsets encoded in unwind data; funclets. | **partial** (mingw PE has `.pdata`; not deliberately exercised) |
| **EH: DWARF CFI / C++ landing pads** | `.eh_frame` unwind around vector spills; cleanup funclets. | **TODO** |
| **Alignment / padding (nop, multi-byte nop, `int3`)** | Inter-function padding; alignment nops inside loops; `vzeroupper` placement. | **partial** (asm emits `vzeroupper` before `ret`) |
| **Position of `vzeroupper` / SSE-AVX transitions** | Missing/extra `vzeroupper`; mixed legacy-SSE+VEX. | **partial** (always emitted at return; missing-case TODO) |
| **Recursion / mutual calls passing vectors** | Cross-call live vector args/returns. | **partial** (globals thread state; no actual vector-arg calls) |

---

## Axis 8 — Data / memory operands

Memory operand *form* changes the load/store microcode the lifter emits and how
it types the accessed vector.

| Value | (a) risk | (b) coverage |
|---|---|---|
| **Aligned moves (`vmovaps`/`vmovdqa64`)** | Alignment-assuming; handler must produce correctly-sized aligned load. | **cover** (asm `vmovaps`/`vmovdqu64`) |
| **Unaligned moves (`vmovups`/`vmovdqu*`)** | Common spill/reload + `loadu` intrinsics. | **cover** (both corpora) |
| **Streaming / non-temporal (`vmovntps`/`vmovntdq`)** | NT hint; same data path but distinct mnemonics. | **TODO** |
| **Base+index+scale, complex SIB** | General addressing; scale 1/2/4/8, index in GPR. | **partial** (asm uses `(%rdi)`, `0x40(%rdi)`, gather SIB) |
| **disp32 / large displacements** | Big offsets into structs/arrays; disp encoding paths. | **partial** (small disps only) |
| **RIP-relative constant pools** | `(%rip)` broadcasts of FP constants; PIC globals. | **partial** (PIC globals appear incidentally) |
| **Gather (vSIB + mask): `vgatherdps`/`vpgatherdd`/`vgatherqpd`** | Vector index register + mask; classic INTERR territory (multi-reg source, mask side effect, partial completion). | **cover** (asm corpus) |
| **Scatter (vSIB + mask): `vscatterdps`/`vpscatterdd`** | Vector index + mask store; even rarer handler. | **cover** (asm corpus) |
| **Gather/scatter prefetch (KNL `vgatherpf*`)** | Prefetch variants of vSIB. | **TODO** |
| **Compress/expand to memory (`vpcompressd m{k}`, `vpexpandd m{k}`)** | Mask-driven variable-length memory access. | **cover** (asm corpus) |
| **Embedded-broadcast memory `{1toN}`** | Broadcasting load form. | **TODO** (see Axis 5) |
| **TLS operands (`%fs:`/`%gs:`)** | Thread-local vector globals; segment-prefixed. | **TODO** |
| **Globals vs stack vs heap operands** | Global → RIP/abs; stack → `[rsp/rbp+..]` spill slots; heap → pointer-deref. Each types the vector location differently. | **partial** (C uses globals + stack; explicit heap-pointer-deref TODO) |
| **Misaligned / boundary-crossing accesses** | Loads straddling page/cacheline; only relevant to addressing microcode. | **TODO** |
| **Partial-width memory (xmm-load into zmm dest with zero-extend)** | Width-extending loads (`vmovss m32, xmm` zeroing upper). | **partial** (scalar `vmovd`/`vmovq`; not full extend matrix) |
| **Sub-vector broadcasts from memory (`vbroadcastf32x4 m128`)** | Loads a lane group and replicates; width bookkeeping. | **partial** (`vbroadcastss mem`; group-broadcast TODO) |

---

## Highest INTERR-risk, currently-uncovered shortlist (prioritized)

Ranked by (likelihood the handler path is missing) × (likelihood real binaries
hit it). Each maps to a concrete generator extension.

1. **Embedded broadcast `{1toN}` on memory operands** (Axis 5/8). No generator
   emits it; icc/clang -O3 emit it constantly. A width/tuple mismatch in a
   memory handler is a direct mistype/INTERR risk. → add `{1toN}` forms to the
   EVEX-fringe asm generator (task #12), e.g.
   `vaddps (%rdi){1to16}, %zmm1, %zmm2`.

2. **Static rounding `{er}` and `{sae}` decorators** (Axis 5). EVEX.b reused as
   rounding-control on reg-reg/scalar forms; a handler that assumes broadcast
   (or ignores the decorator) can emit wrong-shaped microcode. Zero coverage. →
   `vaddps {ru-sae}, %zmm,%zmm,%zmm`, `vcvtps2dq {rz-sae}, ...`,
   `vcmpps {sae}, ...`, scalar `vaddss {rn-sae}, ...`.

3. **High EVEX registers (zmm16-31) in the *masked/rounded/broadcast* handlers**
   (Axis 5). Plain zmm16-31 is covered, but the README explicitly flags
   pre-ZMM-modeling handlers (`reg2mreg`) that drop high regs to `__asm`. Combine
   high regs **with** mask/broadcast/rounding to maximize the chance of hitting
   the unconverted path → cross-product in the EVEX-fringe generator.

4. **MS x64 / __vectorcall / __regcall vector ABIs** (Axis 4). Callee-saved
   xmm6-15/zmm6-15 spill-restore and many vectors live on entry create cross-fn
   `__readzmm` synthesis the SysV corpus never exercises. mingw PE32+ builds and
   loads in idump today (verified) → ABI torture corpus (task #11) +
   cross-compiler PE matrix (task #14).

5. **GFNI / VAES / VPCLMULQDQ (esp. EVEX-512 + broadcast)** (Axis 6). Enabled in
   flags but **no generator emits a single one** of these mnemonics, despite
   being common in crypto/codec code. Likely entire handlers untested. → add
   `vgf2p8affineqb`, `vaesenc`/`vaesenclast` (256/512), `vpclmulqdq` to corpora.

6. **-O0 / -O2 / -O3 real codegen + clang & multi-version** (Axes 1/2). The C
   corpus is gcc -O1 only. -O0 stresses memory-vector typing; -O3 produces loop
   vectorizer idioms (gather/scatter/masked remainders/`vpternlog` fusion) the
   hand-asm only approximates. → build matrix that compiles the existing C corpus
   at {O0,O2,O3,Ofast} × {gcc,clang} (task #14).

7. **Knights-legacy & KNL gather-prefetch (4FMA/ER/PF)** (Axis 6/8). Extremely
   rare but handler almost certainly absent; if IDA decodes them they go to
   `__asm` at best, INTERR at worst. Lower priority (real-world rarity) but
   trivially added to the asm generator when the assembler accepts them.

8. **Segment-override / TLS vector operands and non-PIC absolute addressing**
   (Axis 5/8). `%fs:`/`%gs:` and absolute-addr vector loads exercise
   address-computation microcode paths the current PIC base+index corpus skips.

9. **Illegal / redundant-prefix / overlapping-instruction encodings** (Axes 5/7).
   Beyond-legal fuzzing for decoder-desync and malformed-`insn` robustness →
   dedicated raw-byte generator (task #13).
