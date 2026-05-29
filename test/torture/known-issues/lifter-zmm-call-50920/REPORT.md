# KNOWN LIFTER ISSUE — INTERR 50920 on ZMM values live across a call (high pressure)

**This is OUR plugin's bug, NOT a Hex-Rays bug. Do not file with the decompiler team.**

Verified: with the lifter disabled (`idump --no-plugins repro.so`, or with
`~/.idapro/plugins/lifter.so` removed) the function decompiles **100% cleanly** —
the spilled `__m512i` becomes a proper stack variable, exactly as expected. The
INTERR appears **only** with our plugin loaded (`idump --plugin lifter repro.so`).

## Symptom
`abi_ms_3` aborts with `INTERR 50920` ("Temporary registers cannot cross block
boundaries") at `gcc -Os`. `repro.c` is a 29-line reduction. The currently
*surviving* case is the full `gen_abi_torture.py --seed 1` corpus at `-Os`
(function `abi_ms_3`), where interprocedural register pressure is highest.

## Root cause
Our lifter models ZMM registers as `__readzmm(idx)` / `__writezmm(idx)` helper
calls whose values are 64-byte UDTs. In a function that keeps such a value live
across a real call, Hex-Rays' pre-optimization builds a 64-byte temporary that
spans the call's basic-block boundary and the verifier rejects it (50920).
Natively (ZMM as real registers) Hex-Rays spills to a stackvar and is fine.

## Status of fixes (`src/avx/`)
- Freeing result kregs in `emit_zmm_write_call`/`emit_kmask_write_call` fixes
  low/moderate-pressure cases (incl. this minimal `repro.c`).
- The very-high-pressure full-corpus case is still open. Experiments that did NOT
  help (reverted): virtual-return nesting (regressed other cases), dropping
  `FCI_SPLOK`, `visible_memory=ALLMEM`, freeing `AvxOpLoader` load temporaries.
- Proper fix likely needs reworking the wide-register modeling so values feeding
  calls don't require a cross-block 64-byte temporary (or upstream support).

## Reproduce
```
idump --plugin lifter  repro.so   # INTERR 50920 (with our plugin)
idump --no-plugins     repro.so   # clean (proves it's the plugin, not IDA)
```
