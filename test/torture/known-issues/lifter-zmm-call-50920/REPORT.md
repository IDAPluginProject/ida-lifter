# RESOLVED — INTERR 50920 from an oversized GPR read in the rotate handler

**Status: FIXED** (was a lifter bug, not a Hex-Rays bug). Kept here as a
regression anchor. `repro.so` now decompiles 100% cleanly with the plugin.

## Symptom
`gen_abi_torture.py --seed 1 --funcs 40` built at `gcc -Os` produced function
`abi_ms_3`, which aborted with `INTERR 50920` ("temporary registers cannot
cross block boundaries") — but **only** with the lifter loaded. Native
(`idump --no-plugins`) was always clean.

## Actual root cause (the earlier "ZMM-across-a-call" theory was wrong)
`abi_ms_3` contains `vprold zmm4, zmmword ptr [r12+1C0h], 5` — a rotate with a
**memory** source operand. `handle_v_rotate()` loaded its source with
`reg2mreg(cdg.insn.Op2.reg)` and passed it as a 64-byte argument. For a memory
operand `Op2.reg` is the *base GPR* (`r12`), so the lifter emitted a 64-byte
read of register `r12`:

```
call !_mm512_rol_epi32<... r12.64, #5> => ...   ; r12.64 == read 64 bytes AT r12
```

A 64-byte read starting at `r12` spans `r12,r13,r14,r15,fps,…` and runs off the
end of the GPR file into the microcode **temporary registers** (`rt0`, `rt1`).
Those temps thus became live-in to the first block with no definition, and the
preoptimizer's liveness pass (MMAT_PREOPTIMIZED) rejected them with INTERR
50920. The cross-block / register-pressure framing was a red herring: the bug
fired wherever a rotate/shuffle/byte-shift had a memory source; high pressure
just made `abi_ms_3` the first corpus function to use that form.

## Fix (`src/avx/handlers/handler_logic.cpp`)
Load the reg-or-memory source via `AvxOpLoader` (which loads memory operands
from their effective address) instead of reading the base register, matching
the sibling `handle_v_shift()`. Three handlers shared the bug and were fixed:

- `handle_v_rotate`        — `vprold/vprord/vprolq/vprorq  zmm, m512, imm`
- `handle_vpslldq_vpsrldq` — `vpslldq/vpsrldq             zmm, m512, imm`
- `handle_v_shuffle_int`   — `vpshufd/vpshufhw/vpshuflw   zmm, m512, imm`

All other reg-or-mem source handlers were audited and already use
`AvxOpLoader` / `add_vec_source` (the 3-operand forms' `reg2mreg(Op2)` is the
EVEX.vvvv first source, which is always a register).

## Reproduce / verify the fix
```
idump --plugin lifter  repro.so   # now 100% clean (previously INTERR 50920)
idump --no-plugins     repro.so   # clean (baseline)
```
To see the bad/fixed microcode at generation + interr time, set `AVX_DUMP_MC=1`
(installs a microcode dumper on the hxe_microcode / hxe_interr events).
