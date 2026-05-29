#!/usr/bin/env python3
"""Group the __asm fallbacks left in idump pseudocode by instruction mnemonic.

Reads one or more idump --pseudo dumps (or stdin) and emits a report of every
instruction the lifter did not turn into an intrinsic, grouped by mnemonic and
deduplicated by operand form. Exit status is non-zero when --fail-on-asm is set
and any __asm remains, so it can gate CI.

Usage:
    idump --plugin lifter --pseudo avx512_test.o | report_missing_asm.py
    report_missing_asm.py dump1.txt dump2.txt -o report.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# __asm { mnemonic operands ... }  — stop at }, ; or newline.
ASM_RE = re.compile(r"__asm\s*\{\s*(?P<body>[^}\n;]+)")
TOTAL_FN_RE = re.compile(r"Total functions:\s*(\d+)")
DECOMPILED_RE = re.compile(r"Decompiled OK:\s*(\d+)")


def extract(text: str):
    text = ANSI_RE.sub("", text)
    forms = defaultdict(lambda: defaultdict(int))  # mnemonic -> form -> count
    total = 0
    for m in ASM_RE.finditer(text):
        body = m.group("body").strip()
        if not body or body.startswith("//"):
            continue
        mnemonic = body.split()[0].lower()
        forms[mnemonic][body] += 1
        total += 1
    return forms, total


def render(forms, total, analyzed=None, decompiled=None, symbols=None) -> str:
    unique = sum(len(v) for v in forms.values())
    out = []
    # Coverage banner: distinguishes "lifted everything" from "IDA never saw it".
    if analyzed is not None:
        out.append(f"Functions analyzed by IDA: {analyzed}")
    if decompiled is not None:
        out.append(f"Functions decompiled OK:   {decompiled}")
    if symbols is not None:
        out.append(f"Function symbols in binary: {symbols}")
        if analyzed is not None and symbols > analyzed:
            out.append(
                f"WARNING: {symbols - analyzed} symbol(s) were NOT turned into "
                f"functions — IDA's decoder could not analyze them (e.g. very new "
                f"AVX10.2/AMX opcodes). Those are outside the lifter's reach until "
                f"IDA decodes them, and are NOT counted below."
            )
    if out:
        out.append("")
    out += [
        f"Total __asm occurrences: {total}",
        f"Unique instruction forms: {unique}",
        "",
    ]
    # Most frequent mnemonics first, then alphabetical.
    def key(item):
        mnem, fmap = item
        return (-sum(fmap.values()), mnem)

    for mnem, fmap in sorted(forms.items(), key=key):
        occ = sum(fmap.values())
        out.append(f"## {mnem} (occurrences: {occ}, forms: {len(fmap)})")
        for form, count in sorted(fmap.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append(f"- {form}  [{count}]")
        out.append("")
    if total == 0:
        out.append("No __asm fallbacks — every instruction lifted to an intrinsic.")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", help="idump pseudo dumps (default: stdin)")
    ap.add_argument("-o", "--output", help="write report to file instead of stdout")
    ap.add_argument("--fail-on-asm", action="store_true",
                    help="exit non-zero if any __asm remains")
    ap.add_argument("--symbols", type=int, default=None,
                    help="number of function symbols in the binary (for coverage check)")
    args = ap.parse_args()

    if args.inputs:
        text = ""
        for path in args.inputs:
            with open(path, "r", errors="replace") as fh:
                text += fh.read()
    else:
        text = sys.stdin.read()

    forms, total = extract(text)
    analyzed = max((int(m.group(1)) for m in TOTAL_FN_RE.finditer(text)), default=None)
    decompiled = max((int(m.group(1)) for m in DECOMPILED_RE.finditer(text)), default=None)
    report = render(forms, total, analyzed, decompiled, args.symbols)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report)
        sys.stderr.write(f"{total} __asm occurrence(s); report written to {args.output}\n")
    else:
        sys.stdout.write(report)

    return 1 if (args.fail_on_asm and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
