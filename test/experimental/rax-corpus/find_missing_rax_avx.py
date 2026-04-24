#!/usr/bin/env python3
"""Build a raw-byte AVX corpus from RAX tests and report unlifted asm.

RAX keeps broad AVX/AVX2/FMA instruction fixtures as Rust byte arrays. This
script converts those bytes into one x86_64 object file with one symbol per
fixture, runs idump with the lifter plugin, and groups remaining __asm blocks
by mnemonic and source fixture.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GROUPS = ("avx", "avx2", "fma")
BYTE_ARRAY_RE = re.compile(
    r"#\[test\]\s*fn\s+(?P<test>[A-Za-z0-9_]+)\s*\(\)\s*\{(?P<body>.*?)\n\}",
    re.DOTALL,
)
CODE_RE = re.compile(r"let\s+code\s*=\s*\[(?P<bytes>.*?)\];", re.DOTALL)
COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
FUNCTION_RE = re.compile(r"\b(?:void|int|__int64|_QWORD|char|bool|float|double)\s+(?:__fastcall\s+)?(?P<name>rax_[A-Za-z0-9_]+)\s*\(")
FUNCTION_BANNER_RE = re.compile(r"\bFunction:\s+_?(?P<name>rax_[A-Za-z0-9_]+)\b")
ASM_RE = re.compile(r"__asm\s*\{\s*(?P<body>[^}\n;]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ERROR_RE = re.compile(r"^\s*_?(?P<name>rax_[A-Za-z0-9_]+):\s+(?P<error>INTERR:\s+\d+|[^\n]+)$")


@dataclass(frozen=True)
class Fixture:
    group: str
    source: Path
    test_name: str
    symbol: str
    bytes_: tuple[int, ...]


def default_rax_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    candidates = []
    if os.environ.get("RAX"):
        candidates.append(Path(os.environ["RAX"]))
    candidates.extend([
        repo_root.parent / "rax",
        Path.cwd() / "rax",
    ])
    for candidate in candidates:
        if (candidate / "tests" / "x86_64" / "simd").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "could not locate RAX checkout; pass --rax /path/to/rax or set RAX=/path/to/rax"
    )


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "fixture"


def parse_byte_list(text: str) -> tuple[int, ...]:
    text = COMMENT_RE.sub("", text)
    out: list[int] = []
    for part in text.replace("\n", " ").split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token, 0)
        if value < 0 or value > 0xFF:
            raise ValueError(f"byte value out of range: {token}")
        out.append(value)
    if out and out[-1] == 0xF4:
        out[-1] = 0xC3  # Replace HLT test terminator with RET for IDA function discovery.
    elif not out or out[-1] != 0xC3:
        out.append(0xC3)
    return tuple(out)


def collect_fixtures(rax: Path, groups: tuple[str, ...], only: str | None) -> list[Fixture]:
    fixtures: list[Fixture] = []
    only_re = re.compile(only, re.IGNORECASE) if only else None
    base = rax / "tests" / "x86_64" / "simd"
    for group in groups:
        group_dir = base / group
        if not group_dir.is_dir():
            raise FileNotFoundError(f"missing RAX test directory: {group_dir}")
        for source in sorted(group_dir.glob("*.rs")):
            if source.name == "mod.rs":
                continue
            text = source.read_text(errors="ignore")
            stem = sanitize_name(source.stem)
            for index, match in enumerate(BYTE_ARRAY_RE.finditer(text), 1):
                test_name = match.group("test")
                code_match = CODE_RE.search(match.group("body"))
                if not code_match:
                    continue
                if only_re and not (only_re.search(source.stem) or only_re.search(test_name)):
                    continue
                bytes_ = parse_byte_list(code_match.group("bytes"))
                symbol = f"rax_{group}_{stem}_{sanitize_name(test_name)}_{index}"
                fixtures.append(Fixture(group, source, test_name, symbol, bytes_))
    return fixtures


def asm_symbol(symbol: str) -> str:
    return f"_{symbol}" if platform.system() == "Darwin" else symbol


def write_assembly(fixtures: list[Fixture], path: Path, init_vector_regs: bool) -> None:
    lines = [".intel_syntax noprefix", ".text", ""]
    for fixture in fixtures:
        name = asm_symbol(fixture.symbol)
        byte_text = ", ".join(f"0x{b:02x}" for b in fixture.bytes_)
        lines.extend(
            [
                f"# {fixture.group}/{fixture.source.name}:{fixture.test_name}",
                f".globl {name}",
                ".p2align 4, 0x90",
                f"{name}:",
            ]
        )
        if init_vector_regs:
            for reg in range(16):
                # Use legacy SSE zero idioms so the AVX lifter does not turn the
                # seed into intrinsic calls that themselves read undefined values.
                lines.append(f"  xorps xmm{reg}, xmm{reg}")
        lines.extend([f"  .byte {byte_text}", ""])
    path.write_text("\n".join(lines))


def run(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def compile_object(asm_path: Path, obj_path: Path, cc: str) -> None:
    command = [cc]
    if platform.system() == "Darwin":
        command.extend(["-arch", "x86_64"])
    command.extend(["-c", str(asm_path), "-o", str(obj_path)])
    result = run(command)
    if result.returncode != 0:
        raise RuntimeError(f"compile failed:\n{result.stdout}")


def remove_ida_sidecars(obj_path: Path) -> None:
    for suffix in (".id0", ".id1", ".id2", ".nam", ".til", ".i64", ".idb"):
        sidecar = Path(str(obj_path) + suffix)
        if sidecar.exists():
            if sidecar.is_dir():
                shutil.rmtree(sidecar)
            else:
                sidecar.unlink()


def run_idump(obj_path: Path, output_path: Path, idump: str) -> tuple[str, int]:
    remove_ida_sidecars(obj_path)
    result = run([idump, "--plugin", "lifter", "--pseudo", str(obj_path)], timeout=300)
    output_path.write_text(result.stdout)
    return result.stdout, result.returncode


def parse_unlifted(output: str, by_symbol: dict[str, Fixture]) -> dict[str, list[tuple[str, str]]]:
    current_symbol: str | None = None
    missing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for raw_line in output.splitlines():
        line = ANSI_RE.sub("", raw_line)
        function_match = FUNCTION_BANNER_RE.search(line) or FUNCTION_RE.search(line)
        if function_match:
            current_symbol = function_match.group("name")
        asm_match = ASM_RE.search(line)
        if not asm_match:
            continue
        body = asm_match.group("body").strip()
        mnemonic_match = re.match(r"([A-Za-z][A-Za-z0-9]+)", body)
        if not mnemonic_match:
            continue
        mnemonic = mnemonic_match.group(1).lower()
        if current_symbol and current_symbol in by_symbol:
            fixture = by_symbol[current_symbol]
            source = f"{fixture.group}/{fixture.source.name}:{fixture.test_name}"
        else:
            source = current_symbol or "<unknown>"
        missing[mnemonic].append((source, body))
    return missing


def parse_decompilation_errors(output: str, by_symbol: dict[str, Fixture]) -> list[tuple[str, str, str]]:
    errors: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = ERROR_RE.match(line)
        if not match:
            continue
        symbol = match.group("name")
        if symbol in seen:
            continue
        seen.add(symbol)
        fixture = by_symbol.get(symbol)
        source = f"{fixture.group}/{fixture.source.name}:{fixture.test_name}" if fixture else symbol
        errors.append((source, symbol, match.group("error")))
    return errors


def write_report(
    report_path: Path,
    fixtures: list[Fixture],
    missing: dict[str, list[tuple[str, str]]],
    errors: list[tuple[str, str, str]],
    obj_path: Path,
    pseudo_path: Path,
    idump_rc: int,
) -> None:
    lines = [
        "# RAX AVX Missing ASM Report",
        "",
        f"Corpus object: `{obj_path}`",
        f"Pseudo output: `{pseudo_path}`",
        f"Fixtures: {len(fixtures)}",
        f"idump exit code: {idump_rc}",
        f"Decompilation errors: {len(errors)}",
        f"Missing mnemonics: {len(missing)}",
        "",
    ]
    if errors:
        lines.extend(["## Decompilation Errors", ""])
        for source, symbol, error in errors:
            lines.append(f"- `{error}` in `{symbol}` from `{source}`")
        lines.append("")
    if not missing:
        lines.append("No `__asm` blocks found.")
    for mnemonic in sorted(missing):
        entries = missing[mnemonic]
        lines.extend([f"## {mnemonic} ({len(entries)})", ""])
        seen: set[tuple[str, str]] = set()
        for source, body in entries:
            key = (source, body)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- `{body}` from `{source}`")
        lines.append("")
    report_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rax", help="path to RAX checkout; auto-detected from RAX env or sibling ../rax if omitted")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS), help="comma-separated RAX SIMD groups")
    parser.add_argument("--only", help="regex filter for RAX source/test names")
    parser.add_argument("--build-dir", default="build", help="generated artifact directory")
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"), help="C/assembler driver")
    parser.add_argument("--idump", default=os.environ.get("IDUMP", "idump"), help="idump executable")
    parser.add_argument("--no-idump", action="store_true", help="only generate and compile the corpus object")
    parser.add_argument(
        "--no-init-vector-regs",
        action="store_true",
        help="do not seed YMM registers before each raw fixture",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    build_dir = (script_dir / args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    groups = tuple(group.strip() for group in args.groups.split(",") if group.strip())
    rax_path = Path(args.rax).resolve() if args.rax else default_rax_path()
    fixtures = collect_fixtures(rax_path, groups, args.only)
    if not fixtures:
        print("No RAX fixtures matched", file=sys.stderr)
        return 1

    asm_path = build_dir / "rax_avx_corpus.s"
    obj_path = build_dir / "rax_avx_corpus.o"
    pseudo_path = build_dir / "rax_avx_corpus.pseudo.txt"
    report_path = build_dir / "rax_avx_missing_asm.md"

    write_assembly(fixtures, asm_path, not args.no_init_vector_regs)
    compile_object(asm_path, obj_path, args.cc)

    print(f"Generated {len(fixtures)} fixtures")
    print(f"Assembly: {asm_path}")
    print(f"Object:   {obj_path}")

    if args.no_idump:
        return 0

    output, idump_rc = run_idump(obj_path, pseudo_path, args.idump)
    by_symbol = {fixture.symbol: fixture for fixture in fixtures}
    missing = parse_unlifted(output, by_symbol)
    errors = parse_decompilation_errors(output, by_symbol)
    write_report(report_path, fixtures, missing, errors, obj_path, pseudo_path, idump_rc)

    print(f"Pseudo:   {pseudo_path}")
    print(f"Report:   {report_path}")
    if idump_rc != 0:
        print(f"idump exited with {idump_rc}; decompilation errors are included in the report")
    print(f"Decompilation errors: {len(errors)}")
    print(f"Missing mnemonics: {len(missing)}")
    for mnemonic in sorted(missing):
        print(f"  {mnemonic}: {len(missing[mnemonic])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
