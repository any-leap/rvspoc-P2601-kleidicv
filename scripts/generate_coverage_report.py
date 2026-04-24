#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# Generate filtered coverage reports from an instrumented build directory.
# Produces `coverage.info`, `cobertura-coverage.xml`, and HTML output.
# Usage: `scripts/generate_coverage_report.py [build_directory]`
# If omitted, `build_directory` defaults to `build/ci`.

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

GCOVR_EXCL_LINE_MARKER = "// GCOVR_EXCL_LINE"
BRACE_ONLY_LINE_PATTERN = re.compile(r"^\s*}\s*(?://.*)?$")


def find_excluded_line_numbers(source_path):
    excluded_line_numbers = set()
    try:
        with open(source_path, encoding="utf-8") as source_file:
            for line_number, source_line in enumerate(source_file, start=1):
                if GCOVR_EXCL_LINE_MARKER in source_line:
                    excluded_line_numbers.add(line_number)
                    continue
                if BRACE_ONLY_LINE_PATTERN.match(source_line):
                    excluded_line_numbers.add(line_number)
                    continue
    except OSError:
        pass

    return excluded_line_numbers

# Make sure to find all test binaries that correspond to the profraw files,
# even if they are in subdirectories of the test directory.
def find_test_binaries(build_dir, profraw_files):
    binaries = []
    search_roots = set()
    build_dir = Path(build_dir).resolve()
    for profraw_file in profraw_files:
        for parent in Path(profraw_file).resolve().parents:
            if not parent.is_relative_to(build_dir):
                break
            if (parent / "test").is_dir():
                search_roots.add(parent / "test")
                break

    for search_root in sorted(search_roots or {build_dir / "test"}):
        for root, _, files in os.walk(search_root):
            for filename in files:
                if filename.endswith("-test"):
                    binaries.append(os.path.join(root, filename))
    return sorted(binaries)


def object_arguments(binaries):
    args = []
    for binary in binaries[1:]:
        args.extend(["--object", binary])
    return args


def summary_line_from_genhtml(genhtml_output):
    line_match = re.search(r"lines\.*:\s+([0-9.]+)%", genhtml_output)
    branch_match = re.search(r"branches\.*:\s+([0-9.]+)%", genhtml_output)
    if line_match is None or branch_match is None:
        print("Failed to parse genhtml summary output")
        sys.exit(1)

    line_pct = float(line_match.group(1))
    branch_pct = float(branch_match.group(1))
    combined_coverage = (line_pct + branch_pct) / 2.0
    return f"line and branch coverage: {combined_coverage:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build_directory", nargs="?")
    args = parser.parse_args()

    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = args.build_directory or os.path.join(source_dir, "build", "ci")

    profraw_files = sorted(str(path) for path in Path(build_dir).rglob("*.profraw"))
    if not profraw_files:
        print(f"No .profraw files found in {build_dir}")
        sys.exit(1)

    profdata = os.path.join(build_dir, "coverage.profdata")
    subprocess.check_call(["llvm-profdata", "merge", "-sparse", *profraw_files, "-o", profdata])

    binaries = find_test_binaries(build_dir, profraw_files)
    if not binaries:
        print("No test binary found")
        sys.exit(1)

    generated_config = re.escape(os.path.abspath(build_dir)) + r"/kleidicv/include/kleidicv/config.h$"
    ignore_pattern = rf".*/test(/.*|$)|.*/(googletest|gmock|gtest|_deps)/.*|{generated_config}"
    export_command = [
        "llvm-cov",
        "export",
        "--format=lcov",
        f"--instr-profile={profdata}",
        binaries[0],
        *object_arguments(binaries),
        f"--ignore-filename-regex={ignore_pattern}",
    ]

    lcov_output = subprocess.check_output(export_command, text=True)

    # Filter out lines corresponding to excluded line numbers to avoid issues with lcov_cobertura and genhtml.
    # This is necessary because they don't support the same ignore patterns as llvm-cov export.
    excluded_line_numbers = set()
    filtered_lcov = []
    for line in lcov_output.splitlines():
        if line.startswith("SF:"):
            excluded_line_numbers = find_excluded_line_numbers(line[3:])
        elif line.startswith(("DA:", "BRDA:")):
            line_no = int(line.split(":", 1)[1].split(",", 1)[0])
            if line_no in excluded_line_numbers:
                continue
        filtered_lcov.append(line)

    coverage_info = os.path.join(build_dir, "coverage.info")
    with open(coverage_info, "w") as f:
        f.write("\n".join(filtered_lcov))

    summary_file = os.path.join(build_dir, "cobertura-coverage.xml")

    subprocess.check_call([
        "lcov_cobertura",
        coverage_info,
        "-b",
        source_dir,
        "-o",
        summary_file,
    ])

    html_dir = os.path.join(build_dir, "html", "coverage")
    shutil.rmtree(html_dir, ignore_errors=True)
    genhtml = subprocess.run([
        "genhtml",
        coverage_info,
        "--output-directory",
        html_dir,
        "--prefix",
        source_dir,
        "--source-directory",
        source_dir,
        "--branch-coverage",
        "--no-function-coverage",
        "--show-details",
        "--legend",
        "--quiet",
    ], check=True, text=True, capture_output=True)

    print(summary_line_from_genhtml(genhtml.stdout + genhtml.stderr))


if __name__ == "__main__":
    main()
