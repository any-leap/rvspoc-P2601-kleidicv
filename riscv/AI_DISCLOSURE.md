<!--
SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
SPDX-License-Identifier: Apache-2.0
-->

# AI-assisted code disclosure

Per RVSPOC P2601 submission requirements ("Disclosure of AI-assisted code
with usage percentage noted"), this file describes how AI assistance was
used in producing the RISC-V port.

## Tooling

- **Claude Code (Anthropic Claude Opus 4.7, 1M context)** was used as a
  pair-programming assistant throughout the port. The author drove the
  task list, reviewed every diff, ran every build/test, and made all
  algorithmic and architectural decisions. Claude generated initial
  drafts of source files, suggested intrinsic sequences, and helped
  reason about RVV intrinsic signatures (notably the gcc-13/gcc-14 vnclip
  argument-count change documented in `docs/findings.md` FIND-005).

- No other LLMs, code generators, autocomplete tools, or AI-driven
  search were used.

## Approximate AI-vs-human authorship

These are rough estimates by inspection of git history and conversation
transcripts; they are not exact line counts.

| Area | AI-drafted | Human-revised | Net |
|---|---|---|---|
| RVV intrinsic kernels (`*_rvv.cpp`) | ~75% | ~25% | substantial AI assist for the boilerplate vsetvli/strip-mining + intrinsic naming; humans verified semantics, fixed signature mismatches, and corrected several rounding/sign-extension bugs that the AI introduced |
| Scalar reference implementations (`*_scalar.cpp`) | ~60% | ~40% | AI handled the obvious arithmetic; humans pinned the exact rounding modes / fixed-point shifts to match upstream OpenCV / Arm conventions |
| Test harness (`test/test_*.cpp`, `bench_kleidicv.cpp`) | ~70% | ~30% | AI drafted the structure and reference-vs-vector cross-checks; humans defined the test cases (sizes, edge inputs, tolerances) and chased down failing cases until each test was actually meaningful |
| Build configuration (`CMakeLists.txt`, `scripts/`, `docker/`) | ~30% | ~70% | small, mostly human; AI helped with dispatch wiring boilerplate |
| Documentation (`README.md`, `PORTING_GUIDE.md`, `docs/*.md`) | ~50% | ~50% | AI drafted prose and structure; humans rewrote sections describing project decisions, constraints, and the deliverable mapping |
| Optical-flow port (`riscv/library/src/optical_flow_*` / `standalone_lucas_kanade_*`) | ~70% | ~30% | AI drafted by analogy from upstream SVE2 sources; humans handled the algorithmic correctness — particularly the FractionBits=14 fixed-point convention, the structure-tensor numerics, and the pyramid border policy that exposed the BORDER_TYPE_REVERSE bug captured in `docs/findings.md` FIND-007 |

**Net estimate: ~60% of the new code in `riscv/` is AI-drafted, with
human revision and correctness verification on every line that landed in
the tree.** Every commit and every line that appears in this PR was
read, tested, and approved by a human reviewer; no code was merged
without the test suite passing.

## Verification of AI-drafted code

- All AI-drafted code passed the standard project ctests (46 tests at
  VLEN=128/256/512) before being committed.
- For each new RVV kernel, `objdump -d` was inspected manually to
  confirm the expected vector instructions are emitted in the inner
  loop (`vsetvli` / `vle*` / `v…v` / `vse*` / `vnclip*`).
- For floating-point paths (resize_linear, warp_perspective bilinear),
  RVV vs scalar reference cross-check is bounded to ±1 LSB on u8 and
  documented in the README.
- For optical-flow / Lucas-Kanade, end-to-end displacement recovery on
  synthetic images is verified to within 0.5 px of ground truth at
  multiple pyramid levels.
