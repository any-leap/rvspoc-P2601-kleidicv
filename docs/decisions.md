# Decisions

Tradeoffs explicitly chosen during the P2601 port. New entries get a
monotonically increasing `DEC-NNN` id; deletions leave the id as a placeholder.

## DEC-001 [build/architecture] 用 parallel CMake 根而不是改上游 top-level CMake

- 日期：2026-05-04
- 背景：上游 top-level `CMakeLists.txt` 无条件构建 AArch64 的 OBJECT 库（`kleidicv_neon` 等）并把它们链进公共 `kleidicv` 静态库。在 riscv64 toolchain 下连 configure 都过不去。
- 选项：
  - A. 在上游 top-level CMakeLists 里加 `if (CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")` 分支，把 Arm 子目录跳过。优点：单一构建入口；缺点：侵入上游、和未来 upstream merge 冲突。
  - B. 建并行的 `riscv/library/` CMake 根，自己挑头文件、自己加算子源；上游 tree 完全不动（除了公共头里两处 portability fix）。优点：diff 干净、上游可以照常演进；缺点：双构建入口，最终要合并。
  - C. 完全 fork 出独立项目。优点：最干净；缺点：脱离 KleidiCV 主线，不符合赛题"移植 KleidiCV"的语义。
- 决定：选 B。
- 理由：现阶段 RISC-V 只有 1 个算子的 scalar 实现，强行让 upstream top-level 通过需要大量 `if(NOT RISCV)` 分支，污染极大。等 RVV 算子铺到 90%+ 再做 final integration（Phase Final），那时 diff 自然就小了。
- #scaffolding #build

## DEC-002 [build/headers] 公共头给 RISC-V 开门是必要的，但只改最小

- 日期：2026-05-04
- 背景：`kleidicv.h` 写死 `#error "KleidiCV is only supported for aarch64"`；`ctypes.h` 写死 `typedef __fp16 float16_t`（`__fp16` 是 aarch64 编译器内建）。这两条让 riscv 端连 include 都过不了。
- 选项：
  - A. 在 riscv 子项目里把整个 public header **复制并改** 一份。优点：零上游改动；缺点：双份维护，公共 ABI 漂移风险高。
  - B. 上游 header 加最小 ifdef（允许 `__riscv`、float16_t 在非 aarch64 走 `_Float16` 或 opaque struct fallback）。优点：单一权威头；缺点：动了上游。
- 决定：选 B。两处共 ~12 行改动，纯 portability，不改 API 语义；将来 upstream PR 也容易接受。
- 理由：复制 header 是反模式，会引入静默漂移。最小 ifdef 是这种情况下教科书做法。
- #headers #portability

## DEC-003 [build/abi] RISC-V 暂不实现 dispatcher，直接用单一 backend 函数指针

- 日期：2026-05-04
- 背景：上游用 `KLEIDICV_MULTIVERSION_C_API_*` 宏在静态初始化时把 SVE2/SME 探测后选最优 impl，赋给 `kleidicv_xxx` 函数指针。
- 选项：
  - A. 现在就搬一个完整的 RISC-V dispatcher（探测 RVV → 选 RVV，否则选 scalar）。
  - B. 暂时直接 `kleidicv_xxx = &kleidicv::scalar::xxx`，所有"backend variant"只有 scalar 一个。等 Phase 3 同时存在 scalar+RVV 时再加 dispatcher。
- 决定：选 B。
- 理由：现在还没 RVV 实现可派发；过早抽象 dispatcher 只是占位代码。运行时 RVV 探测可以晚做（用 `getauxval(AT_HWCAP) & COMPAT_HWCAP_ISA_V` 或读 `/proc/cpuinfo`），不卡 Phase 2。
- #abi #dispatch
