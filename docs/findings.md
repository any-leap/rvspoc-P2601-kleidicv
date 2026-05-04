# Findings

Non-obvious facts uncovered while working on P2601. New entries get a
monotonically increasing `FIND-NNN` id.

## FIND-001 [build/dev-env] 上游仓库已包含完整 KleidiCV 26.03 源码

- 日期：2026-05-04
- 现象：`gh repo view rv2036/rvspoc-P2601-kleidicv` 看起来只是个空仓，但 fork 之后发现里面有 527 个文件，11M 大小，`kleidicv/` `kleidicv_thread/` `benchmark/` `test/` `examples/` 全是上游 KleidiCV 26.03 源码。
- 根因/机制：组委会已经把上游 mirror 进 GitHub 仓库（pushed 2026-04-25），选手只需在它之上加 RISC-V backend 并 PR。我们不用自己 vendor KleidiCV 源码。
- 证据/复现：`git log --oneline --all | head` 能看到 26.03 release tag 之前的历史；`grep -r "neon\|sve" kleidicv/src/` 命中数百处。
- #scaffolding #upstream

## FIND-002 [build/dev-env] ports.ubuntu.com 在国内不可用，必须换镜像

- 日期：2026-05-04
- 现象：基于 `ubuntu:24.04` 的 Dockerfile 跑 `apt-get install` 拉 riscv64 cross toolchain，单个 18MB 的 `gcc-13-riscv64-linux-gnu` 包 20 分钟下不完，大量 `Ign:` 重试。
- 根因/机制：`ports.ubuntu.com`（arm64/riscv 包源）在国内出口路径极慢，与 `archive.ubuntu.com` 不是同一镜像。
- 证据/复现：见 GOT-001。
- #network #docker #cn

## FIND-003 [build/dev-env] Ubuntu 24.04 sources 在 deb822 格式，sed 要改对路径

- 日期：2026-05-04
- 现象：网上很多换源教程改 `/etc/apt/sources.list`，在 24.04 上无效——这个文件不存在。
- 根因/机制：noble 起改用 deb822 格式，sources 在 `/etc/apt/sources.list.d/ubuntu.sources`，每条记录是 `URIs:` `Types:` `Suites:` 多行结构。
- 证据/复现：`docker run --rm ubuntu:24.04 ls /etc/apt/sources.list.d/` → 只有 `ubuntu.sources`。
- #docker #ubuntu24

## FIND-005 [rvv/intrinsics] gcc 13.3 不带 RVV segment ld/st intrinsics

- 日期：2026-05-04
- 现象：写 `__riscv_vsseg3e8_v_u8m1(...)` 编译报 implicit declaration；`vuint8m1x3_t` tuple 类型也不存在；`__riscv_vcreate_v_u8m1x3` 同样。
- 根因/机制：gcc 13.3 的 RVV intrinsics 是 v1.0 spec 的早期版本，segment ld/st (`vlseg*`/`vsseg*`) 和 tuple types (`vuintNmKxM_t`) 在 gcc 14 及之后才齐。Ubuntu 24.04 noble 默认 gcc 13。
- 证据/复现：
  ```
  riscv64-linux-gnu-gcc -march=rv64gcv -c <test using vsseg3e8> -> implicit declaration
  ls /usr/lib/gcc-cross/riscv64-linux-gnu/13/include/riscv_vector.h  -> exists, but no vsseg
  ```
- 影响 / 工作绕路：
  - 多通道 interleave 输出 (gray_to_rgb, rgb_to_yuv interleave 部分)：用 N 次 strided store (`vsse*` with stride=channel_count)。功能等价，性能上比一次 vsseg 差但能跑。
  - 多通道 deinterleave 输入 (split, rgb_to_yuv 加载 RGB 等)：strided load `vlse*`。
  - 真要 vsseg 性能：升级到 noble-backports 或自己 build gcc 14；或者切换到 LLVM。**暂不做**——qemu 上 strided 和 segment 性能差距没意义，等真板再说。
- #rvv #toolchain #gcc

## FIND-004 [build/qemu] qemu-user 必须给 `-L sysroot` 才能跑动态链接的 riscv64 ELF

- 日期：2026-05-04
- 现象：`qemu-riscv64 ./hello_rvv` 报 `Could not open '/lib/ld-linux-riscv64-lp64d.so.1'`。
- 根因/机制：交叉编译产生的 ELF 写死的 interpreter 路径是目标 sysroot 里的（`/lib/ld-linux-riscv64-lp64d.so.1`），qemu-user 在宿主 rootfs 里找不到。`-L /usr/riscv64-linux-gnu` 让 qemu 把这个路径作为虚拟根。
- 证据/复现：`file riscv/smoke/build/hello_rvv` 能看到 `interpreter /lib/ld-linux-riscv64-lp64d.so.1`。
- #qemu #cross-compile
