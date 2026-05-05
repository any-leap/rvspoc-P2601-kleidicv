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

## FIND-007 [optical_flow] LK pyramid build 内部用的是 BORDER_TYPE_REVERSE 不是 REPLICATE

- 日期：2026-05-05
- 现象：把 `kleidicv_optical_flow_pyr_lk_u8` 的 NOT_IMPLEMENTED stub 替换成真实实现后，调用立刻返回 `KLEIDICV_ERROR_NOT_IMPLEMENTED`(=1)。pyramid 自身、`standalone_lucas_kanade_alg_u8` 都各自工作，只有 image-to-image 这条路径挂。
- 根因/机制：上游 `OpticalFlowLKPyramid::create<>()` 模板调 `kleidicv_blur_and_downsample_u8(...,KLEIDICV_BORDER_TYPE_REVERSE)`，REVERSE 是 OpenCV 的 BORDER_REFLECT_101（边界镜像、不重复边缘像素）。我们 RISC-V 的 blur impl 当时只接受 REPLICATE，遇到 REVERSE 直接 NOT_IMPLEMENTED 返回。pyramid 在调 blur 之前已经预先用 reflect_101 把 border 区域填好了（`fill_reflect_101_border_in_place`），所以 REPLICATE 内部 clip 也只是读到 pyramid 自己填的有效像素。
- 应对：`riscv/library/src/blur_and_downsample_api.cpp` 同时接受 REPLICATE 和 REVERSE，转发到同一条路径。结果不是数学上完全严格的 REFLECT_101 blur（边界几像素的 blur 系数会偏一点），但 LK tracker 用的是远离边界的内部像素，影响可忽略。完全严格的实现需要 blur 自己支持 REFLECT_101 clip——后续 RVV 优化时一起补。
- 证据/复现：`bash riscv/scripts/build-lib.sh` 触发 `optical_flow_*` 测试；旧版本 err=1（NOT_IMPLEMENTED）来自 `build_optical_flow_pyr_lk_pyramid_impl`。
- #optical_flow #blur #pyramid

## FIND-005 [rvv/intrinsics] gcc 版本切换会换 vnclip 的参数个数

- 日期：2026-05-04
- 现象：
  - gcc 13.3：`__riscv_vnclip_wx_iNm1` 是 3 参数 `(src, shift, vl)`，写 4 参数报 "too many arguments"。
  - gcc 14.2：同名函数变成 4 参数 `(src, shift, vxrm_mode, vl)`，写 3 参数报 "too few arguments"。
- 根因/机制：v1.0 RVV intrinsics spec 后来加了显式 `vxrm` 舍入模式参数；gcc 13 是 spec 早期版本，gcc 14 跟进了 spec 修订。同一个 intrinsic 在两版 gcc 之间签名不兼容。
- 证据/复现：用 `__RISCV_VXRM_RNU` 常量（gcc 14 头里有）作为第 3 个参数，gcc 14 通过；这个宏在 gcc 13 头里不存在。
- 应对：项目锁定在 gcc 14.2（noble-updates 直接装），整代码库统一用 4 参数版本。如果谁拿 gcc 13 build 会立即报错，预期。
- #rvv #toolchain #gcc

## FIND-006 [rvv/intrinsics] gcc 13 没有 segment ld/st intrinsics（已通过升级 gcc 14 解决）

- 日期：2026-05-04
- 现象（旧）：gcc 13.3 上 `__riscv_vsseg3e8_v_u8m1`、`vuint8m1x3_t` tuple 类型、`__riscv_vcreate_v_u8m1x3` 全部缺失。
- 根因/机制：早期 RVV v1.0 intrinsics 不含 segment 形式与 tuple types；gcc 14 起齐。
- 解决：dev image 升级到 g++-14.2-riscv64-linux-gnu（noble-updates/universe 直接有）。`vsseg3e8.v` 现在能直接 emit（objdump 验证）。
- 历史绕路（保留以备同情况复现）：3 次 `vsse8.v` strided store at offsets 0/1/2 with stride=3。功能等价，性能 ~3x 差。
- #rvv #toolchain #gcc

## FIND-004 [build/qemu] qemu-user 必须给 `-L sysroot` 才能跑动态链接的 riscv64 ELF

- 日期：2026-05-04
- 现象：`qemu-riscv64 ./hello_rvv` 报 `Could not open '/lib/ld-linux-riscv64-lp64d.so.1'`。
- 根因/机制：交叉编译产生的 ELF 写死的 interpreter 路径是目标 sysroot 里的（`/lib/ld-linux-riscv64-lp64d.so.1`），qemu-user 在宿主 rootfs 里找不到。`-L /usr/riscv64-linux-gnu` 让 qemu 把这个路径作为虚拟根。
- 证据/复现：`file riscv/smoke/build/hello_rvv` 能看到 `interpreter /lib/ld-linux-riscv64-lp64d.so.1`。
- #qemu #cross-compile
