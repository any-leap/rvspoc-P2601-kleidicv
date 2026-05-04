# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Dev image for RISC-V backend work on KleidiCV.
# Multi-arch: works natively on both x86_64 and arm64 hosts (Mac M-series OK).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Use USTC mirror — ports.ubuntu.com is unbearably slow from CN networks.
# Ubuntu 24.04 (noble) ships sources in /etc/apt/sources.list.d/ubuntu.sources.
RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.ustc.edu.cn/ubuntu-ports|g; s|http://archive.ubuntu.com/ubuntu|http://mirrors.ustc.edu.cn/ubuntu|g' \
      /etc/apt/sources.list.d/ubuntu.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      ccache \
      cmake \
      curl \
      file \
      g++-riscv64-linux-gnu \
      gcc-riscv64-linux-gnu \
      git \
      gnupg \
      less \
      libc6-dev-riscv64-cross \
      ninja-build \
      pkg-config \
      python3 \
      python3-pip \
      qemu-user \
      qemu-user-static \
      vim-tiny \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Sanity: gcc 13.x in noble ships RVV 1.0 intrinsics.
RUN riscv64-linux-gnu-gcc --version | head -1 \
    && qemu-riscv64 --version | head -1

WORKDIR /workspace
