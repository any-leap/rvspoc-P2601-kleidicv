# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
      ca-certificates \
      ccache \
      cmake \
      curl \
      doxygen \
      electric-fence \
      g++-aarch64-linux-gnu \
      gcc-aarch64-linux-gnu \
      git \
      gpg \
      graphviz \
      lcov \
      less \
      libnuma1 \
      liburing2 \
      lsb-release \
      ninja-build \
      patch \
      pipx \
      python3 \
      shellcheck \
      software-properties-common \
      wget

# Install latest version of most LLVM tools except clang-format and clang-tidy
# because we prioritise stability when it comes to how we format our code and
# which cland-tidy checks we expect.
ENV LLVM_VERSION=23
ENV CLANG_FORMAT_VERSION=20
ENV CLANG_TIDY_VERSION=20

RUN wget -q https://apt.llvm.org/llvm-snapshot.gpg.key -O /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "8b2a587ffd672c4687e7581dad4b2f6c1bb2ad6b480cd9771ba2ff48e0b8c75d /etc/apt/trusted.gpg.d/apt.llvm.org.asc" | sha256sum -c - \
    && . /etc/os-release \
    && add-apt-repository "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME} main" \
    && add-apt-repository "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-${CLANG_FORMAT_VERSION} main" \
    && add-apt-repository "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-${CLANG_TIDY_VERSION} main" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        clang-${LLVM_VERSION} \
        clang-format-${CLANG_FORMAT_VERSION} \
        clang-tidy-${CLANG_TIDY_VERSION} \
        clang-tools-${LLVM_VERSION} \
        libclang-rt-${LLVM_VERSION}-dev \
        lld-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}

# Populate the contents of libclang-rt-dev:arm64 on an x86 host.
# Package installation fails, so instead the package is downloaded and extracted into the root filesystem.
RUN if [ $(dpkg --print-architecture) = amd64 ]; then \
    dpkg --add-architecture arm64 \
    && (apt-get -y update || true )\
    && cd /tmp \
    && apt-get -y download libclang-rt-${LLVM_VERSION}-dev:arm64 \
    && dpkg-deb -x libclang-rt*.deb / \
    && rm -rf libclang-rt* \
    && dpkg --remove-architecture arm64 \
    ; fi

RUN pipx install cpplint==1.6.1
RUN pipx install gcovr==6.0
RUN pipx install reuse==3.0.0
RUN pipx install lcov-cobertura==2.1.1

# This tool cannot be downloaded automatically.
# The downloaded archive shall be put next to this Dockerfile when building.
# Please download Arm Instruction Emulator (25.0, Ubuntu) from the link below, and put the archive here when building.
# https://developer.arm.com/Tools%20and%20Software/Arm%20Instruction%20Emulator
ENV ARMIE_VERSION=arm-instruction-emulator_25.0_Ubuntu-22.04
WORKDIR /tmp
COPY ARMIE-BN-UBUNT-r25p0-00rel0.tgz ./armie.tgz

RUN if [ $(dpkg --print-architecture) = arm64 ]; then \
    md5sum -c <<'EOF' \
    && tar xzf ./armie.tgz \
    && chmod +x ./${ARMIE_VERSION}/${ARMIE_VERSION}.sh \
    && ./${ARMIE_VERSION}/${ARMIE_VERSION}.sh -a \
    && rm ./${ARMIE_VERSION}/${ARMIE_VERSION}.sh \
    && mv /opt/arm/arm-instruction-emulator-25.0_Generic-AArch64_Ubuntu-22.04_aarch64-linux /opt/arm/armie \
    ; fi
7c63fb51097a7c2b81c315c553df2ff5 ./armie.tgz
EOF
RUN rm -rf ./${ARMIE_VERSION} ./armie.tgz
ENV PATH=${PATH}:/usr/lib/llvm-${LLVM_VERSION}/bin:/root/.local/bin:/opt/arm/armie/bin64
ENV CC=clang-${LLVM_VERSION}
ENV CXX=clang++-${LLVM_VERSION}

ARG OPENCV_VERSION=4.13.0
RUN wget \
  https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz \
  -O /opt/opencv-${OPENCV_VERSION}.tar.gz
RUN wget \
  https://github.com/opencv/opencv_extra/archive/refs/tags/${OPENCV_VERSION}.tar.gz \
  -O /opt/opencv-extra-${OPENCV_VERSION}.tar.gz
