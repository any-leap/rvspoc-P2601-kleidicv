# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# If changing this image ID, please also change it in .gitlab-ci.yml
FROM registry.gitlab.arm.com/kleidi/kleidicv:24

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ccache \
    clangd-${LLVM_VERSION} \
    gdb-multiarch \
    git-email \
    libemail-valid-perl \
    nano \
    openssh-client \
    qemu-user \
    ripgrep \
    sudo \
    vim \
    zsh

# Needed to run pipx packages originally installed for the root user
RUN chmod +x /root

# Set up directory for volume to persist data between sessions.
RUN mkdir -p /persist && chmod a+rwx /persist

# Add a normal user with sudo rights.
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user
ARG USERNAME
RUN useradd -s /usr/bin/zsh -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER $USERNAME
