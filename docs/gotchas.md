# Gotchas

下次还可能再踩的坑及恢复步骤。新条目用递增 `GOT-NNN`。

## GOT-001 [docker/network] 国内构建 ubuntu:24.04 镜像极慢

### 症状
`apt-get install gcc-13-riscv64-linux-gnu` 单个包 20 分钟下不完，buildx 在非 TTY 模式下日志为空看不到进度，疑似卡死。

### 原因
`ports.ubuntu.com` 在国内不通畅。

### 恢复步骤
镜像里改 USTC（http，不要 https，否则 ca-certificates 没装会先失败）：

```dockerfile
RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.ustc.edu.cn/ubuntu-ports|g; s|http://archive.ubuntu.com/ubuntu|http://mirrors.ustc.edu.cn/ubuntu|g' \
      /etc/apt/sources.list.d/ubuntu.sources
```

跑 `docker build` 时一定加 `--progress=plain`，否则非 TTY 下看不到任何日志。

#docker #cn-network

## GOT-002 [qemu/cross] qemu-user 跑不动交叉编译产物

### 症状
```
qemu-riscv64: Could not open '/lib/ld-linux-riscv64-lp64d.so.1': No such file or directory
```

### 原因
qemu-user 没有 sysroot，找不到 RISC-V 的动态链接器。

### 恢复步骤
启动 qemu 时加 `-L /usr/riscv64-linux-gnu`（路径是 ubuntu 的 cross 包安装位置）。

#qemu
