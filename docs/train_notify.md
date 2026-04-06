# 训练注意

## 训练环境

```bash
--- CPU Info ---
Model name:                           AMD EPYC 9654 96-Core Processor
Thread(s) per core:                   2
Core(s) per socket:                   96
Socket(s):                            1

--- GPU Info ---
NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 570.207, 97887 MiB
```

## Docker 环境配置

### 基于 Ubuntu 镜像从零安装

使用Docker配置环境，容器使用`docker pull wtyyy/ubuntu:22.04`，需额外安装`apt install -y libgomp1 libglu1`

再使用当前的[pyproject.toml](./pyproject.toml)配置，相对于上游仓库做了以下修改：

1. **Python 版本**：`requires-python` 从 `==3.10.*` 改为 `==3.11.*`
2. **Isaac Sim/Lab 升级**：
   - `isaaclab` 从 `==2.0.2` 升级到 `==2.3.2.post1`
   - `isaacsim` 从 `==4.5.0` 升级到 `==5.1.0`
3. **PyTorch 版本锁定**：`torch` 从 `>=2.5.0` 改为 `==2.7.0`
4. **新增 PyTorch CUDA 12.8 索引源**：添加 `pytorch-cu128` index（`https://download.pytorch.org/whl/cu128`），并将 `torch`、`torchvision`、`torchaudio` 指向该源，因为RTX 6000必须用CUDA 12.8以上版本
5. **tyro 放宽版本约束**：从 `>=0.9.18` 改为无版本限制
6. **新增 `wandb` 依赖**
7. **uv 配置调整**：`dev-dependencies` 从 `[tool.uv]` 迁移到 `[dependency-groups]`，`[tool.uv]` 下保留 `index-strategy = "unsafe-best-match"`

### 基于 IsaacLab 镜像高效安装

直接使用`docker pull wtyyy/isaaclab:2.3.2.post1`，启动容器
```bash
docker run -it --name ${USER}-isaaclab \
  -e DEFAULT_UID="$(id -u)" \
  -e DEFAULT_GID="$(id -g)" \
  -e DISPLAY \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e "__NV_PRIME_RENDER_OFFLOAD=1" \
  -e "__GLX_VENDOR_LIBRARY_NAME=nvidia" \
  -v /etc/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json:ro \
  --device /dev/input \
  --group-add $(getent group input | cut -d: -f3) \
  --net=host \
  -v /path/to/your/BFM-Zero:/home/user/bfm_zero \
  -v ${HOME}/isaaclab_docker/.cache/ov:/home/user/.cache/ov \
  -v ${HOME}/isaaclab_docker/.nvidia-omniverse:/home/user/.nvidia-omniverse \
  wtyyy/isaaclab:2.3.2.post1 zsh
```

将上述`/path/to/your/BFM-Zero`替换为你本地的BFM-Zero代码路径，容器内会自动映射到`/home/user/bfm_zero`，进入容器后安装依赖
```bash
cd /home/user/bfm_zero
uv pip install -e .
```

完成安装，仅需安装少量包即可，且自动使用用户的文件修改权限，便于文件处理

## 开始训练
打开wandb日志记录，并设置为默认的用户名保存
```bash
python -m humanoidverse.train
```

## 训练结果

模型会自动保存在`{work_dir}/checkpoint/`下，默认为`results/bfmzero-isaac/checkpoint/`

```bash
results/bfmzero-isaac/checkpoint/
├── <agent 模型文件>          # agent.save() 保存的模型权重
├── buffers/train/            # replay buffer（若 checkpoint_buffer=True）
└── train_status.json         # 记录当前训练步数 {"time": xxx}
```
