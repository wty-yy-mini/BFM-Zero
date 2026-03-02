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

## 环境配置

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
