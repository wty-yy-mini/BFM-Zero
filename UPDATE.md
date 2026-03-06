# UPDATE

## 20260306 v0.1

训练配置切换为使用 `robot=g1/g1_29dof_mode15`，这个配置和unitree_rl_lab完全一致，并且符合mode15的hip.{pitch, roll}齿轮比均为22.5的要求（原来使用的hard_waist锁腰的版本，可能和腰部剧烈震荡有关）

## 20260303

训练占用显存47.8Gb，平均689FPS（环境步数/sec），训练20M环境步数，eval/tracking_eval/emd能降到0.92247，目标要求降到0.75以下，预计50M或100M能达到要求，用时20.2h或40.4h完成

## 20260302

### pyproject.toml

1. **Python 版本**：`requires-python` 从 `==3.10.*` → `==3.11.*`
2. **Isaac Sim/Lab 升级**：
   - `isaaclab` 从 `==2.0.2` → `==2.3.2.post1`
   - `isaacsim` 从 `==4.5.0` → `==5.1.0`
3. **PyTorch 版本锁定**：`torch` 从 `>=2.5.0` → `==2.7.0`
4. **新增 PyTorch CUDA 12.8 索引源**：添加 `pytorch-cu128` index（`https://download.pytorch.org/whl/cu128`），并将 `torch`、`torchvision`、`torchaudio` 指向该源
5. **tyro 放宽版本**：从 `>=0.9.18` → 无版本限制
6. **新增 `wandb` 依赖**
7. **uv 配置调整**：`dev-dependencies` 从 `[tool.uv]` 迁移到 `[dependency-groups]`

### humanoidverse/train.py

1. **新增轻量进度日志**：添加 `wandb_log_step_interval` 配置（默认 10000 步），每隔该步数记录并打印 `env_steps`、`buffer_size`、`elapsed_minutes`、`rollout_FPS`。开启 wandb 时同步到 `progress/` 组。解决了训练初期长时间无日志输出的问题
2. **work_dir 添加时间戳**：`results/bfmzero-isaac` → `results/bfmzero-isaac_{YYYYMMDD_HHMMSS}`，避免多次训练目录覆盖
3. **wandb 默认开启**：`use_wandb` 从 `False` → `True`
4. **wandb_ename 清空**：从 `'yitangl'` → `''`（使用 wandb login 的默认用户）
