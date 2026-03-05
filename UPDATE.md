# UPDATE
## 20260305
完成62h训练[wandb曲线图](https://wandb.ai/yy1695651/bfmzero-isaac/runs/xaln7qml)，35h达到86.4M训练步数时候就emd已经达到0.75以下了，最低emd为134M时候的0.737，平均FPS为682

## 20260303 v0.1
在G1 NX 16Gb上运行，修改相应参数并将onnxruntime设置为cuda推理，速度还是太慢，无法使用，后续需要迁移到CPP版本。

在sudo权限下安装好torch, torchvision, onnxruntime，我安装的版本如下：
```bash
onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
torch-2.9.1-cp310-cp310-linux_aarch64.whl
torchvision-0.24.1-cp310-cp310-linux_aarch64.whl
```

安装uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 创建和root共享包的环境
uv venv --system-site-packages --python 3.10
source .venv/bin/activate
uv sync
```

执行推理，目前只测试了`goal.sh`，但是在腰部出现明显的抖动不稳定，后续必须迁移到CPP版本才能使用。
```bash
./rl_policy/goal.sh
```
