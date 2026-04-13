# 导出模型配置

支持三种导出

## tracking

指定模仿重定向轨迹中的轨迹id导出latent

```bash
uv run python -m humanoidverse.tracking_inference \
    --model_folder /mnt/data/Coding/BFM-Zero/results/bfmzero-isaac_20260411_101302 \
    --data_path humanoidverse/data/dailylife_data_v1_bfmzero.pkl \
    --no-headless \
    --save_mp4 \
    --motion-list 0 28 29 43 52 53 62 67 77 104
```

## reward

指定奖励函数

```bash
uv run python -m humanoidverse.reward_inference \
    --model_folder /mnt/data/Coding/BFM-Zero/results/bfmzero-isaac_20260411_101302 \
    --save_mp4 
```

## goal

执行目标状态

```bash
uv run python -m humanoidverse.goal_inference \
    --model_folder /mnt/data/Coding/BFM-Zero/results/bfmzero-isaac_20260411_101302 \
    --data_path humanoidverse/data/lafan_29dof.pkl \
    --save_mp4
```
