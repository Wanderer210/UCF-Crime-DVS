# Version 1: MSF + STDP 无监督预训练编码器

目标：用 STDP 无监督编码器替代/增强原论文的 Spikingformer 特征提取器，生成与 MSF 兼容的 `T x D` clip-level 特征。

推荐流程：

```bash
# 1) 克隆原论文代码
git clone https://github.com/YBQian-Roy/UCF-Crime-DVS.git
cd UCF-Crime-DVS

# 2) 把本目录复制进仓库
cp -r /path/to/msf_stdp_version1 ./stdp_version1

# 3) 只用正常视频事件帧进行 STDP 无监督预训练
python stdp_version1/train_stdp_encoder.py \
  --event_frame_root /path/to/UCF-Crime-DVS/EventFrame \
  --normal_keyword Normal \
  --image_size 128 \
  --epochs 1 \
  --max_frames_per_video 128 \
  --save_path checkpoints/stdp_encoder.pth

# 4) 提取 STDP 特征，输出每个视频一个 .npy，形状为 [T, 512]
python stdp_version1/extract_stdp_features.py \
  --event_frame_root /path/to/UCF-Crime-DVS/EventFrame \
  --ckpt checkpoints/stdp_encoder.pth \
  --out_root features_stdp \
  --image_size 128

# 5) 用原论文的 MSF 训练脚本读取 features_stdp
# 如果 main.py 支持 feature_root/input_dim 参数：
python main.py --feature_root features_stdp --feature_dim 512
```

如果原论文 `main.py` 没有暴露 `--feature_root`，就把原来的预训练特征目录替换为 `features_stdp`，或在数据读取处把特征路径改成 `features_stdp`。

## 设计说明

- `STDPFeatureEncoder` 输出维度默认是 `c1 + c2 = 128 + 384 = 512`。
- 第 1 层学习低级事件边缘/局部运动变化。
- 第 2 层学习更抽象的时空事件组合。
- 每一层使用 Conv-LIF + pair-based STDP：
  - pre-before-post 的相关性增强权重；
  - post-trace 与当前 pre 的相关性抑制权重；
  - 权重被 clamp 并按卷积核归一化，防止无监督学习发散。
- 训练阶段建议只用正常视频，让编码器形成“正常事件动力学原型”。
- 提取出的 `.npy` 特征可以直接送入 MSF，作为 Spikingformer 特征的替代输入。
