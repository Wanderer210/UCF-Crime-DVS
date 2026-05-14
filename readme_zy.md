添加的 msf_stdp_version1/ 不是直接修改原论文 MSF 主网络，而是新增了一个 STDP 无监督特征提取模块。它的作用是：
Event Frames→STDP Encoder→STDP Features→MSF
也就是说，它先用 STDP 从事件帧中学习时空脉冲特征，再把这些特征保存成 .npy，之后再送入原论文的 MSF 异常检测网络。
验证一个问题：
不依赖 Spikingformer 预训练特征，仅用 STDP 无监督突触可塑性学习事件流特征，是否也能支撑 MSF 完成异常检测？

原论文的流程是：
Event Frames→Spikingformer→MSF→Anomaly Score
我添加的 Version 1 改成：
Event Frames→STDP Encoder→MSF→Anomaly Score
所以它主要改进的是 特征提取阶段。

stdp_encoder.py：核心 STDP 编码器
STDPFeatureEncoder：把脉冲变成 MSF 可用特征
event_frame_dataset.py：读取事件帧数据
train_stdp_encoder.py：无监督预训练 STDP 编码器
extract_stdp_features.py：提取 STDP 特征
inspect_stdp_features.py：检查特征是否正确
minimal_msf_adapter.py：给 MSF 读取 STDP 特征用

1. 第一步：STDP 无监督预训练
python msf_stdp_version1/train_stdp_encoder.py \
  --event_frame_root /path/to/EventFrame \
  --normal_keyword Normal \
  --image_size 128 \
  --epochs 1 \
  --max_frames_per_video 128 \
  --save_path checkpoints/stdp_encoder.pth

2. 第二步：提取所有视频特征
python msf_stdp_version1/extract_stdp_features.py \
  --event_frame_root /path/to/EventFrame \
  --ckpt checkpoints/stdp_encoder.pth \
  --out_root features_stdp \
  --image_size 128

3. 第三步：用 MSF 训练异常检测