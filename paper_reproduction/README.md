# Real-IISR 论文复现

本次在 FLIR-IISR 上完成了公开数据和官方推理样例核验。项目当前已转为使用自己的数据集重新训练原模型。论文第 4.1 节报告使用 VQVAE 与 VAR 预训练参数、1,192 对训练图和 265 对测试图、batch size 4、AdamW、学习率 0.00005、权重衰减 0.05 和 10,000 次更新。代码仓库提供的两个初始化文件名是 `VQVAE.pth` 和 `VARSR.pth`。[论文](https://arxiv.org/html/2603.04745v1#S4.SS1) · [作者训练说明](https://github.com/JZD151/Real-IISR#-training) · [本地训练配置](config.json)

## 已核验的数据及样例

- 已校验并解包作者的 FLIR-IISR 发布包。`HR/`、`LR/` 和 `LR_4x/` 各有 1,457 张，同名一一对应。归档 SHA-256 为 `bbf265e50344c4151b45f21a4148242d956921e116d5efc10abdcd0c7f8c08d4`。原训练加载器读取全分辨率 LR 与 HR，对齐裁块，再对 LR 做 4 倍下采样。[发布清单](flir_release_inventory.json) · [作者答复](https://github.com/JZD151/Real-IISR/issues/5#issuecomment-4561339974)
- 数据包中的 `scene.json` 与 `degradation.json` 分别给出场景和退化标签。当前标签有 1,414 对光学模糊和 43 对运动模糊；论文第 3.4 节报告 1,305 对和 152 对。对公开历史版本的检查也没有找到原论文训练／测试 ID 或 Set5／Set15 的具体成员。[发布历史核验](sources/flir_dataset_release_history.json) · [划分讨论](https://github.com/JZD151/Real-IISR/issues/4) · [标签讨论](https://github.com/JZD151/Real-IISR/issues/8)
- 已在作者仓库提供的七对示例上运行其 `test.py` 和发布的成品权重。这七对均为 128×128 输入和 512×512 参考，完整保存了 7 张输出。这里按 RGB 全幅、逐图等权计算 PSNR 25.799 dB、SSIM 0.8355。这是官方入口的可运行性检查，其数值仅适用于这七对示例。[逐图结果](upstream_examples/metrics.json) · [首张输出](upstream_examples/testset/VARPrediction/000009_0.png) · [运行日志](upstream_examples/official_test.log)

## 训练状态

FLIR 正式训练为 **0/10,000 次更新**。自行划分 FLIR 的方案已经撤回。公开材料给出训练与测试的数量，仍缺逐图成员。CPU 检查验证了作者加载器的两个样本分支和损失函数。当前本地实验使用项目已有的 SIRST 训练清单。[CPU 核验](training_cpu_check.json) · [本地训练记录](../results/local_retrain/updates.jsonl)
