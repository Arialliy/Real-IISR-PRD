# Real-IISR 先导环境

2026-11-21 的统一指标、S1/S2 权重关系、短程重放及人工点斑和真实配对实验已归档。以下较早条目保留各自的输入范围、采样器和指标定义。跨方法比较采用本轮固定输入与原尺寸浮点计分。[统一实测报告](results/denoise_v2/README.md) · [模型与权重 SHA-256](results/denoise_v2/experiment_registry.json) · [统一评价运行与核验](results/protocol_v2/README.md) · [TIDY 和 SCaN-TIR 的版本](results/denoise_v2/tidy_control/README.md)

## 固定版本

| 项目 | 本次配置 |
| --- | --- |
| 上游源码 | `vendor/Real-IISR`，提交 `70355f7e9d6b849272d429d12e854f5890b89ae2`。[官方提交](https://github.com/JZD151/Real-IISR/commit/70355f7e9d6b849272d429d12e854f5890b89ae2) |
| 官方发布权重 | `checkpoints/Real-IISR.pth`，14,246,400,682 字节。[固定版本下载地址](https://huggingface.co/datasets/yuanzsz/FLIR-IISR/resolve/db77fe45d0b103cfc1f435d550446045041b3043/Real-IISR.pth) |
| 权重 SHA-256 | `fe1c2600ec6aeb18bfa674f7726a1dcd0aec45d9fbba25c7f1b9fe266b3ef9a9` |
| 权重结构 | `depth=24`，尺度序列 `1,2,3,4,6,9,13,18,24,32`，末级对应 512×512 输入。参数从 checkpoint 的 `args` 和 `trainer.config` 读取。 |
| GPU | NVIDIA GeForce RTX 5090，单卡 32,607 MiB，驱动 580.178.04 |
| 本机容器镜像 | `deeppro-official-eval:20261009`，镜像 ID `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92` |
| 容器运行时 | Python 3.11.12，PyTorch 2.7.0+cu128，CUDA 12.8，NumPy 2.2.5，Pillow 11.0.0，SciPy 1.16.2 |
| 项目内依赖 | `.runtime/python` 中的 `huggingface-hub==0.25.0`。该版本取自上游依赖文件；容器的 PyTorch 版本适配本机 RTX 5090。 |

## 官方入口直接处理本地 SIRST 原图

使用本地随包训练清单的 213 张原始 PNG 运行作者的 `test.py` 与发布权重。官方加载器枚举 `HR/`，并从 `LR/` 读取模型输入，因此两个目录都按训练清单顺序链接到同一份原图；没有独立的干净参考。此运行未加人工噪声或更新参数。官方流程将这批输入缩放到 512×512，执行推理和 AdaIN 校正后保存 RGB PNG。已逐图核对全部 213 个 ID 并成功解码所有输出。下列项目内脚本记录了在容器中运行原版入口的命令与工作目录。[运行脚本](results/official_sirst_train/run.sh) · [输入清单](results/official_sirst_train/input_manifest.json) · [运行日志](results/official_sirst_train/official_entry.log) · [示例输出](results/official_sirst_train/testset/VARPrediction/0001_Misc_119.png)

## 本机复现

在项目根目录执行。三个检查均使用随包的 SIRST 训练清单。原始模型的固定 SHA-256 会在权重加载前核对。

```bash
run_prd() {
    docker run --rm --gpus device=0 --user "$(id -u):$(id -g)" \
        --network none --entrypoint python \
        -v "$PWD:/project" -w /project \
        -e PYTHONPATH=/project/.runtime/python -e PYTHONDONTWRITEBYTECODE=1 \
        -e PYTHONUNBUFFERED=1 \
        deeppro-official-eval:20261009 "$@"
}

run_prd check_training_pairs.py --output results/training_pair_checks.json
run_prd probe_backbone.py --samples 8 --seed 0 --swap-condition --official-example \
    --output results/representation_probe.json
run_prd probe_heatmap_padding.py --trace results/representation_probe.json \
    --output results/heatmap_padding_probe.json
```

本轮已完成全部 213 张 SIRST 训练图的数据与损失检查，并对训练清单前 8 张图执行固定权重的模型探针。原始 tokenizer 使用 FP32，预测与解码使用 CUDA FP16 autocast，指标使用 FP32。上游仓库的一张示例图也完成了前向检查。结果位于 [数据检查](results/training_pair_checks.json) 和 [模型探针](results/representation_probe.json)。

样本生成种子为 0。推理时单独将上游采样器的 `g_seed` 设为 0，每次前向使用相同起始状态。FP16 下最大 logits 可能并列，`top_k=1` 会保留并列项。第一版诊断只在进程开始时设置随机种子，保留在 [初步记录](results/representation_probe_unfixed_sampling.json)。改为逐次固定种子后，预测路径的注入区平均绝对误差由 19.59 降至 1.98 DN；模型权重与八张输入图相同。热响应补边复核使用逐次固定种子的记录。

## 热响应补边复核

对八张训练图分别使用零灰度或有效区均值补边，在同一模型上只切换热响应统计范围。表中是八幅图等权的均值。

| 热响应统计范围 | 两种补边的有效区热响应平均绝对差（0–1） | 两种补边的有效区输出平均绝对差（DN） | 向内去除 16 像素后的输出平均绝对差（DN） | 零补边背景 RMSE（DN） | 均值补边背景 RMSE（DN） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 整个输入 | 0.495 | 13.47 | 12.05 | 22.36 | 17.28 |
| 仅有效像素 | 0.000 | 13.77 | 12.35 | 22.46 | 17.11 |

两种模式下，相同输入重复硬推理的最大差均为 0 DN。有效区热响应的填充值依赖消失后，最终输出仍随填充值而变化。这次干预没有区分其他条件分支或解码路径的影响。各图数值见[热响应补边复核](results/heatmap_padding_probe.json)。

## 本地重训原结构的初始化

作者的仓库用 `VQVAE.pth` 提供图像编码器、码本和解码器的初值，用 `VARSR.pth` 提供可复用的自回归预测参数。正式本地基线采用这两份起始文件。已核对两个文件的完整 SHA-256、大小和实际可读字段。研究使用本地 NUAA-SIRST 的 213 张训练图，生成同尺寸带噪输入，以加噪前图像作为参考。论文报告作者在 FLIR-IISR 上进行了 10,000 次迭代；本地采用公开 README 的 20 轮示例预算，记录实际更新次数。[初始化记录](results/local_retrain/initialization.json) · [作者训练说明](vendor/Real-IISR/README.md) · [论文第 4.1 节](https://arxiv.org/html/2603.04745v1#S4.SS1)

## 成品权重迁移先导中的 S1 共同预热

在本地 SIRST 训练清单的 213 张原图上完成一次完整预热。数据种子为 0，epoch 和 draw 均为 0。使用固定置换种子 20261120，每个优化步依次对同一张原图的两支样本反传并取平均，共 213 次更新。输入保持原像素尺度，工作网格为 512×512，两支都用 0 DN 补边。

训练参数共 53,588,545 个，来自原 VAR 最后一个块、输出头及其归一化调制、`cb_cond` 和条件码本的 U、V、gate。其他参数冻结，BatchNorm 和 DropPath 使用评估模式。参考图编码使用 FP32；VAR 用 FP16 autocast 和参考 token 前缀计算交叉熵；硬 argmax 后原 CAC 和解码器使用 FP32。交叉熵按各尺度有效区域覆盖率加权，像素均方误差只统计有效像素，顺序项按两端均有效的 8×8 邻接块归一化。三项系数依次为 1、0.2 和 0.8。此阶段图像损失沿 CAC 反传，VAR 的更新来自交叉熵。

使用上文定义的 `run_prd`。预热命令将结果写入当前项目中的 `checkpoints/s1_warmup.pt` 与 `results/s1_warmup.json`。需要独立复跑时，用 `--outdir` 指定新的项目内子目录，以保留存档。

```bash
run_prd train_warmup.py --steps 213 --seed 0 --schedule-seed 20261120
run_prd probe_backbone.py --warmup checkpoints/s1_warmup.pt --samples 8 --seed 0 --swap-condition
```

该轮 GPU 运算及终点保存共约 92 秒，峰值已分配显存为 11.04 GiB。更新文件保存 23 个参数张量的终点绝对值和对应官方权重哈希。各步损失、梯度、参数变化与完整训练图 ID 见[预热记录](results/s1_warmup.json)。

## 成品权重迁移先导中的 S2 自由前缀训练

S1 在两次训练图合成抽样上的硬推理背景误差都上升，因此增加一次 16 步的诊断性训练。用置换种子 20261121 从原有 213 张 SIRST 训练图中预选 16 张。数据种子为 0，epoch 为 0，draw 为 1。每张原图的两支按顺序反传、梯度取平均，完成一次优化更新。S2 从完整 S1 终点加载同一组 23 个可训练张量，重新初始化 AdamW，沿用 S1 的学习率、损失系数和 0 DN 补边。

S2 在每一级用自身预测的硬 argmax token 生成下一级前缀。参考图 token 只用于交叉熵监督。硬前向保留原始索引查表，反向使用温度 1 的 softmax 直通估计，CFG 系数为 1。参考图量化用 FP32，模型自由生成与解码用 FP16 autocast，输出转 FP32 后计算损失。训练前在三张 S1 训练图上复核了与原缓存路径采用相同 argmax 规则时的精确一致性。性能评估另用上游固定种子的采样器，FP16 并列最大值的选择规则与训练所用 argmax 有区别。

使用上文的 `run_prd`。执行前需有本项目的 S1 终点权重与对应的一致性记录。新一轮独立复跑用 `--outdir` 写入项目内新的输出目录。

```bash
run_prd train_s2_pilot.py
```

16 次更新耗时 157.4 秒，峰值已分配显存为 11.17 GiB。早期尝试的损失缩放 32768 在首次更新前出现非有限梯度。完整运行从 4096 起步，16 步均有限，经过范数上限 2 的裁剪后各步 CAC 梯度均大于零。23 个参数张量均相对 S1 有变化。更新文件 `checkpoints/s2_common_pilot.pt` 的 SHA-256 为 `40e8b180968de648dabc16c6a7329cd58a4ed2d0190b1de93efe0efa5c5c1491`。样本 ID、父权重哈希、逐步损失和梯度见[S2 训练记录](results/s2_common_pilot.json)。

## 训练图上的硬推理对照

在 S1 与 S2 之间执行下面的第一条命令可生成 S2 所需的一致性记录。评价命令按冻结模型、S1、S2 的顺序运行，上一步保存的逐图输入与权重哈希用于核验下一步的比较。两次抽样都使用训练清单内相同的 213 张原图。

```bash
run_prd verify_rollout.py --warmup checkpoints/s1_warmup.pt --precision fp16 --backward --samples 3 --output results/rollout_check_s1_fp16.json
for draw in 0 1; do
    run_prd evaluate_warmup.py --draw-id "$draw"
    run_prd evaluate_warmup.py --draw-id "$draw" --warmup checkpoints/s1_warmup.pt
    run_prd evaluate_warmup.py --draw-id "$draw" --warmup checkpoints/s1_warmup.pt --pilot checkpoints/s2_common_pilot.pt
done
run_prd evaluate_warmup.py --argmax --draw-id 0 --warmup checkpoints/s1_warmup.pt --pilot checkpoints/s2_common_pilot.pt
run_prd check_s1_hard_sampler.py
run_prd check_column_baseline.py
run_prd probe_s2_token_sources.py
```

固定参数去列条纹仅读取有效范围内的带噪像素。原图和掩码用于计分，各方法的输入哈希、参考和评价区域逐图对齐。第 10.1 节汇总比较结果。[研究方案](Real-IISR_Weak_Target_Denoising_Research_Plan.md)
