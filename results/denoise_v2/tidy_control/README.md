# TIDY 与真实配对图的先导对照

**同一批 NUAA 输入。** 采用作者发布的 TIDY 成品权重，在统一评价的 213 张训练原图、draw0、draw1 上分别推理。TIDY 在本项目的训练次数为零，S1 用这 213 张原图训练过。TIDY 使用原发布版 PyTorch FP32 前向和自动补边。S1 使用已核验的 FP16 贪心推理。两者的有效输入灰度逐像素相同，均在原尺寸上计分，三通道输出限幅后取浮点均值。[作者实现](https://github.com/williamrheeth/TIDY/tree/ebb20eaa615cc3ae3fbad6d25ee48de46e7034c0) · [NUAA 成绩](nuaa.json) · [逐目标结果](nuaa_per_target.csv)

| NUAA 训练图的第一份固定噪声 | 整图 MAE | 背景 RMSE | 背景列残差 | 原生目标对比度绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| 保留带噪输入 | 5.11 | 6.44 | 5.75 | 2.25 |
| 固定列校正 | 2.88 | 3.69 | 2.64 | 1.25 |
| Real-IISR 本地重训 S1 | 3.56 | 5.52 | 1.95 | 18.41 |
| TIDY 作者权重 | 15.12 | 21.52 | 4.83 | 38.90 |

单位为 8 位灰度级，四列均越低越好。两模型默认输出的条纹残留为 1.95 与 4.83，背景 RMSE 也不同。表中差距反映当前权重在该输入上的表现，同等去噪强度下的比较需另有重叠的背景操作点。正式比较应明确权重来源及本地训练预算。[统一评估](../../protocol_v2/summary.json) · [同图核验](verification.json)

逐图再检查背景是否接近。以两模型列残差相差不超过 0.5 DN 作描述性筛选，draw0 有 15/213 张，其目标对比度绝对误差为 S1 的 15.01 和 TIDY 的 34.77 DN，背景 RMSE 分别为 6.56 和 18.74 DN。draw1 有 12/213 张，对比度误差分别为 24.29 和 39.68 DN。同时要求列残差差不超过 1 DN、背景 RMSE 差不超过 2 DN 时，两份抽样均有 0 张符合。以上统计对每张图使用固定模型，筛选规则仅用于描述此次训练图上的可比范围。[匹配数量、阈值与成员](background_overlap.json)

**SCaN-TIR 真实双机配对。** 使用 `300_floor5` 序列中按帧序号等距预选的 32 对 8 位图，取原尺寸水平居中的 512×512 窗口，左右各去 64 列。所有方法共享裁窗。右相机图作输入，左相机图作参考。两机虽已校正几何，逐像素残差仍可能包含视差和灰度映射差异。该序列属于 TIDY 论文的训练侧，32 帧来自同一场景，只作真实数据流程与方向的检查。论文把 `303_floor7` 单列为测试序列。[论文第 IV 节](https://arxiv.org/html/2606.19813v1#S4) · [数据卡与两路图像说明](https://huggingface.co/datasets/williamrhee/SCaN-TIR)

| 32 对真实图，逐帧等权 | 整图 MAE ↓ | 平均有符号灰度差 | PSNR ↑ | SSIM ↑ |
| --- | ---: | ---: | ---: | ---: |
| 右相机原始输入 | 74.88 | +71.63 | 9.83 | 0.1403 |
| 固定列校正 | 74.89 | +71.65 | 9.83 | 0.1419 |
| Real-IISR 本地重训 S1 | 72.46 | +68.08 | 10.08 | 0.1912 |
| TIDY 作者权重 | 57.95 | +54.94 | 11.85 | 0.2163 |

这里可检验对真实成对图的恢复误差。这批帧的左右图原本就有较大亮度差，资料没有提供此次测量所需的点斑尺度真值。逐帧指标保留实际配对文件名及输入、参考、输出的内容哈希。[原始输入](scan_input_frames.csv) · [传统列校正](scan_column_frames.csv) · [S1](scan_s1_frames.csv) · [TIDY](scan_tidy_frames.csv)

### 复算

项目内的 `.runtime/tidy_official` 固定在 `ebb20eaa615cc3ae3fbad6d25ee48de46e7034c0`。将发布权重保存为 `checkpoints/tidy.pth`，SHA-256 为 `37a45473779afcd73abd941e71787caac658325d813342a1bbf787612180906b`。数据包保存为 `datasets/SCaN-TIR/300_floor5.tar.gz`，SHA-256 为 `06a495daec5db7593335138e6fee8e5c8f9b21f4ff7ed83a7070a094523f533b`。[官方权重](https://drive.google.com/file/d/1C0qBxbgoZjGgQ6WFHB5QBsyl6ZQD53h2/view) · [固定版本数据包](https://huggingface.co/datasets/williamrhee/SCaN-TIR/blob/d51ae1a5ef1c99938929d74060e6b65138e2a7f5/SCaN-TIR/300_floor5.tar.gz)

```bash
bash results/denoise_v2/tidy_control/run.sh nuaa
for method in input column s1 tidy; do
    bash results/denoise_v2/tidy_control/run.sh scan --method "$method"
done
```
