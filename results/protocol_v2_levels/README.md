# 固定噪声水平与列校正强度

沿用随包的 NUAA-SIRST 213 张训练原图。比较时，同一条件下的每个方法都读入逐像素相同的图。模型在本地训练过这些原图，改变噪声只检验对噪声变化的敏感性。所有误差均相对加噪前的原图计算。[固定配置](../../configs/denoise_v2/noise_levels.json)

| 条件 | 每列共享噪声的标准差 | 每像素独立噪声的标准差 | 噪声编号 |
| --- | ---: | ---: | ---: |
| 原图 clean | 0 | 0 | 0 |
| 既有 draw0、draw1 | 各图从 0–12 取样 | 各图从 0–4 取样 | 0、1 |
| 新抽样 fresh | 各图从 0–12 取样 | 各图从 0–4 取样 | 20261121 |
| 固定轻度 low | 3 | 1 | 20261121 |
| 固定中度 medium | 6 | 2 | 20261121 |
| 固定重度 high | 12 | 4 | 20261121 |

表中数值均为 8 位灰度级（DN），是输入限幅之前所加噪声的标准差。后三个固定水平与 fresh 对每张图复用同一组标准正态随机数。背景列偏置与像素噪声同时调幅，这套设计检验联合强度变化。

列校正在原尺寸上按行做宽 7 的中值滤波，估计每列的偏置，再按强度减去中心化后的偏置并限幅。主比较固定强度为 1。强度曲线依次取 0、0.5、0.75、1 和 1.25，强度 0 对应直接保留输入。对每张图使用相同强度。主表取图像等权均值，局部对比度先在图内对标注区域等权。新噪声的配对区间使用 5,000 次按原图重采样的百分位 bootstrap，固定种子 20261121。[分级均值](means.csv) · [强度曲线](column_strengths.png) · [逐图记录](per_image.csv) · [逐标注区域记录](per_target.csv) · [新抽样的配对差与区间](fresh_paired_intervals.csv)

在项目根目录运行以下命令。最后一步与主评价表的 1,278 条输入和固定列校正记录逐项核对，需要先生成 `results/protocol_v2/per_image.csv`。GPU 默认取 0，可通过 `SIRST_EVAL_GPU` 指定。加噪输入、零噪声分支和首幅图重复推理的检查均有记录。[数据检查](data_checks.json) · [发布模型重复推理](published_repeat.json) · [配对与跨表核验](verification.json)

```bash
bash results/protocol_v2_levels/run.sh checks
bash results/protocol_v2_levels/run.sh column
for model in published s1 s2; do
    bash results/protocol_v2_levels/run.sh "$model"
done
bash results/protocol_v2_levels/run.sh summary
```
