# SIRST 原网络的两个训练终点对照

评估使用本地 NUAA-SIRST 的 213 张训练图。S1 是同尺寸去噪训练的第 20 轮终点。S2 从该权重开始，按预先固定的日程做一次全训练集自由前缀适配。三种输入分别是固定的两次腐蚀抽样和未额外加噪的原图。每个模型都逐张自由推理，不读取目标掩膜；掩膜用于离线评价。[原训练列表](../../datasets/NUAA-SIRST/img_idx/train_NUAA-SIRST.txt) · [S2 训练登记](../s2_common/preregistration.json)

主结果使用每尺度确定性的首个最大值索引。另列作者实现的固定种子 top-1 采样，便于检查并列最大值的处理是否改变结果。两种方式都使用作者的缓存自回归前向、CFG 1、FP16；显式关闭矩阵运算及 cuDNN 的 TF32。取网络返回的 [0,1] RGB 三通道浮点均值，随后在原尺寸有效区域评分。指标的灰度单位为 0–255 DN。所有原图均小于 512×512，评价时裁掉输入居中填充的区域，完整保留原图视野。

## 指标

每张图独立计算指标，再对合格图像等权平均。目标对比度在图内先对合格实例等权平均。

| 字段 | 计算口径 |
| --- | --- |
| `whole_mae_dn`、`whole_psnr_db` | 原图全部有效像素的绝对误差均值、峰值 255 的 PSNR。完美重建以 JSON null 表示 +∞，汇总中单列数目。 |
| `weighted_background_column_residual_dn` | 输出减参考形成误差图；在排除原生目标及 2 像素外扩的背景中，各列平均误差的加权标准差，权重为该列有效背景像素数。 |
| `plan_background_rmse_dn`、`plan_background_mean_error_dn` | 在上述背景上的均方根误差与有符号误差均值。 |
| `native_median_ring_contrast_bias_dn`、`native_median_ring_contrast_abs_error_dn` | 原生八连通目标的核心均值减去周围背景环中位数；分别统计输出对比度减参考对比度的有符号偏差及其绝对值。背景环使用 2 像素方形外扩，排除所有目标及填充区。 |
| `native_target_mae_dn` | 原图全部原生目标像素上的平均绝对误差。 |
| `valid_mae_dn`、`background_rmse_dn`、`background_column_residual_dn`、`native_contrast_abs_error_dn` | 保留旧版指标。其背景掩膜还排除校准点斑预留窗口，列指标使用至少 8 个背景像素的列等权计算，局部对比度使用环带均值。 |

## 产物与重做

每个端点和输入条件单独输出逐图 JSON，文件名称由 `s1` 或 `s2` 与 `draw0`、`draw1` 或 `drawclean` 组合。对应的 `*_float/` 目录按原图 ID 保存主采样器的原生高度×宽度 float32 灰度数组，范围为 [0,1]。逐图字段记录数组路径及 SHA-256。两端、三种条件均完成后，`comparison.json` 给出配对差值；生成前逐图校验数组、输入和参考的一致性。[评估程序](../../evaluate_sirst_endpoints.py) · [受控数值样例](../local_retrain/metric_checks.json)

从本目录运行 `SIRST_EVAL_GPU=1 bash run.sh --endpoint s1 --draw 0` 可复算一组。替换端点和抽样即可运行其他组，运行 `bash run.sh --compare` 在 CPU 上汇总六组记录。这些训练来源可用于诊断损失与目标保真之间的关系，独立测试集的泛化结论需要另外评价。
