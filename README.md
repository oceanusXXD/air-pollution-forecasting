# air-pollution-forecasting# air-pollution-forecasting



Air pollution forecasting project for predicting air quality using machine learning models.Air pollution forecasting project for predicting air quality using machine learning models.

## Classification results analysis
```
在当前数据集与任务设定下（CO 水平离散为 three-class，基于聚合后的数值特征）：

纯 XGBoost 与 naive baseline 已经提供了较强、稳定的性能：

在多个预测时长（h+1, h+6, h+12, h+24）上，XGB-only 的 Test F1_macro 一般在 0.55–0.57 左右；
某些短期 horizon（如 h+1）上，naive baseline 甚至优于 XGB 和所有深度模型尝试。
引入 DeepGBM（XGB 叶子索引经哈希 + EmbeddingBag + 小型 MLP）后：

Deep 部分在训练集上的 F1_macro 接近 1.0，明显过拟合；
在验证集和测试集上，Deep-only 模型的表现普遍低于 XGB-only；
即便对 Deep 与 XGB 的输出概率做加权融合，并在验证集上自动搜索最优融合权重，Deep+XGB 在测试集上的表现仍未能稳定超过 XGB-only，多数情况下略低于或仅接近 XGB-only。
结论：

在当前数据、特征与标签设计下，XGB / naive 已经挖掘了大部分可泛化信号，DeepGBM 引入的额外模型容量主要用于拟合噪声，未带来可靠的泛化收益；
因此，本项目的主线模型选择保持为 XGB / naive，不启用 DeepGBM 作为生产或报告中的默认方案；
DeepGBM 代码保留在仓库中，仅作为已尝试过的实验路线和后续方法探索（例如改为回归任务、引入显式时序模型等）的参考基础。
```