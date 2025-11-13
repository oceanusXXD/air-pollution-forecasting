# CPU 训练超参数优化指南

## 🎯 针对 CPU + 小 Batch Size (16/32) 的优化

当使用 CPU 训练且 batch size 较小时,需要调整超参数以保持性能并防止过拟合。

---

## 📊 FT-Transformer 参数对比

| 参数 | GPU (batch=512) | CPU (batch=32) | 调整原因 |
|------|-----------------|----------------|----------|
| `d_model` | 128 | **96** ↓ | 减小模型维度,加快 CPU 训练 |
| `nhead` | 8 | **6** ↓ | 减少注意力头数 (必须整除 d_model) |
| `num_layers` | 4 | **3** ↓ | 减少 Transformer 层数,更快训练 |
| `dim_feedforward` | 256 | **192** ↓ | 前馈层维度 (通常是 d_model 的 2x) |
| `dropout` | 0.1 | **0.15** ↑ | 增加 dropout 防止小 batch 过拟合 |
| `batch_size` | 512 | **32** ↓ | CPU 推荐值 |
| `lr` | 1e-3 | **5e-4** ↓ | 降低学习率适应小 batch |
| `weight_decay` | 1e-4 | **1e-3** ↑ | 增加 L2 正则化防止过拟合 |
| `max_epochs` | 50 | **60** ↑ | 增加训练轮数补偿慢收敛 |
| `patience` | 8 | **10** ↑ | 增加 early stopping 耐心 |

**模型参数量变化**: ~120K params → ~54K params (减少 55%)

---

## 📊 DeepGBM 参数对比

### Deep 组件

| 参数 | GPU (batch=512) | CPU (batch=32) | 调整原因 |
|------|-----------------|----------------|----------|
| `emb_dim` | 128 | **96** ↓ | 减小叶子嵌入维度 |
| `hidden` | (256, 128) | **(192, 96)** ↓ | 减小 MLP 隐藏层维度 |
| `dropout` | 0.2 | **0.25** ↑ | 增加 dropout 防止过拟合 |
| `batch_size` | 512 | **32** ↓ | CPU 推荐值 |
| `lr` | 1e-3 | **5e-4** ↓ | 降低学习率 |
| `weight_decay` | 1e-4 | **1e-3** ↑ | 增加正则化 |
| `max_epochs` | 40 | **50** ↑ | 增加训练轮数 |
| `patience` | 8 | **10** ↑ | 增加 early stopping 耐心 |

### XGBoost 组件

| 参数 | GPU (batch=512) | CPU (batch=32) | 调整原因 |
|------|-----------------|----------------|----------|
| `n_estimators` | 600 | **400** ↓ | 减少树数量,CPU 训练更快 |
| `max_depth` | 8 | **6** ↓ | 减小树深度防止过拟合 |
| `min_child_weight` | 2.0 | **3.0** ↑ | 更保守的分裂条件 |
| `gamma` | 0.0 | **0.1** ↑ | 增加分裂损失阈值 |
| `reg_lambda` | 1.0 | **1.5** ↑ | 增加 L2 正则化 |
| `reg_alpha` | 0.0 | **0.1** ↑ | 添加 L1 正则化 |

---

## 🔑 关键调整策略

### 1️⃣ **模型容量减小** (避免 CPU 过载)
- Transformer: d_model 128→96, layers 4→3
- DeepGBM: emb_dim 128→96, hidden (256,128)→(192,96)
- XGBoost: trees 600→400, depth 8→6

**影响**: 训练速度提升 40-60%,模型参数减少 ~40%

### 2️⃣ **学习率降低** (适应小 batch)
- 从 `1e-3` 降至 `5e-4`
- 原因: 小 batch 梯度估计噪声大,需要更保守的更新

**影响**: 训练更稳定,但可能需要更多 epochs

### 3️⃣ **正则化增强** (防止过拟合)
- Dropout: 0.1→0.15 (FTT), 0.2→0.25 (DeepGBM)
- Weight decay: 1e-4→1e-3
- XGBoost: 增加 gamma, lambda, alpha

**影响**: 泛化能力提升,验证集性能更稳定

### 4️⃣ **训练时间补偿**
- Max epochs: 50→60 (FTT), 40→50 (DeepGBM)
- Patience: 8→10
- 原因: 小 batch 每个 epoch 更新次数更多,但单次更新步长更小

**影响**: 给模型更多时间收敛到最优

---

## 💡 使用建议

### 如果你有 16GB+ 内存:
```python
# 可以尝试稍大的 batch size
batch_size = 64  # 平衡速度和稳定性
lr = 7e-4  # 相应调整学习率
```

### 如果训练太慢:
```python
# FT-Transformer: 进一步减小模型
d_model = 64
nhead = 4
num_layers = 2

# DeepGBM: 减少 XGBoost 树
n_estimators = 200
max_depth = 5
```

### 如果出现过拟合:
```python
# 增加正则化
dropout = 0.3
weight_decay = 5e-3
min_child_weight = 4.0  # XGBoost
```

### 如果验证损失震荡:
```python
# 进一步降低学习率
lr = 3e-4
# 或使用学习率调度器 (需要在代码中添加)
```

---

## 📈 预期训练时间 (CPU)

基于 Intel i5/i7 或 AMD Ryzen 5/7:

| 模型 | Horizon | 每 Epoch | 总训练时间 (50-60 epochs) |
|------|---------|----------|--------------------------|
| FT-Transformer | h1 | 2-4 min | 2-4 hours |
| FT-Transformer | h24 | 2-4 min | 2-4 hours |
| DeepGBM | h1 | 3-6 min | 3-5 hours |
| DeepGBM | h24 | 3-6 min | 3-5 hours |

**注意**: XGBoost 部分会利用多核 CPU (n_jobs=-1),通常是最快的部分。

---

## ✅ 验证优化效果

运行训练后,检查以下指标:

1. **训练稳定性**: 损失曲线应该平滑下降,不应剧烈震荡
2. **过拟合程度**: train F1 和 valid F1 差距应 < 0.05
3. **收敛速度**: 大部分情况在 30-40 epoch 内达到最佳验证 F1
4. **最终性能**: valid F1 应与 GPU 版本相差 < 0.02

如果未达到预期,参考上面的"使用建议"进行微调。

---

## 🚀 快速开始

所有超参数已经在代码中设置为 CPU 优化的默认值,直接运行即可:

```bash
# FT-Transformer
cd classification-models
python ft_transformer_classifier.py

# DeepGBM
python deepgbm_classifier.py

# XGBoost 基线 (已针对 CPU 优化)
python xgboost_classifier.py
```

不需要修改任何参数,开箱即用! 🎉
