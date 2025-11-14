# 空气污染预测 - CO浓度分类

基于机器学习的空气质量预测项目，使用 XGBoost、DeepGBM 和 FT-Transformer 对 CO 浓度进行三分类预测（低/中/高）。

---

## 📋 项目简介

**任务目标**: 预测未来不同时间窗口的 CO 浓度等级
- **预测时间窗口**: h+1 (1小时后), h+6 (6小时后), h+12 (12小时后), h+24 (24小时后)
- **分类标签**: Low (低浓度) / Mid (中等浓度) / High (高浓度)
- **输入特征**: 823 维表格特征（包含历史污染物浓度、气象数据、时间特征、滞后特征等）

**硬件环境**:
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: 2核心 (Docker资源限制: 1.8 CPU, 4GB 内存)

---

## 🚀 快速使用

```bash
# 一键训练所有模型（后台运行，nohup模式）
./train_all_docker.sh

# 实时查看训练日志
tail -f classification-analysis/xgboost_training.log
tail -f classification-analysis/deepgbm_training.log
tail -f classification-analysis/ft_transformer_training.log
```

**说明**: 训练会在后台运行，关闭终端或 VS Code 不影响训练进程。所有日志和结果保存在 `classification-analysis/` 目录。

---

## 📊 模型说明与对比

### 1. XGBoost Classifier (Baseline) ✅

**模型特点**:
- 基于梯度提升决策树的传统机器学习方法
- 使用 XGBoost 2.1.4，针对不同预测时间窗口采用自适应超参数
- 无需 GPU，CPU 训练即可获得稳定性能

**超参数策略**:
- **短期预测 (h+1)**: 较深的树 (max_depth=8)、较快的学习率 (lr=0.05)，适合快速收敛
- **中期预测 (h+6)**: 中等深度 (max_depth=6)、增强正则化 (L1=0.5, L2=2.0)
- **长期预测 (h+12/24)**: 较浅的树 (max_depth=4-5)、更强正则化、更多训练轮数，防止过拟合

**性能表现**:
| Horizon | 测试集 Accuracy | 测试集 F1-Macro | 训练时间 |
|---------|----------------|----------------|----------|
| h+1     | 79.93%         | 0.7803         | 12秒     |
| h+6     | 62.36%         | 0.5652         | 19秒     |
| h+12    | 57.29%         | 0.5292         | 27秒     |
| h+24    | 56.61%         | 0.5219         | 33秒     |

*(GPU加速训练，NVIDIA RTX 4090)*

**关键特性**:
- ✅ 类别不平衡处理：使用样本加权 (class weights)
- ✅ 早停机制：基于验证集 mlogloss，避免过拟合
- ✅ 特征重要性分析：输出 Top-20 重要特征及可视化

---

### 2. DeepGBM (Deep Gradient Boosting Machine) ✅

**模型架构**:
DeepGBM 是 XGBoost 与深度神经网络的混合模型，采用两阶段训练：
1. **阶段一**: 训练浅层 XGBoost（50棵树，depth=6）
2. **阶段二**: 提取叶节点索引作为类别特征，训练深度 MLP
   - 叶节点嵌入层: 150,000 桶 → 64维向量
   - 全连接层: 原始特征 (823维) + 叶节点嵌入 → 隐藏层 (128, 64) → 3分类
   - 总参数量: ~970万参数

**训练配置**:
- 优化器: AdamW (lr=5e-4, weight_decay=1e-3)
- Batch size: 64
- 最大 Epochs: 25
- 早停策略: patience=8

**性能表现** (h+24 示例):
- Deep-only: Acc=51.48%, F1=0.4928
- XGB-only: Acc=55.44%, F1=0.4967  
- **Deep+XGB (ensemble)**: Acc=54.27%, F1=0.5115

**优势**:
- 结合树模型的特征交互能力和神经网络的表达能力
- 叶节点嵌入捕获非线性决策边界
- 支持 GPU 加速训练

**已知问题与修复**:
- ⚠️ XGBoost 2.x API 变更：原 `predict(..., pred_leaf=True)` 不可用
- ✅ 修复方案：使用 `model.apply(X)` 或 `Booster.predict(DMatrix, pred_leaf=True)` 获取叶节点索引
- ⚠️ Pandas Categorical 类型问题：`torch.from_numpy()` 无法直接处理 Categorical
- ✅ 修复方案：在 `_DeepDS.__init__()` 中将标签转为 NumPy 数组

**训练时间预估**: 每个 horizon 约 40-60 分钟 (CPU 环境)

---

### 3. FT-Transformer (实验性) ⚠️

**模型架构**:
- Feature Tokenizer + Transformer Encoder
- 参数量: ~39万 (优化后: d_model=96, 3层, 4注意力头)

**CPU 训练挑战**:
- ⚠️ **模型过大导致训练困难**: 虽然已将参数从 75万降至 39万，但在 2核 CPU 上：
  - 首个 batch 的前向传播耗时 30-60秒（模型编译）
  - 单个 epoch 预计耗时 3-5分钟
  - 完整训练（30 epochs × 4 horizons）需要 6-10 小时
- ⚠️ **训练卡顿问题**: 在某些环境下，训练循环在初始化 tqdm 后无响应

**优化措施**:
- 减少模型复杂度: d_model=96, num_layers=3, nhead=4
- 增大 batch size: 32 (提升梯度稳定性)
- 降低学习率: 5e-4 (适配 CPU 训练)
- 减少最大 epochs: 30 (原50)
- 移除 tqdm 进度条，改为直接 print 输出

**建议**:
- 🎯 **推荐硬件**: NVIDIA RTX 4090 或更高算力的 GPU
- 🎯 **训练策略**: 在 GPU 上训练，或仅将 FT-Transformer 用于性能对比实验
- 🎯 **生产部署**: 使用 XGBoost 或 DeepGBM，FT-Transformer 仅作学术研究

---

## 📂 结果输出位置

训练完成后，所有结果保存在 `classification-analysis/` 目录下：

```
classification-analysis/
├── xgboost_training.log              # XGBoost 训练日志
├── deepgbm_training.log              # DeepGBM 训练日志
├── ft_transformer_training.log       # FT-Transformer 训练日志
│
├── xgboost/                          # XGBoost 模型结果
│   ├── h1/
│   │   ├── xgb_classifier_h1.joblib         # 训练好的模型文件
│   │   ├── xgb_results_h1.json              # 详细评估指标 (JSON)
│   │   ├── confusion_matrices_h1.png        # 混淆矩阵可视化
│   │   ├── feature_importance_h1.png        # 特征重要性排序图
│   │   ├── training_history_h1.png          # 训练曲线 (mlogloss)
│   │   └── xgb_feature_importance_h1.csv    # 特征重要性表格
│   ├── h6/  h12/  h24/  (同上结构)
│
├── deepgbm/                          # DeepGBM 模型结果
│   ├── h1/
│   │   ├── deepgbm_xgb_h1.joblib            # XGBoost 基学习器
│   │   ├── deepgbm_deep_h1.pth              # 深度组件 (PyTorch)
│   │   ├── deepgbm_results_h1.json          # 评估指标
│   │   └── confusion_matrices_h1.png        # 混淆矩阵
│   ├── h6/  h12/  h24/  (同上结构)
│
└── ft_transformer/                   # FT-Transformer 模型结果
    ├── h1/
    │   ├── ft_transformer_h1.pth            # 模型权重
    │   ├── ft_results_h1.json               # 评估指标
    │   └── confusion_matrices_h1.png        # 混淆矩阵
    └── h6/  h12/  h24/  (同上结构)
```

**关键文件说明**:
- **`*_results_*.json`**: 包含详细的评估指标
  ```json
  {
    "test_accuracy": 0.7997,
    "test_f1_macro": 0.7819,
    "test_f1_weighted": 0.8045,
    "training_time_seconds": 180.5,
    "per_class_metrics": { ... }
  }
  ```
- **`confusion_matrices_*.png`**: 训练集/验证集/测试集的混淆矩阵对比
- **`feature_importance_*.png`**: XGBoost 特征重要性 Top-20 柱状图
- **`training_history_*.png`**: 训练过程中的验证集 mlogloss 曲线

---

## 🔧 系统要求

### Docker 环境 (推荐)
- Docker 20.10+
- Docker Compose 2.0+
- 磁盘空间: 10GB+
- 内存: 4GB+

### 硬件配置
- **XGBoost**: CPU 即可，2核心足够
- **DeepGBM**: CPU 可训练，但速度较慢 (推荐 GPU)
- **FT-Transformer**: 强烈推荐 GPU (RTX 4090 或更高)

### Docker 资源限制
当前配置 (`docker-compose.yml`):
```yaml
resources:
  limits:
    cpus: '1.8'      # 使用 90% 的 2核 CPU
    memory: 4G       # 4GB 内存限制
  reservations:
    cpus: '1.0'
    memory: 2G
```

**环境变量** (并行库线程数限制):
```yaml
OMP_NUM_THREADS: 2
MKL_NUM_THREADS: 2
NUMEXPR_NUM_THREADS: 2
OPENBLAS_NUM_THREADS: 2
```

---

## 📈 评估指标说明

所有模型在以下指标上进行评估：

1. **Accuracy (准确率)**: 整体分类正确率
2. **F1-Score**:
   - Macro-F1: 三类别 F1 的平均值 (关注类别平衡)
   - Weighted-F1: 按样本数加权的 F1 (关注整体性能)
3. **Per-class Metrics**: 每个类别的 Precision / Recall / F1-Score
4. **Confusion Matrix**: 预测与真实标签的交叉分布
5. **Baseline Comparison**: 与朴素基线 (naive baseline) 的性能提升

**类别分布 (h+1 训练集)**:
- Low: 39.1%
- Mid: 27.8%
- High: 33.1%

---

## 🐛 已知问题与解决方案

### 1. XGBoost API 兼容性
- **问题**: XGBoost 2.x 移除了 sklearn 接口的 `callbacks` 参数
- **解决**: 在 `fit()` 中移除 `callbacks`，使用 `verbose=True` 代替

### 2. DeepGBM 叶节点提取
- **问题**: `predict(..., pred_leaf=True)` 在 XGBoost 2.x 不可用
- **解决**: 使用 `model.apply(X)` 或 Booster API 的 `predict(DMatrix, pred_leaf=True)`

### 3. Pandas Categorical 类型
- **问题**: PyTorch 无法直接将 Categorical 转为 Tensor
- **解决**: 在 Dataset 构造时先转为 NumPy 数组: `y.to_numpy()` 或 `y.values`

### 4. FT-Transformer CPU 训练缓慢
- **问题**: Transformer 计算复杂度高，CPU 上每个 epoch 3-5分钟
- **解决**: 
  - 短期：减少模型参数、增大 batch size、减少 epochs
  - 长期：迁移到 GPU 训练

---

## 📝 开发日志

**2025-11-14**:
- ✅ 完成 XGBoost 超参数优化（针对不同 horizon 的自适应策略）
- ✅ 修复 DeepGBM 叶节点提取 bug (XGBoost 2.x 兼容性)
- ✅ 修复 DeepGBM Dataset 类型转换 bug (Categorical → NumPy)
- ✅ 优化 FT-Transformer 模型大小（75万 → 39万参数）
- ⚠️ FT-Transformer 在 CPU 上训练存在性能瓶颈，建议使用 GPU

**2025-11-13**:
- ✅ 项目初始化，配置 Docker 环境
- ✅ 实现 XGBoost、DeepGBM、FT-Transformer 三个分类器
- ✅ 数据预处理：特征工程、train/valid/test 划分

---

## 📧 技术支持

如遇到问题，请查看：
1. 训练日志: `classification-analysis/*_training.log`
2. Docker 日志: `docker compose logs -f air-pollution-classifier`
3. 容器状态: `docker compose ps`

---

**License**: MIT  
**Python Version**: 3.11  
**XGBoost Version**: 2.1.4  
**PyTorch Version**: 2.0+
