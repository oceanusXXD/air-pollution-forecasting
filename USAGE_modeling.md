# 使用说明（建模同学请看）

本包包含**已切分并完成缩放的训练/验证/测试集**，可直接用于训练。未缩放的全量特征在 `features_h+{H}.parquet` 中，仅用于复查或再造特征。

> 步长 H ∈ {1, 6, 12, 24}。示例以 H=24 为例，换其他步长把路径中的 `h24`/列名中的 `+24` 改掉即可。

---

## 1) 回归（预测 y_t+H）
```python
import pandas as pd
# 载入训练集（已按训练段拟合的标准化器变换过）
df = pd.read_parquet("artifacts/splits/h24/train.parquet")
X = df.drop(columns=["y_t+24", "naive_yhat_t+24"])  # 可保留 anomaly_flag 作为特征
y = df["y_t+24"]

# 例：用 XGBoost / LightGBM / 线性回归等任意回归模型训练
# model.fit(X, y)
```

### 评估（示例，使用验证集）
```python
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

val = pd.read_parquet("artifacts/splits/h24/valid.parquet")
Xv = val.drop(columns=["y_t+24", "naive_yhat_t+24"])
yv = val["y_t+24"]

# yhat = model.predict(Xv)
# rmse = mean_squared_error(yv, yhat, squared=False)

# 朴素基线（naive）：直接用当前 CO 作为下一步预测
rmse_naive = mean_squared_error(yv, val["naive_yhat_t+24"], squared=False)
print("Naive RMSE:", rmse_naive)
```

---

## 2) 分类（预测 co_level_t+H ∈ {low, mid, high}）
```python
import pandas as pd
df = pd.read_parquet("artifacts/splits/h24/train.parquet")
X = df.drop(columns=["y_t+24", "naive_yhat_t+24", "co_level_t+24"])
y = df["co_level_t+24"]  # 分类标签
# 例：任意分类器（LogReg / XGBoost / LightGBM / RF 等）
# clf.fit(X, y)
```

### 评估（F1/准确率）
```python
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

val = pd.read_parquet("artifacts/splits/h24/valid.parquet")
Xv = val.drop(columns=["y_t+24", "naive_yhat_t+24", "co_level_t+24"])
yv = val["co_level_t+24"]

# yhat = clf.predict(Xv)
# print("ACC:", accuracy_score(yv, yhat))
# print("F1-macro:", f1_score(yv, yhat, average="macro"))
```

---

## 3) 列与含义（关键）
- 目标（回归）：`y_t+H`
- 目标（分类）：`co_level_t+H ∈ {low, mid, high}`（阈值 1.5 / 2.5 mg/m³）
- 朴素基线：`naive_yhat_t+H = CO(GT)_t`
- 常见特征：`lag{1,2,3,6,12,24}`、`r{3,6,12,24}_{mean,std}`、`hour/weekday/month`、`anomaly_flag`（可选）

> 提醒：`splits/` 已做**防泄漏**（缩放器仅在 train 拟合）。不要将 train/valid/test 合并后再拟合缩放器。

---

## 4) 路径总览
```
artifacts/
  clean_air_quality.parquet                # 清洗后未特征化数据
  features_h+{1,6,12,24}.parquet           # 未缩放全特征（复查/再加工）
  splits/
    h{1,6,12,24}/                          # 已缩放，可直接建模
      train.parquet
      valid.parquet
      test.parquet
```

---

## 5) 常见问题
- **文件太大？** 用 Release 附件下载 zip 包（见仓库 Releases）。
- **换步长？** 把路径中的 `h24` 改为 `h1/h6/h12`，列名中的 `+24` 同步修改。
- **特征过多？** 模型中可做特征选择或正则化（如 L1/L2、基于重要度筛选）。
