# Docker 使用指南

## 🐳 快速开始

### 1. 构建 Docker 镜像

```bash
# 在项目根目录运行
docker compose build
```

### 2. 运行模型训练

#### 方式 A: 交互式容器 (推荐)

启动容器并进入交互式 shell:

```bash
docker compose run --rm air-pollution-classifier bash
```

进入容器后运行模型:

```bash
# 在容器内执行
cd classification-models

# 运行 XGBoost 基线
python xgboost_classifier.py

# 运行 FT-Transformer
python ft_transformer_classifier.py

# 运行 DeepGBM
python deepgbm_classifier.py
```

#### 方式 B: 直接运行单个模型

```bash
# 运行 XGBoost
docker compose run --rm air-pollution-classifier \
    python classification-models/xgboost_classifier.py

# 运行 FT-Transformer
docker compose run --rm air-pollution-classifier \
    python classification-models/ft_transformer_classifier.py

# 运行 DeepGBM
docker compose run --rm air-pollution-classifier \
    python classification-models/deepgbm_classifier.py
```

#### 方式 C: 后台运行

```bash
# 启动容器在后台
docker compose up -d air-pollution-classifier

# 进入运行中的容器
docker exec -it air-pollution-classifier bash

# 停止容器
docker compose down
```

### 3. 使用 Jupyter Notebook (可选)

```bash
# 启动 Jupyter 服务
docker compose up jupyter

# 在浏览器打开: http://localhost:8888
# 无需 token (已在 docker-compose.yml 中配置)
```

---

## 📁 数据和结果

### 目录挂载

Docker 容器会自动挂载以下目录:

```
宿主机                          →  容器内
./data_artifacts               →  /app/data_artifacts (只读)
./classification-analysis      →  /app/classification-analysis (读写)
./classification-models        →  /app/classification-models (只读)
```

### 输出位置

所有训练结果会保存在:
- **宿主机**: `./classification-analysis/{model_name}/h{horizon}/`
- **容器内**: `/app/classification-analysis/{model_name}/h{horizon}/`

输出文件包括:
- `metrics.json` - 性能指标
- `confusion_matrix_*.png` - 混淆矩阵图
- `model.pkl` / `model.pt` - 训练好的模型
- `scaler.pkl` - 特征标准化器

---

## 🔧 常用命令

### 查看运行日志

```bash
# 实时查看日志
docker compose logs -f air-pollution-classifier

# 查看最近 100 行日志
docker compose logs --tail=100 air-pollution-classifier
```

### 资源监控

```bash
# 查看容器资源使用情况
docker stats air-pollution-classifier
```

### 清理资源

```bash
# 停止并删除容器
docker compose down

# 删除镜像
docker compose down --rmi all

# 删除所有(包括 volumes)
docker compose down -v
```

### 重新构建

```bash
# 强制重新构建(当修改了 requirements.txt 或 Dockerfile)
docker compose build --no-cache
```

---

## ⚙️ 自定义配置

### 调整资源限制

编辑 `docker-compose.yml` 中的资源配置:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # 修改为你的 CPU 核心数
      memory: 8G       # 修改为可用内存大小
```

### 修改线程数

编辑 `docker-compose.yml` 中的环境变量:

```yaml
environment:
  - OMP_NUM_THREADS=4     # 修改为你的 CPU 核心数
  - MKL_NUM_THREADS=4
  - NUMEXPR_NUM_THREADS=4
```

### 使用 GPU (如果有)

需要安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

修改 `docker-compose.yml`:

```yaml
air-pollution-classifier:
  # ... 其他配置 ...
  
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

修改 Dockerfile 使用 GPU 版本:

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
```

---

## 🐛 故障排查

### 问题 1: 内存不足

**症状**: 容器被 killed 或 OOM 错误

**解决方案**:
```bash
# 增加 Docker 内存限制
# Windows/Mac: Docker Desktop -> Settings -> Resources -> Memory
# Linux: 修改 docker-compose.yml 中的 memory limit
```

### 问题 2: 权限错误

**症状**: 无法写入 classification-analysis 目录

**解决方案**:
```bash
# Linux/Mac: 确保目录有写权限
chmod -R 777 classification-analysis

# 或在容器内以 root 运行 (已默认)
```

### 问题 3: 模块未找到

**症状**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:
```bash
# 重新构建镜像
docker compose build --no-cache
```

### 问题 4: 数据文件未找到

**症状**: `FileNotFoundError: data_artifacts/splits/...`

**解决方案**:
```bash
# 确保 data_artifacts 目录存在且包含数据
ls data_artifacts/splits/

# 检查 volume 挂载
docker compose config
```

---

## 📊 批量训练示例

### 训练所有模型和所有 horizons

创建一个训练脚本 `train_all.sh`:

```bash
#!/bin/bash
# 在容器内运行

cd /app/classification-models

echo "=== Training XGBoost ==="
python xgboost_classifier.py

echo "=== Training FT-Transformer ==="
python ft_transformer_classifier.py

echo "=== Training DeepGBM ==="
python deepgbm_classifier.py

echo "=== All training completed ==="
```

运行:

```bash
# 复制脚本到容器
docker cp train_all.sh air-pollution-classifier:/app/

# 运行脚本
docker exec -it air-pollution-classifier bash /app/train_all.sh
```

### 并行训练不同 horizons

```bash
# 使用 Docker Compose scale (需要修改配置支持)
# 或者开启多个容器分别训练
docker compose run -d --name trainer-h1 air-pollution-classifier \
    python classification-models/xgboost_classifier.py

docker compose run -d --name trainer-h6 air-pollution-classifier \
    python classification-models/ft_transformer_classifier.py
```

---

## 🎯 最佳实践

1. **首次运行**: 使用交互式模式测试,确保一切正常
2. **开发阶段**: 使用 volume 挂载,无需重新构建即可测试代码修改
3. **生产训练**: 使用 `docker compose run --rm` 一次性运行
4. **结果备份**: 定期备份 `classification-analysis` 目录
5. **资源监控**: 使用 `docker stats` 监控资源使用

---

## 📚 更多资源

- [Docker Compose 文档](https://docs.docker.com/compose/)
- [PyTorch Docker 镜像](https://hub.docker.com/r/pytorch/pytorch)
- [XGBoost 文档](https://xgboost.readthedocs.io/)

---

## 🆘 需要帮助?

如遇问题,请检查:
1. Docker 版本: `docker --version` (推荐 20.10+)
2. Docker Compose 版本: `docker compose version` (推荐 v2.0+)
3. 可用磁盘空间: 至少需要 5GB
4. 可用内存: 至少需要 8GB

**注意**: 如果你使用的是旧版 Docker Compose (v1.x),命令格式是 `docker-compose` (带连字符)。本项目推荐使用新版 Docker Compose v2+ (`docker compose` 无连字符)。
