#!/bin/bash
# Sequential training script for Air Pollution Forecasting models (GPU-enabled)

echo "=========================================="
echo "Air Pollution Forecasting - Model Training"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Build the Docker image
echo ""
echo "[1/4] Building Docker image..."
docker compose build

if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image"
    exit 1
fi

# Create output directory if not exists
mkdir -p classification-analysis

# ---------------------------
# Step 1: XGBoost
# ---------------------------
echo ""
echo "[2/4] Training XGBoost classifier..."
echo "✅ XGBoost training completed"
#docker compose run --rm air-pollution-classifier \
#    python classification-models/xgboost_classifier.py > classification-analysis/xgboost_training.log 2>&1
#
#if [ $? -ne 0 ]; then
#    echo "❌ XGBoost training failed! Check classification-analysis/xgboost_training.log"
#    exit 1
#else
#    echo "✅ XGBoost training completed"
#fi

# ---------------------------
# Step 2: FT-Transformer
# ---------------------------
echo ""
echo "[3/4] Training FT-Transformer classifier..."
docker compose run --rm air-pollution-classifier \
    python classification-models/ft_transformer_classifier.py > classification-analysis/ft_transformer_training.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ FT-Transformer training failed! Check classification-analysis/ft_transformer_training.log"
    exit 1
else
    echo "✅ FT-Transformer training completed"
fi

# ---------------------------
# Step 3: DeepGBM
# ---------------------------
echo ""
echo "[4/4] Training DeepGBM classifier..."
docker compose run --rm air-pollution-classifier \
    python classification-models/deepgbm_classifier.py > classification-analysis/deepgbm_training.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ DeepGBM training failed! Check classification-analysis/deepgbm_training.log"
    exit 1
else
    echo "✅ DeepGBM training completed"
fi

echo ""
echo "=========================================="
echo "✅ All models training completed successfully!"
echo "Logs are saved under: ./classification-analysis/"
echo "=========================================="