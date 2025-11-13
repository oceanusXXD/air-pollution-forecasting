#!/bin/bash
# Quick start script for training all models in Docker

echo "=========================================="
echo "Air Pollution Forecasting - Model Training"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Build the image
echo ""
echo "[1/4] Building Docker image..."
docker compose build

if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image"
    exit 1
fi

# Create output directory if not exists
mkdir -p classification-analysis

echo ""
echo "[2/4] Training XGBoost classifier..."
docker compose run --rm air-pollution-classifier \
    python classification-models/xgboost_classifier.py 2>&1 | tee classification-analysis/xgboost_training.log

if [ $? -ne 0 ]; then
    echo "❌ XGBoost training failed!"
    exit 1
fi

echo ""
echo "[3/4] Training FT-Transformer classifier..."
docker compose run --rm air-pollution-classifier \
    python classification-models/ft_transformer_classifier.py 2>&1 | tee classification-analysis/ft_transformer_training.log

if [ $? -ne 0 ]; then
    echo "❌ FT-Transformer training failed!"
    exit 1
fi

echo ""
echo "[4/4] Training DeepGBM classifier..."
docker compose run --rm air-pollution-classifier \
    python classification-models/deepgbm_classifier.py 2>&1 | tee classification-analysis/deepgbm_training.log

if [ $? -ne 0 ]; then
    echo "❌ DeepGBM training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ All models trained successfully!"
echo "Logs saved under: ./classification-analysis/"
echo "=========================================="
