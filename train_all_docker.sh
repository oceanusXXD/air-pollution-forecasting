#!/bin/bash
# Quick start script for training all models in Docker

echo "=========================================="
echo "Air Pollution Forecasting - Training All Models"
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
echo "[2/4] Training XGBoost baseline..."
docker compose run --rm air-pollution-classifier \
    python classification-models/xgboost_classifier.py

echo ""
echo "[3/4] Training FT-Transformer..."
docker compose run --rm air-pollution-classifier \
    python classification-models/ft_transformer_classifier.py

echo ""
echo "[4/4] Training DeepGBM..."
docker compose run --rm air-pollution-classifier \
    python classification-models/deepgbm_classifier.py

echo ""
echo "=========================================="
echo "✓ All models trained successfully!"
echo "Results saved in: ./classification-analysis/"
echo "=========================================="
