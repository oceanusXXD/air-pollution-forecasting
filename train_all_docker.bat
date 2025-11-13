@echo off
REM Quick start script for training all models in Docker (Windows)

echo ==========================================
echo Air Pollution Forecasting - Training All Models
echo ==========================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

REM Build the image
echo.
echo [1/4] Building Docker image...
docker compose build

if errorlevel 1 (
    echo Error: Failed to build Docker image
    exit /b 1
)

REM Create output directory if not exists
if not exist "classification-analysis" mkdir classification-analysis

echo.
echo [2/4] Training XGBoost baseline...
docker compose run --rm air-pollution-classifier python classification-models/xgboost_classifier.py

echo.
echo [3/4] Training FT-Transformer...
docker compose run --rm air-pollution-classifier python classification-models/ft_transformer_classifier.py

echo.
echo [4/4] Training DeepGBM...
docker compose run --rm air-pollution-classifier python classification-models/deepgbm_classifier.py

echo.
echo ==========================================
echo All models trained successfully!
echo Results saved in: .\classification-analysis\
echo ==========================================

pause
