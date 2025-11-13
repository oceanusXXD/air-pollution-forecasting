# air-pollution-forecasting# air-pollution-forecasting



Air pollution forecasting project for predicting air quality using machine learning models.Air pollution forecasting project for predicting air quality using machine learning models.



## 🚀 Quick Start## Project Structure



### Using Docker (Recommended)```

air-pollution-forecasting/

```bash├── data/                      # Raw and processed data files

# Build and run all models├── data-processing/           # Scripts for data cleaning and preprocessing

docker-compose build├── regression-analysis/       # Regression analysis notebooks and scripts

docker-compose run --rm air-pollution-classifier bash├── classification-analysis/   # Classification analysis notebooks and scripts

├── regression-models/         # Regression model implementations

# Or use the quick start script└── classification-models/     # Classification model implementations

# Linux/Mac:```

./train_all_docker.sh

## Directory Descriptions

# Windows:

train_all_docker.bat- **data/**: Store raw and processed datasets for air pollution forecasting

```- **data-processing/**: Data cleaning, transformation, and feature engineering scripts

- **regression-analysis/**: Analysis and experiments for regression-based forecasting

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for detailed Docker usage.- **classification-analysis/**: Analysis and experiments for classification-based forecasting

- **regression-models/**: Implementation of regression models (e.g., Linear Regression, Random Forest)

### Local Installation- **classification-models/**: Implementation of classification models (e.g., Logistic Regression, SVM)


```bash
# Install dependencies
pip install -r requirements.txt

# Run models
cd classification-models
python xgboost_classifier.py
python ft_transformer_classifier.py
python deepgbm_classifier.py
```

## Project Structure

```
air-pollution-forecasting/
├── data_artifacts/            # Processed data in parquet format
├── classification-models/     # Classification model implementations
├── classification-analysis/   # Training results and visualizations
├── Dockerfile                 # Docker container definition
├── docker-compose.yml         # Docker Compose configuration
├── requirements.txt           # Python dependencies
└── CPU_TRAINING_GUIDE.md     # CPU optimization guide
```

## 📊 Available Models

### Classification Models (CO Level Prediction)

1. **XGBoost Classifier** - Baseline gradient boosting model
2. **FT-Transformer** - Feature Tokenizer Transformer for tabular data
3. **DeepGBM** - Hybrid model combining XGBoost + Deep Neural Networks

All models predict CO pollution levels across multiple horizons:
- h1: 1 hour ahead
- h6: 6 hours ahead
- h12: 12 hours ahead
- h24: 24 hours ahead

## Directory Descriptions

- **data_artifacts/**: Processed datasets in parquet format with train/valid/test splits
- **classification-models/**: Implementation of classification models
- **classification-analysis/**: Training outputs including metrics, plots, and saved models
- **regression-models/**: Implementation of regression models (future work)
- **regression-analysis/**: Analysis for regression-based forecasting (future work)

## 📖 Documentation

- [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Comprehensive Docker usage guide
- [CPU_TRAINING_GUIDE.md](CPU_TRAINING_GUIDE.md) - CPU training optimization tips
- [USAGE_modeling.md](USAGE_modeling.md) - Modeling methodology and best practices

## 🔧 System Requirements

### Local Installation
- Python 3.11+
- 8GB+ RAM
- CPU: 4+ cores recommended

### Docker Installation
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 5GB+ free disk space

## 💡 Features

- ✅ Three state-of-the-art classification models
- ✅ CPU-optimized hyperparameters (batch_size=32)
- ✅ Progress bars and detailed training logs
- ✅ Automatic early stopping
- ✅ Comprehensive evaluation metrics and visualizations
- ✅ Docker support for reproducible training
- ✅ Multi-horizon forecasting (1, 6, 12, 24 hours)

## 📈 Model Performance

All models are evaluated on:
- Accuracy
- F1-score (macro & weighted)
- Per-class precision/recall
- Confusion matrices
- Comparison with naive baseline

Results are saved in `classification-analysis/{model_name}/h{horizon}/`

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch_size in model code
- Use Docker with memory limits
- Close other applications

### Slow Training
- Models are optimized for CPU training
- Expected training time: 2-4 hours per model
- Use Docker to run in background

### Missing Data
Ensure `data_artifacts/splits/` contains:
```
data_artifacts/splits/
├── h1/
│   ├── train.parquet
│   ├── valid.parquet
│   └── test.parquet
├── h6/
├── h12/
└── h24/
```

## 📝 License

Only for unsw 9417 team project use
