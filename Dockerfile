# Air Pollution Forecasting - Dockerfile
# Multi-stage build for optimized image size

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY classification-models/ ./classification-models/
COPY data_artifacts/ ./data_artifacts/

# Create output directories
RUN mkdir -p classification-analysis

# Set the default command
CMD ["bash"]

# Optional: If you want to run a specific model by default
# CMD ["python", "classification-models/ft_transformer_classifier.py"]
