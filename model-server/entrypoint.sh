#!/bin/bash

# Копируем модель в директорию models (которая монтируется как volume)
if [ ! -f /app/models/readmission_model.onnx ]; then
    echo "Copying model files to /app/models..."
    mkdir -p /app/models
    cp /app/readmission_model.onnx /app/models/ 2>/dev/null || true
    cp /app/model_metadata.json /app/models/ 2>/dev/null || true
    echo "Model files copied to /app/models"
fi

# Запускаем uvicorn
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 8 --limit-concurrency 1000


