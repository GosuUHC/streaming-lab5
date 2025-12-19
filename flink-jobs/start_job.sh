#!/bin/bash

echo "=============================================="
echo "FLINK JOB STARTER - Patient Monitoring"
echo "=============================================="

# Ждём фиксированное время для готовности сервисов
echo "Waiting 60 seconds for all services to be ready..."
sleep 60

echo "=============================================="
echo "STARTING FLINK JOB"
echo "=============================================="

# Проверка модели
if [ -f /opt/flink/models/readmission_model.onnx ]; then
    echo "ONNX model found: /opt/flink/models/readmission_model.onnx"
else
    echo "WARNING: ONNX model NOT found: /opt/flink/models/readmission_model.onnx"
fi

cd /opt/flink/usrlib
exec python3 -u patient_monitoring.py


