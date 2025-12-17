FROM python:3.9-slim

WORKDIR /app

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Установка зависимостей для Kafka
RUN pip install --no-cache-dir kafka-python

# Копирование кода мониторинга
COPY data_drift_monitor.py .
COPY monitor_service.py .

# Создание директории для данных
RUN mkdir -p /data

# Команда запуска (можно переключить на monitor_service.py для непрерывного мониторинга)
CMD ["python", "monitor_service.py"]

