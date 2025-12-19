FROM flink:1.19.0-scala_2.12

# Установка Python зависимостей для PyFlink
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgomp1 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/flink/usrlib

# Копирование Python файлов
COPY requirements.txt .
COPY patient_monitoring.py .
COPY readmission_detection.py .

# Установка Python зависимостей
RUN pip3 install --no-cache-dir -r requirements.txt

# Копирование Kafka connector JAR
COPY jars/flink-sql-connector-kafka-3.2.0-1.19.jar /opt/flink/lib/

# Копирование скрипта запуска
COPY start_job.sh /opt/flink/start_job.sh
RUN chmod +x /opt/flink/start_job.sh

# Создание директории для моделей (будет монтироваться через volume)
RUN mkdir -p /opt/flink/models

# Настройка переменных окружения для Java и PyFlink
# Java уже установлена в базовом образе Flink
ENV PYFLINK_CLIENT_EXECUTABLE=/usr/bin/python3
ENV PYFLINK_EXECUTABLE=/usr/bin/python3

WORKDIR /opt/flink

CMD ["/opt/flink/start_job.sh"]

