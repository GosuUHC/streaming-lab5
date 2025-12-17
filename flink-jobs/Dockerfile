FROM flink:1.19.0-scala_2.12

# Установка Python зависимостей для PyFlink
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
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

WORKDIR /opt/flink

