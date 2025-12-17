import json
import time
import random
import logging
from kafka import KafkaProducer
import pandas as pd

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientDataGenerator:
    def __init__(self):
        # Загружаем шаблон данных для реалистичных значений
        self._load_data_template()
        self._wait_for_kafka()
        self.producer = KafkaProducer(
            bootstrap_servers=['kafka:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            retries=5,
            request_timeout_ms=30000
        )
        logger.info("Kafka producer initialized")
        
    def _load_data_template(self):
        """Загрузка шаблона данных для реалистичных значений"""
        try:
            # Создаем базовые распределения на основе предоставленных данных
            self.age_dist = lambda: random.randint(18, 90)
            self.gender_dist = lambda: random.choice(['Male', 'Female', 'Other'])
            self.bp_dist = lambda: f"{random.randint(100, 160)}/{random.randint(60, 100)}"
            self.cholesterol_dist = lambda: random.randint(150, 300)
            self.bmi_dist = lambda: round(random.uniform(18.0, 40.0), 1)
            self.diabetes_dist = lambda: random.choice(['Yes', 'No'])
            self.hypertension_dist = lambda: random.choice(['Yes', 'No'])
            self.medication_dist = lambda: random.randint(0, 10)
            self.stay_dist = lambda: random.randint(1, 10)
            self.destination_dist = lambda: random.choice(['Home', 'Nursing_Facility', 'Rehab'])
            
        except Exception as e:
            logger.error(f"Error loading data template: {e}")
            raise
        
    def _wait_for_kafka(self):
        """Ожидание готовности Kafka"""
        max_retries = 30
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                producer = KafkaProducer(
                    bootstrap_servers=['kafka:9092'],
                    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                    request_timeout_ms=5000
                )
                producer.close()
                logger.info("Kafka is ready!")
                return
            except Exception as e:
                retry_count += 1
                logger.info(f"Waiting for Kafka... ({retry_count}/{max_retries})")
                time.sleep(2)
        
        raise Exception("Kafka not ready after waiting")
        
    def generate_patient_data(self, patient_id):
        """Генерация реалистичных данных пациента"""
        return {
            'patient_id': patient_id,
            'age': self.age_dist(),
            'gender': self.gender_dist(),
            'blood_pressure': self.bp_dist(),
            'cholesterol': self.cholesterol_dist(),
            'bmi': self.bmi_dist(),
            'diabetes': self.diabetes_dist(),
            'hypertension': self.hypertension_dist(),
            'medication_count': self.medication_dist(),
            'length_of_stay': self.stay_dist(),
            'discharge_destination': self.destination_dist(),
            'timestamp': int(time.time() * 1000)
        }
    
    def generate_late_data(self, count=100):
        """Генерация late-arriving данных"""
        base_time = int(time.time() * 1000) - 360000  # 6 минут назад
        logger.info(f"Generating {count} late events")
        
        for i in range(count):
            data = self.generate_patient_data(random.randint(1, 1000))
            data['timestamp'] = base_time + random.randint(0, 300000)  # Разброс 5 минут
            self.producer.send('patient-readmissions', data)
            
        self.producer.flush()
        logger.info("Late data generation completed")
    
    def start_normal_load(self, events_per_second=500):
        """Генерация нормальной нагрузки"""
        logger.info(f"Starting load generation: {events_per_second} events/sec")
        
        late_data_counter = 0
        patient_counter = 1
        
        while True:
            batch_start = time.time()
            
            for i in range(events_per_second):
                data = self.generate_patient_data(patient_counter)
                self.producer.send('patient-readmissions', data)
                patient_counter += 1
            
            # Периодическая генерация late данных
            late_data_counter += 1
            if late_data_counter >= 10:  # Каждые 10 секунд
                self.generate_late_data(20)
                late_data_counter = 0
            
            self.producer.flush()
            
            # Поддержание стабильной скорости
            batch_time = time.time() - batch_start
            sleep_time = max(0, 1.0 - batch_time)
            time.sleep(sleep_time)

if __name__ == '__main__':
    try:
        generator = PatientDataGenerator()
        # Даем время другим сервисам запуститься
        time.sleep(10)
        generator.start_normal_load(events_per_second=50)
    except Exception as e:
        logger.error(f"Generator failed: {e}")
        raise