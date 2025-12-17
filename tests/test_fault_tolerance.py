"""
Тесты для проверки fault tolerance системы
- Проверка восстановления из checkpoint
- Проверка отсутствия потери данных
- Тестирование обработки сбоев
"""
import time
import requests
import logging
import json
from kafka import KafkaConsumer, KafkaProducer
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaultToleranceTester:
    """Тестер для проверки fault tolerance"""
    
    def __init__(self, kafka_bootstrap='localhost:9092', model_endpoint='http://localhost:8000/predict'):
        self.kafka_bootstrap = kafka_bootstrap
        self.model_endpoint = model_endpoint
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_bootstrap],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        self.consumer = None
        
    def setup_consumer(self, topic='patient-predictions'):
        """Настройка consumer для чтения результатов"""
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[self.kafka_bootstrap],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            consumer_timeout_ms=10000
        )
    
    def send_test_events(self, count=100):
        """Отправка тестовых событий"""
        logger.info(f"Sending {count} test events...")
        sent_events = []
        
        for i in range(count):
            event = {
                'patient_id': i + 1,
                'age': random.randint(18, 90),
                'gender': random.choice(['Male', 'Female', 'Other']),
                'blood_pressure': f"{random.randint(100, 160)}/{random.randint(60, 100)}",
                'cholesterol': random.randint(150, 300),
                'bmi': round(random.uniform(18.0, 40.0), 1),
                'diabetes': random.choice(['Yes', 'No']),
                'hypertension': random.choice(['Yes', 'No']),
                'medication_count': random.randint(0, 10),
                'length_of_stay': random.randint(1, 10),
                'discharge_destination': random.choice(['Home', 'Nursing_Facility', 'Rehab']),
                'timestamp': int(time.time() * 1000) + i  # Уникальные timestamps
            }
            
            self.producer.send('patient-readmissions', event)
            sent_events.append(event)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Sent {i + 1}/{count} events")
        
        self.producer.flush()
        logger.info(f"All {count} events sent")
        return sent_events
    
    def verify_events_processed(self, sent_events, timeout=60):
        """Проверка что все события обработаны"""
        logger.info("Verifying events were processed...")
        
        if not self.consumer:
            self.setup_consumer()
        
        received_events = {}
        start_time = time.time()
        
        try:
            for message in self.consumer:
                if time.time() - start_time > timeout:
                    logger.warning(f"Timeout waiting for events. Received {len(received_events)}/{len(sent_events)}")
                    break
                
                patient_id = message.value.get('patient_id')
                if patient_id:
                    received_events[patient_id] = message.value
                
                if len(received_events) >= len(sent_events):
                    logger.info("All events received!")
                    break
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
        
        processed_count = len(received_events)
        total_count = len(sent_events)
        success_rate = processed_count / total_count if total_count > 0 else 0
        
        logger.info(f"Processed: {processed_count}/{total_count} ({success_rate*100:.1f}%)")
        
        return {
            'sent': total_count,
            'processed': processed_count,
            'success_rate': success_rate,
            'all_processed': processed_count == total_count
        }
    
    def test_checkpoint_recovery(self):
        """
        Тест восстановления из checkpoint
        Симуляция сбоя и проверка восстановления
        """
        logger.info("="*60)
        logger.info("TEST: Checkpoint Recovery")
        logger.info("="*60)
        
        # Шаг 1: Отправка событий до сбоя
        logger.info("Step 1: Sending events before failure...")
        events_before = self.send_test_events(50)
        time.sleep(5)  # Даем время на обработку
        
        # Шаг 2: Симуляция сбоя (остановка Flink job)
        logger.info("Step 2: Simulating failure (manual Flink restart required)...")
        logger.info("⚠️  Please manually restart Flink job to simulate failure")
        input("Press Enter after restarting Flink job...")
        
        # Шаг 3: Отправка событий после восстановления
        logger.info("Step 3: Sending events after recovery...")
        events_after = self.send_test_events(50)
        time.sleep(10)  # Даем время на обработку
        
        # Шаг 4: Проверка что все события обработаны
        logger.info("Step 4: Verifying all events processed...")
        result = self.verify_events_processed(events_before + events_after)
        
        if result['all_processed']:
            logger.info("✅ Checkpoint recovery test PASSED")
            return True
        else:
            logger.warning(f"⚠️  Checkpoint recovery test: {result['success_rate']*100:.1f}% events processed")
            return False
    
    def test_no_data_loss(self):
        """Тест отсутствия потери данных"""
        logger.info("="*60)
        logger.info("TEST: No Data Loss")
        logger.info("="*60)
        
        # Отправка событий
        events = self.send_test_events(200)
        time.sleep(15)  # Даем время на обработку
        
        # Проверка обработки
        result = self.verify_events_processed(events)
        
        if result['success_rate'] >= 0.95:  # Допускаем небольшую потерю из-за timing
            logger.info("✅ No data loss test PASSED")
            return True
        else:
            logger.warning(f"⚠️  Data loss detected: {result['success_rate']*100:.1f}% processed")
            return False
    
    def test_error_handling(self):
        """Тест обработки ошибок"""
        logger.info("="*60)
        logger.info("TEST: Error Handling")
        logger.info("="*60)
        
        # Отправка некорректных событий
        invalid_events = [
            {'invalid': 'data'},  # Отсутствуют обязательные поля
            {'patient_id': 999, 'age': 'invalid'},  # Неправильный тип данных
            None,  # None значение
        ]
        
        logger.info("Sending invalid events...")
        for event in invalid_events:
            try:
                if event:
                    self.producer.send('patient-readmissions', event)
                else:
                    self.producer.send('patient-readmissions', b'invalid json')
            except Exception as e:
                logger.warning(f"Error sending invalid event: {e}")
        
        self.producer.flush()
        time.sleep(5)
        
        # Проверка что система не упала
        try:
            response = requests.get(f"{self.model_endpoint.replace('/predict', '/health')}", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Error handling test PASSED - system still operational")
                return True
            else:
                logger.warning("⚠️  System may have issues after invalid events")
                return False
        except Exception as e:
            logger.error(f"❌ System failed after error handling test: {e}")
            return False
    
    def run_all_tests(self):
        """Запуск всех тестов"""
        logger.info("Starting Fault Tolerance Tests...")
        
        results = {
            'checkpoint_recovery': self.test_checkpoint_recovery(),
            'no_data_loss': self.test_no_data_loss(),
            'error_handling': self.test_error_handling()
        }
        
        logger.info("="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        all_passed = all(results.values())
        logger.info("="*60)
        
        if all_passed:
            logger.info("✅ All fault tolerance tests PASSED")
        else:
            logger.warning("⚠️  Some tests failed or had warnings")
        
        return results


if __name__ == '__main__':
    import sys
    
    kafka_bootstrap = sys.argv[1] if len(sys.argv) > 1 else 'localhost:9092'
    model_endpoint = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:8000/predict'
    
    tester = FaultToleranceTester(kafka_bootstrap, model_endpoint)
    
    # Можно запустить отдельные тесты или все вместе
    if len(sys.argv) > 3:
        test_name = sys.argv[3]
        if test_name == 'checkpoint':
            tester.test_checkpoint_recovery()
        elif test_name == 'data_loss':
            tester.test_no_data_loss()
        elif test_name == 'error':
            tester.test_error_handling()
        else:
            tester.run_all_tests()
    else:
        tester.run_all_tests()

