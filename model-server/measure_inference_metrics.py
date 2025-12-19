"""
Скрипт для измерения метрик инференса ML модели
Измеряет latency (p50, p95, p99) и error rate
"""
import requests
import time
import numpy as np
import logging
import random
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_request() -> Dict:
    """Генерация тестового запроса"""
    return {
        'patient_id': random.randint(1, 10000),
        'age': random.randint(18, 90),
        'gender': random.choice(['Male', 'Female', 'Other']),
        'blood_pressure': f"{random.randint(100, 160)}/{random.randint(60, 100)}",
        'cholesterol': random.randint(150, 300),
        'bmi': round(random.uniform(18.0, 40.0), 1),
        'diabetes': random.choice(['Yes', 'No']),
        'hypertension': random.choice(['Yes', 'No']),
        'medication_count': random.randint(0, 10),
        'length_of_stay': random.randint(1, 10),
        'discharge_destination': random.choice(['Home', 'Nursing_Facility', 'Rehab'])
    }


def measure_metrics(model_endpoint: str = "http://localhost:8000/predict", 
                    num_requests: int = 1000) -> Dict:
    """
    Измерение метрик инференса
    
    Args:
        model_endpoint: URL эндпоинта модели
        num_requests: Количество запросов для тестирования
    
    Returns:
        Словарь с метриками: p50, p95, p99, error_rate
    """
    latencies = []
    errors = []
    error_count = 0
    
    logger.info(f"Starting metrics measurement: {num_requests} requests to {model_endpoint}")
    
    start_time = time.time()
    
    for i in range(num_requests):
        request_start = time.time()
        
        try:
            sample_request = generate_sample_request()
            response = requests.post(
                model_endpoint,
                json=sample_request,
                timeout=1.0  # 1 секунда timeout
            )
            
            request_latency = (time.time() - request_start) * 1000  # в миллисекундах
            
            if response.status_code == 200:
                latencies.append(request_latency)
            else:
                error_count += 1
                errors.append({
                    'status_code': response.status_code,
                    'error': response.text,
                    'latency_ms': request_latency
                })
                logger.warning(f"Request {i} failed with status {response.status_code}")
                
        except requests.exceptions.Timeout:
            error_count += 1
            request_latency = (time.time() - request_start) * 1000
            errors.append({
                'error_type': 'Timeout',
                'latency_ms': request_latency
            })
            logger.warning(f"Request {i} timed out")
            
        except Exception as e:
            error_count += 1
            request_latency = (time.time() - request_start) * 1000
            errors.append({
                'error_type': type(e).__name__,
                'error': str(e),
                'latency_ms': request_latency
            })
            logger.error(f"Request {i} failed: {e}")
        
        # Небольшая задержка для имитации реальной нагрузки
        if i % 100 == 0 and i > 0:
            logger.info(f"Progress: {i}/{num_requests} requests processed")
    
    total_time = time.time() - start_time
    
    # Расчет метрик
    if latencies:
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
    else:
        p50 = p95 = p99 = avg_latency = min_latency = max_latency = 0
    
    error_rate = error_count / num_requests if num_requests > 0 else 0
    
    metrics = {
        'p50_ms': round(p50, 2),
        'p95_ms': round(p95, 2),
        'p99_ms': round(p99, 2),
        'avg_latency_ms': round(avg_latency, 2),
        'min_latency_ms': round(min_latency, 2),
        'max_latency_ms': round(max_latency, 2),
        'error_rate': round(error_rate, 4),
        'error_count': error_count,
        'successful_requests': len(latencies),
        'total_requests': num_requests,
        'total_time_sec': round(total_time, 2)
    }
    
    return metrics, errors


def print_metrics_report(metrics: Dict, errors: List):
    """Вывод отчета о метриках"""
    print("\n" + "="*60)
    print("INFERENCE METRICS REPORT")
    print("="*60)
    print(f"\n📊 Latency Metrics (ms):")
    print(f"  p50:  {metrics['p50_ms']:.2f} ms")
    print(f"  p95:  {metrics['p95_ms']:.2f} ms")
    print(f"  p99:  {metrics['p99_ms']:.2f} ms")
    print(f"  Avg:  {metrics['avg_latency_ms']:.2f} ms")
    print(f"  Min:  {metrics['min_latency_ms']:.2f} ms")
    print(f"  Max:  {metrics['max_latency_ms']:.2f} ms")
    
    print(f"\n❌ Error Metrics:")
    print(f"  Error Rate: {metrics['error_rate']*100:.2f}%")
    print(f"  Errors: {metrics['error_count']}/{metrics['total_requests']}")
    print(f"  Successful: {metrics['successful_requests']}/{metrics['total_requests']}")
    
    print(f"\n⏱️  Total Time: {metrics['total_time_sec']:.2f} seconds")
    
    # Проверка требования p99 < 30ms
    print(f"\n✅ Requirement Check:")
    if metrics['p99_ms'] < 30:
        print(f"  ✅ PASS: p99 = {metrics['p99_ms']:.2f} ms < 30 ms")
    else:
        print(f"  ❌ FAIL: p99 = {metrics['p99_ms']:.2f} ms >= 30 ms")
    
    if errors and len(errors) <= 10:
        print(f"\n⚠️  Sample Errors:")
        for i, error in enumerate(errors[:5], 1):
            print(f"  {i}. {error}")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    import sys
    
    # Параметры из командной строки
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/predict"
    num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    try:
        metrics, errors = measure_metrics(endpoint, num_requests)
        print_metrics_report(metrics, errors)
        
        # Сохранение результатов в файл
        import json
        with open('inference_metrics.json', 'w') as f:
            json.dump({
                'metrics': metrics,
                'errors': errors[:100]  # Сохраняем только первые 100 ошибок
            }, f, indent=2)
        
        logger.info("Metrics saved to inference_metrics.json")
        
    except Exception as e:
        logger.error(f"Failed to measure metrics: {e}")
        sys.exit(1)

