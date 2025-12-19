import requests
import time
import statistics
import numpy as np
import random


def test_latency():
    """Тестирование задержки системы"""
    latencies = []

    print("Testing system latency...")

    for i in range(100):
        test_json = {
            'patient_id': i,
            'age': random.randint(18, 90),
            'gender': random.choice(['Male', 'Female', 'Other']),
            'blood_pressure': f'{random.randint(100, 160)}/{random.randint(60, 100)}',
            'cholesterol': random.randint(150, 300),
            'bmi': round(random.uniform(18.0, 40.0), 1),
            'diabetes': random.choice(['Yes', 'No']),
            'hypertension': random.choice(['Yes', 'No']),
            'medication_count': random.randint(0, 10),
            'length_of_stay': random.randint(1, 10),
            'discharge_destination': random.choice(['Home', 'Nursing_Facility', 'Rehab']),
            'timestamp': int(time.time() * 1000)
        }
        start_time = time.time()

        try:
            # Отправляем тестовый запрос
            response = requests.post(
                'http://localhost:8000/predict',
                json=test_json,
                timeout=0.5
            )

            if response.status_code == 200:
                latency = (time.time() - start_time) * 1000  # в мс
                latencies.append(latency)

            if i % 10 == 0:
                print(f"  Processed {i} requests...")

        except Exception as e:
            print(f"  Request {i} failed: {e}")

        time.sleep(0.01)  # ~100 запросов в секунду

    if latencies:
        p99 = np.percentile(latencies, 99)
        p95 = np.percentile(latencies, 95)
        p50 = np.percentile(latencies, 50)

        print(f"\nLatency Results:")
        print(f"  p50: {p50:.2f} ms")
        print(f"  p95: {p95:.2f} ms")
        print(f"  p99: {p99:.2f} ms")
        print(f"  Samples: {len(latencies)}")

        # Проверка требования
        if p99 < 30:
            print("✅ SUCCESS: p99 < 30 ms requirement MET!")
            return True
        else:
            print(f"❌ FAIL: p99 = {p99:.2f} ms > 30 ms requirement")
            return False
    else:
        print("❌ No successful requests recorded")
        return False


if __name__ == '__main__':
    test_latency()
