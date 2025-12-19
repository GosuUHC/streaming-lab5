from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import KafkaSource, KafkaSink
from pyflink.datastream.formats import JsonRowDeserializationSchema, JsonRowSerializationSchema
from pyflink.common import WatermarkStrategy, Time, Duration
from pyflink.common.typeinfo import Types
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor, StateTtlConfig
import json
import requests
import logging


class PatientMonitoringProcessFunction(KeyedProcessFunction):
    def __init__(self):
        self.model_url = "http://model-server:8000/predict"
        self.redis_url = "redis://redis:6379"

    def open(self, runtime_context: RuntimeContext):
        # Настройка state TTL для оптимизации производительности
        ttl_config = StateTtlConfig \
            .new_builder(Duration.of_hours(1)) \
            .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite) \
            .cleanup_full_snapshot() \
            .build()

        state_descriptor = ValueStateDescriptor("patient_state", Types.PY_DICT)
        state_descriptor.enable_time_to_live(ttl_config)
        self.patient_state = runtime_context.get_state(state_descriptor)

    def process_element(self, value, ctx):
        patient_data = {
            'patient_id': value['patient_id'],
            'age': value['age'],
            'glucose_level': value['glucose_level'],
            'blood_pressure': value['blood_pressure'],
            'bmi': value['bmi'],
            'readmission_risk': 0.0
        }

        # Получение исторических данных из state
        historical_data = self.patient_state.value()
        if historical_data:
            # Расчет изменений показателей
            glucose_change = abs(
                patient_data['glucose_level'] - historical_data.get('glucose_level', 0))
            bp_change = abs(
                patient_data['blood_pressure'] - historical_data.get('blood_pressure', 0))

            patient_data['glucose_change'] = glucose_change
            patient_data['bp_change'] = bp_change
            patient_data['previous_readmission_risk'] = historical_data.get(
                'readmission_risk', 0)

        # Инференс модели
        try:
            response = requests.post(
                self.model_url,
                json=patient_data,
                timeout=0.05  # 50ms timeout для соблюдения p99 < 30ms
            )
            if response.status_code == 200:
                prediction = response.json()
                patient_data['readmission_risk'] = prediction['risk_score']
                patient_data['anomaly_detected'] = prediction['anomaly']

                # Обновление state
                self.patient_state.update(patient_data)

                # Эмит результата с timestamp для мониторинга задержки
                result = {
                    'patient_id': patient_data['patient_id'],
                    'risk_score': patient_data['readmission_risk'],
                    'anomaly': patient_data['anomaly_detected'],
                    'processing_timestamp': ctx.timestamp(),
                    'event_timestamp': value['timestamp']
                }

                yield result

        except Exception as e:
            logging.error(f"Model inference error: {e}")
            # Отправка в DLQ
            dlq_data = {
                'original_data': patient_data,
                'error': str(e),
                'timestamp': ctx.timestamp()
            }
            ctx.output(self.dlq_tag, dlq_data)


def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(8)  # Оптимизация для низкой задержки

    # Настройка checkpointing для exactly-once semantics
    env.enable_checkpointing(1000)  # Каждую секунду
    env.get_checkpoint_config().set_min_pause_between_checkpoints(500)
    env.get_checkpoint_config().set_checkpoint_timeout(60000)

    # Источник Kafka
    source = KafkaSource.builder() \
        .set_bootstrap_servers('kafka:9092') \
        .set_topics('patient-readmissions') \
        .set_group_id('flink-readmission-detection') \
        .set_starting_offsets() \
        .set_value_only_deserializer(
            JsonRowDeserializationSchema.builder()
            .type_info(Types.PY_DICT)
            .build()
    ) \
        .build()

    # Watermark стратегия для обработки late-arriving данных
    watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5)) \
        .with_timestamp_assigner(lambda event, timestamp: event['timestamp'])

    # Создание стрима
    ds = env.from_source(source, watermark_strategy, "Kafka Source")

    # Обработка
    processed_stream = ds \
        .key_by(lambda x: x['patient_id']) \
        .process(PatientMonitoringProcessFunction()) \
        .name("patient-monitoring-process")

    # Синк для результатов
    sink = KafkaSink.builder() \
        .set_bootstrap_servers('kafka:9092') \
        .set_record_serializer(
            JsonRowSerializationSchema.builder()
            .with_type_info(Types.PY_DICT)
            .build()
    ) \
        .set_topic('readmission-predictions') \
        .build()

    processed_stream.sink_to(sink)

    env.execute("Patient Readmission Detection")


if __name__ == '__main__':
    main()
