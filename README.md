# 5. Потоковая обработка и работа с real-time данными (Лаба в процессе!)

- 1 - Готово
- 2 - Почти готово
- 3 - Готово
- 4 - В процессе
- 5 - Почти готово
- 6 - В процессе
- 7 - Почти готово

#### Датасет:

Вариант 6 - [Медицинские данные о повторных госпитализациях](https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset)

#### Бизнес-сценарий

3. Мониторинг IoT-устройств

- Цель: детекция аномалий в работе устройств
- Требования:
- Latency < 30 мс для инференса
- Обработка out-of-order событий
- Поддержка high velocity (1000+ событий/сек)

#### Уникальное требование к задержке

- 6. p99 < 30 мс


## Часть 1: Проектирование потоковой системы

### 1. Выбор между Micro-batch и True Streaming обработкой

**Выбор: True Streaming обработка с Apache Flink**

**Обоснование выбора:**

| Критерий | True Streaming (Flink) | Micro-batch (Spark) | Обоснование |
|----------|------------------------|---------------------|-------------|
| **Задержка обработки** | < 30 мс (p99) | 100-500 мс | Требование p99 < 30 мс критично для мониторинга пациентов в реальном времени |
| **Скорость данных** | 1000+ событий/сек | 1000+ событий/сек | Оба подхода справляются с нагрузкой, но Flink обеспечивает меньшую задержку |
| **Требования к надежности** | Exactly-once semantics | Exactly-once semantics | Оба поддерживают, но Flink имеет более легковесный checkpointing |
| **Обработка late data** | Watermark + allowedLateness | Watermark-based | Flink предлагает более гибкие стратегии через side outputs |
| **State management** | Managed state с TTL | State management через DStreams | Flink предоставляет лучшую производительность для stateful операций |

**Сравнение затрат:**

| Аспект | True Streaming (Flink) | Micro-batch (Spark) |
|--------|------------------------|---------------------|
| **Вычислительные ресурсы** | Средние (оптимизирован для low-latency) | Высокие (требует больше памяти для batch) |
| **Операционные затраты** | Низкие (простая настройка checkpointing) | Средние (требует настройки batch intervals) |
| **Время разработки** | Низкое (прямолинейный API) | Среднее (требует настройки микро-батчей) |

### 2. Архитектура системы

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Data Sources  │───▶│   Kafka Cluster  │───▶│  Flink Cluster  │
│                 │    │                  │    │                  │
│ • Patient IoT   │    │ • Topics:        │    │ • Streaming Jobs │
│ • EHR Systems   │    │   - patient-data │    │ • Stateful       │
│ • Medical Devices│   │   - predictions  │    │   Processing     │
└─────────────────┘    └──────────────────┘    └─────────┬────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐              │
│   Monitoring    │    │   ML Serving     │◀────────────┘
│                 │    │                  │
│ • Grafana       │    │ • FastAPI Model  │
│ • Prometheus    │    │ • Redis Features │
│ • Alerting      │    │ • Model Registry │
└─────────────────┘    └──────────────────┘
```

**Компоненты архитектуры:**

1. **Источники данных**: IoT устройства пациентов, EHR системы
2. **Ingestion Layer**: Kafka для буферизации и распределения нагрузки
3. **Processing Layer**: Flink для потоковой обработки
4. **ML Serving**: FastAPI + Redis для online инференса
5. **Monitoring**: Prometheus + Grafana для мониторинга
6. **Storage**: Redis для state и feature store

### 3. Выбор технологии потоковой обработки

**Выбор: Apache Flink**

**Обоснование:**

| Требование | Решение в Flink | Преимущество перед Spark Streaming |
|------------|-----------------|-----------------------------------|
| **p99 < 30 мс** | True streaming с event-time processing | Spark micro-batch имеет минимум 100 мс задержку |
| **Exactly-once** | Lightweight checkpointing | Более эффективный механизм чем Spark's WAL |
| **State management** | Managed keyed state с TTL | Лучшая производительность для оконных операций |
| **Late data** | Watermark + allowedLateness + side outputs | Более гибкая обработка чем в Spark |
| **Resource usage** | Оптимизирован для low-latency | Меньший overhead чем micro-batch подход |

**Конкретные преимущества Flink для нашего случая:**
- Нативная поддержка event-time processing
- Эффективная обработка out-of-order событий
- Low-latency без компромиссов в throughput
- Простота настройки state TTL для медицинских данных

### 4. Обработка late-arriving данных

**Допустимая задержка: 5 минут**

**Стратегия: Watermark + allowedLateness + Side Outputs**

**Обоснование стратегии:**
- **Watermark (2 мин)**: Определяет когда закрывать окна
- **AllowedLateness (3 мин)**: Позволяет обновлять результаты окон
- **Side Outputs**: Обработка данных пришедших позже 5 минут

### 5. Матрица требований

| Требование | Решение | Обоснование |
|------------|---------|-------------|
| **Задержка p99 < 30 мс** | Flink true streaming + оптимизированная модель | Flink обеспечивает sub-100ms задержку, модель оптимизирована для быстрого инференса |
| **Exactly-once semantics** | Checkpointing каждые 500 мс + Kafka transactions | Гарантирует обработку без потерь и дубликатов даже при сбоях |
| **Обработка late-arriving данных** | Watermark(2min) + allowedLateness(3min) + side outputs | Позволяет обрабатывать данные задержкой до 5 минут с разными стратегиями |
| **State management** | Managed keyed state с TTL 24 часа | Автоматическая очистка устаревших состояний пациентов |
| **Интеграция с ML** | Online инференс через FastAPI + кэширование в Redis | Обеспечивает задержку инференса < 20 мс |
| **Мониторинг** | Prometheus metrics + Grafana dashboards | Позволяет отслеживать задержки, throughput и качество данных в реальном времени |
| **Fault tolerance** | Automatic restart с exponential backoff | Система восстанавливается после сбоев без потери данных |
| **Обработка out-of-order** | Event-time processing + watermarks | Корректная обработка данных пришедших не по порядку |
| **High velocity (1000+ events/sec)** | Parallelism 8 + оптимизированные операторы | Обработка высокой нагрузки с горизонтальным масштабированием |
| **Data drift detection** | Monitoring PSI + алерты в Prometheus | Обнаружение изменений в распределении данных |
