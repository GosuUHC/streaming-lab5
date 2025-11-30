import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_data():
    """Загрузка и предобработка данных"""
    try:
        # Используем предоставленные данные для обучения
        df = pd.read_csv('hospital_readmissions_30k.csv')
        logger.info(f"Data loaded: {df.shape}")

        # Предобработка данных
        df = preprocess_data(df)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Создаем синтетические данные если файл не найден
        return create_synthetic_data()


def preprocess_data(df):
    """Предобработка данных"""
    # Копируем данные
    df_processed = df.copy()

    # Обработка blood_pressure (разделяем на systolic и diastolic)
    df_processed[['systolic_bp', 'diastolic_bp']] = df_processed['blood_pressure'].str.split(
        '/', expand=True).astype(float)

    # Кодирование категориальных переменных
    df_processed['gender'] = df_processed['gender'].map(
        {'Male': 0, 'Female': 1, 'Other': 2})
    df_processed['discharge_destination'] = df_processed['discharge_destination'].map({
        'Home': 0, 'Nursing_Facility': 1, 'Rehab': 2
    })
    df_processed['diabetes'] = df_processed['diabetes'].map(
        {'Yes': 1, 'No': 0})
    df_processed['hypertension'] = df_processed['hypertension'].map({
                                                                    'Yes': 1, 'No': 0})
    df_processed['readmitted_30_days'] = df_processed['readmitted_30_days'].map({
                                                                                'Yes': 1, 'No': 0})

    # Удаляем исходную колонку blood_pressure
    df_processed = df_processed.drop('blood_pressure', axis=1)

    return df_processed


def create_synthetic_data():
    """Создание синтетических данных если файл не найден"""
    logger.info("Creating synthetic data...")
    np.random.seed(42)
    n_samples = 1000

    data = {
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice([0, 1, 2], n_samples),
        'systolic_bp': np.random.randint(100, 180, n_samples),
        'diastolic_bp': np.random.randint(60, 120, n_samples),
        'cholesterol': np.random.randint(150, 300, n_samples),
        'bmi': np.random.uniform(18, 40, n_samples),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'hypertension': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'medication_count': np.random.randint(0, 11, n_samples),
        'length_of_stay': np.random.randint(1, 11, n_samples),
        'discharge_destination': np.random.choice([0, 1, 2], n_samples)
    }

    # Создаем целевой признак на основе комбинации факторов
    risk_factors = (
        data['age'] * 0.01 +
        data['systolic_bp'] * 0.005 +
        data['diabetes'] * 0.3 +
        data['hypertension'] * 0.2 +
        data['medication_count'] * 0.05 +
        (data['bmi'] > 30) * 0.15
    )

    # Преобразуем в вероятность и генерируем бинарный таргет
    probabilities = 1 / (1 + np.exp(-risk_factors))
    data['readmitted_30_days'] = np.random.binomial(1, probabilities)

    return pd.DataFrame(data)


def train_model(df):
    """Обучение модели"""
    # Определяем фичи и таргет
    feature_columns = [
        'age', 'gender', 'systolic_bp', 'diastolic_bp', 'cholesterol',
        'bmi', 'diabetes', 'hypertension', 'medication_count',
        'length_of_stay', 'discharge_destination'
    ]

    X = df[feature_columns]
    y = df['readmitted_30_days']

    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Обучаем легковесную модель
    model = RandomForestClassifier(
        n_estimators=50,  # Меньше деревьев для скорости
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("\nFeature Importance:")
    logger.info(feature_importance)

    return model, feature_columns


def save_model(model, feature_columns):
    """Сохранение модели и метаданных"""
    joblib.dump(model, 'readmission_model.joblib')

    # Сохраняем информацию о фичах
    model_metadata = {
        'feature_columns': feature_columns,
        'model_type': 'RandomForest',
        'version': '1.0'
    }

    joblib.dump(model_metadata, 'model_metadata.joblib')
    logger.info("Model and metadata saved successfully")


if __name__ == '__main__':
    logger.info("Starting model training...")

    # Загрузка и предобработка данных
    df = load_and_preprocess_data()
    logger.info(f"Processed data shape: {df.shape}")
    logger.info(f"Readmission rate: {df['readmitted_30_days'].mean():.3f}")

    # Обучение модели
    model, feature_columns = train_model(df)

    # Сохранение модели
    save_model(model, feature_columns)
    logger.info("Model training completed!")
