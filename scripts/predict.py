"""
predict.py
==========
Запуск из корня репозитория:
    python scripts/predict.py

Что делает скрипт:
  1. Загружает сохранённый Pipeline из results/churn_pipeline.pkl
  2. Читает data/new_customers.csv (без колонки Churn)
  3. Откладывает customerID, делает predict_proba, присоединяет обратно
  4. Сохраняет results/predictions.csv (customerID, churn_pred, churn_proba)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Кастомные трансформеры должны быть импортируемы при десериализации pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from preprocessing import (   # noqa: F401  (нужны для pickle)
    ColumnDropper,
    FeatureEngineer,
    TotalChargesConverter,
)

# ─────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────
PIPELINE_PATH   = os.path.join('results', 'churn_pipeline.pkl')
NEW_CUSTOMERS   = os.path.join('data',    'new_customers.csv')
PREDICTIONS_OUT = os.path.join('results', 'predictions.csv')

# Замороженный порог — должен совпадать с тем, что выбран в train.py.
# Если хочешь передавать его динамически, можно сохранять в results/threshold.txt
# и читать здесь. Пока хардкодим дефолт; train.py напечатает реальное значение.
DEFAULT_THRESHOLD = 0.35


def load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pipeline не найден: {path}\n"
            "Сначала запусти: python scripts/train.py"
        )
    return joblib.load(path)


def load_threshold(results_dir: str = 'results') -> float:
    """
    Читает порог из results/threshold.txt (если train.py его сохранил).
    Иначе возвращает DEFAULT_THRESHOLD.
    """
    thr_path = os.path.join(results_dir, 'threshold.txt')
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            thr = float(f.read().strip())
        print(f"   Порог загружен из файла: {thr:.4f}")
        return thr
    print(f"   threshold.txt не найден — используем дефолт {DEFAULT_THRESHOLD}")
    return DEFAULT_THRESHOLD


def predict(pipeline, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Принимает сырой DataFrame (без колонки Churn).
    Возвращает DataFrame с колонками: customerID, churn_pred, churn_proba.
    """
    # Сохраняем customerID до того, как pipeline его удалит
    if 'customerID' in df.columns:
        customer_ids = df['customerID'].reset_index(drop=True)
    else:
        customer_ids = pd.Series(range(len(df)), name='customerID')

    # pipeline внутри сам вызовет ColumnDropper — customerID не попадёт в модель
    proba      = pipeline.predict_proba(df)[:, 1]
    pred_label = (proba >= threshold).astype(int)

    return pd.DataFrame({
        'customerID':  customer_ids,
        'churn_pred':  pred_label,
        'churn_proba': np.round(proba, 4),
    })


def main():
    print("=" * 60)
    print("Предсказание оттока для новых клиентов")
    print("=" * 60)

    # 1. Загрузка pipeline
    print(f"\n1. Загрузка pipeline из {PIPELINE_PATH}")
    pipeline = load_pipeline(PIPELINE_PATH)

    # 2. Загрузка порога
    print("\n2. Загрузка порога отсечения")
    threshold = load_threshold()

    # 3. Загрузка новых клиентов
    print(f"\n3. Загрузка новых клиентов из {NEW_CUSTOMERS}")
    if not os.path.exists(NEW_CUSTOMERS):
        raise FileNotFoundError(
            f"Файл не найден: {NEW_CUSTOMERS}\n"
            "Создай data/new_customers.csv (5 строк из датасета без колонки Churn)."
        )
    df_new = pd.read_csv(NEW_CUSTOMERS)
    print(f"   Строк: {len(df_new)}, Колонок: {len(df_new.columns)}")

    if 'Churn' in df_new.columns:
        print("   ⚠️  Найдена колонка Churn — удаляем перед предсказанием.")
        df_new = df_new.drop(columns=['Churn'])

    # 4. Предсказание
    print("\n4. Предсказание...")
    results = predict(pipeline, df_new, threshold)

    # 5. Вывод в консоль
    print("\n── Результаты ──")
    print(results.to_string(index=False))
    print(f"\n   Помечено как churner: {results['churn_pred'].sum()} из {len(results)}")

    # 6. Сохранение
    os.makedirs('results', exist_ok=True)
    results.to_csv(PREDICTIONS_OUT, index=False)
    print(f"\n✓ Предсказания сохранены: {PREDICTIONS_OUT}")


if __name__ == '__main__':
    main()