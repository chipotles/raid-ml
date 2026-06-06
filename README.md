# Telco Customer Churn Prediction

Проект по предсказанию оттока клиентов телеком-оператора. Модель помогает маркетинговой команде заранее выявлять клиентов, которые собираются уйти, чтобы позвонить им до отмены подписки.

---

## Структура репозитория

```
project/
├── README.md
├── requirements.txt
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── new_customers.csv
├── notebook/
│   └── EDA.ipynb
├── scripts/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
└── results/
    ├── plots/
    │   ├── pr_curve_oof.png
    │   └── confusion_matrix.png
    ├── predictions.csv
    ├── churn_pipeline.pkl
    └── results.md
```

---

## Запуск

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/predict.py
```

---

## Данные

Датасет: **IBM Telco Customer Churn** (~7000 строк, 21 колонка).  
Целевая переменная: `Churn` (Yes/No → 1/0).  
Дисбаланс классов: **73% не уходят, 27% уходят**.

**Ловушка в данных:** колонка `TotalCharges` загружается как `object` из-за пустых строк. Обрабатывается через `pd.to_numeric(..., errors='coerce')` внутри Pipeline, пропуски заполняются медианой через `SimpleImputer`.

---

## Почему не accuracy

Majority-class baseline даёт **~73% accuracy**, просто предсказывая "не уйдёт" для всех клиентов. Это бесполезная метрика — модель ничего не предсказывает, но выглядит хорошо.

Пропустить реального уходящего клиента (false negative) **в 5 раз дороже**, чем позвонить лояльному клиенту зря (false positive). Поэтому основная метрика — **recall** по классу churn.

---

## Feature Engineering

Все признаки создаются внутри Pipeline через кастомный трансформер `FeatureEngineer` — утечки данных нет, признаки автоматически воспроизводятся при инференсе.

| Признак | Формула | Обоснование |
|---|---|---|
| `tenure_bucket` | bins: 0–12, 13–24, 25–48, 49+ | Новые клиенты уходят значительно чаще — нелинейная зависимость от срока |
| `charges_per_month` | `TotalCharges / max(tenure, 1)` | Убирает влияние срока на сумму — показывает реальную месячную нагрузку |
| `n_services` | Количество активных услуг (не `No` / `No internet service`) | Чем больше услуг — тем сложнее клиенту уйти |
| `is_new_customer` | `1` если `tenure ≤ 12` | Бинарный флаг высокого риска для новых клиентов |

> **Важно:** `n_services` считается правильно — исключает `No`, `No internet service` и `No phone service`. Наивный подсчёт через `== "Yes"` молча занижает результат для `InternetService`.

---

## Pipeline

```
TotalChargesConverter         # object → float, пустые строки → NaN
FeatureEngineer               # создание новых признаков
ColumnDropper                 # удаление customerID
ColumnTransformer
  ├── числовые → SimpleImputer(median) → StandardScaler
  └── категориальные → OneHotEncoder(handle_unknown='ignore')
Classifier
```

Все шаги препроцессинга фитируются **только на тренировочных данных** внутри Pipeline. Никакого leakage.

---

## Сравнение моделей (5-fold Stratified CV)

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Dummy (most_frequent) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Dummy (stratified) | 0.278 ± 0.025 | 0.275 ± 0.025 | 0.276 ± 0.025 | 0.507 ± 0.017 |
| Logistic Regression | 0.796 ± 0.034 | 0.518 ± 0.017 | 0.628 ± 0.021 | 0.846 ± 0.011 |
| Random Forest | 0.462 ± 0.021 | 0.638 ± 0.032 | 0.535 ± 0.023 | 0.826 ± 0.010 |
| Gradient Boosting | 0.520 ± 0.022 | 0.651 ± 0.038 | 0.578 ± 0.024 | 0.842 ± 0.011 |

**Лучшая модель: Logistic Regression** — наивысший recall (0.796) при конкурентном ROC-AUC (0.846). Все три модели значительно превосходят оба baseline по recall и ROC-AUC.

---

## Подбор гиперпараметров

GridSearchCV на полном Pipeline, `scoring='recall'`, 5-fold CV:

```python
param_grid = {
    'model__C':      [0.01, 0.1, 1, 10],
    'model__solver': ['lbfgs', 'saga'],
}
```

**Лучшие параметры:** `C=10`, `solver='saga'`  
**CV recall после тюнинга:** 0.798

---

## Подбор порога

Порог по умолчанию (0.5) — не бизнес-решение. Порог выбирался на OOF-вероятностях тренировочного набора (`cross_val_predict`), тест-сет не трогался.

Цель: **recall ≥ 0.80** с максимальной доступной precision.

**Замороженный порог: 0.497**  
OOF precision: 0.518 | OOF recall: 0.800

---

## Финальные результаты на тест-сете

| Метрика | Значение | Требование |
|---|---|---|
| **Recall (churn)** | **0.799** | ≥ 0.75 ✅ |
| **Precision (churn)** | **0.501** | ≥ 0.45 ✅ |
| **ROC-AUC** | **0.841** | — |

### Confusion Matrix

|  | Pred: No churn | Pred: Churn |
|---|---|---|
| **Actual: No churn** | 737 | 298 |
| **Actual: Churn** | 75 | 299 |

---

## Бизнес-интерпретация

При пороге **0.497** модель помечает примерно **425 клиентов из 1409** как потенциально уходящих.  
Из них **299 — реальные уходящие клиенты** (true positive).  
Только **75 реальных уходящих** остаются незамеченными (false negative).

На каждые **1000 клиентов** это означает:
- ~**300 звонков** маркетинга в неделю
- ~**212 из них** — реальные уходящие клиенты, которых можно удержать
- Пропускается только ~**53 реальных уходящих** из 1000

Учитывая что пропущенный уходящий клиент в **5× дороже** лишнего звонка — модель экономически оправдана.

---

## Предсказание на новых клиентах

```bash
python scripts/predict.py
```

Принимает `data/new_customers.csv` (без колонки `Churn`), сохраняет `results/predictions.csv`:

| customerID | churn_pred | churn_proba |
|---|---|---|
| 1024-GUALD | 1 | 0.8345 |
| 0484-JPBRU | 0 | 0.1622 |
| 3620-EHIMZ | 0 | 0.0200 |
| 6910-HADCM | 1 | 0.8193 |
| 8587-XYZSF | 0 | 0.0235 |
