"""
train.py
========
Запуск из корня репозитория:
    python scripts/train.py

Что делает скрипт:
  1. Загружает данные, подготавливает целевую переменную.
  2. Стратифицированный сплит 80/20 (тест трогается только один раз).
  3. Baseline: DummyClassifier.
  4. 5-fold CV для трёх классификаторов → таблица метрик.
  5. GridSearchCV на лучшей модели (весь Pipeline).
  6. Подбор порога по OOF-вероятностям (recall ≥ 0.80).
  7. Финальная оценка на тест-сете с замороженным порогом.
  8. Сохранение Pipeline и confusion matrix.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Импорт кастомных трансформеров из того же пакета scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from preprocessing import (
    CAT_COLS,
    NUM_COLS,
    ColumnDropper,
    FeatureEngineer,
    TotalChargesConverter,
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────
DATA_PATH     = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
RESULTS_DIR   = 'results'
PLOTS_DIR     = os.path.join(RESULTS_DIR, 'plots')
PIPELINE_PATH = os.path.join(RESULTS_DIR, 'churn_pipeline.pkl')
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5
RECALL_TARGET = 0.80   # минимальный recall при подборе порога


# ─────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────
def make_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def build_preprocessor():
    """ColumnTransformer: числовые → impute+scale, категориальные → OHE."""
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', num_pipe, NUM_COLS),
        ('cat', cat_pipe, CAT_COLS),
    ], remainder='drop')


def build_pipeline(classifier):
    """Полный Pipeline: конвертация → инженерия → дроп ID → препроцессинг → модель."""
    return Pipeline([
        ('convert',  TotalChargesConverter()),
        ('engineer', FeatureEngineer()),
        ('drop',     ColumnDropper()),
        ('prep',     build_preprocessor()),
        ('model',    classifier),
    ])


def cv_metrics(pipe, X, y, cv, label=''):
    """Запускает cross_validate и возвращает строку для сводной таблицы."""
    scoring = ['recall', 'precision', 'f1', 'roc_auc']
    scores  = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    row = {'Model': label}
    for metric in scoring:
        vals = scores[f'test_{metric}']
        row[metric.capitalize()] = f"{vals.mean():.3f} ± {vals.std():.3f}"
    return row


def find_threshold(y_true, y_proba, recall_target=RECALL_TARGET):
    """
    Находит наименьший порог, при котором recall >= recall_target,
    и возвращает (threshold, precision, recall).
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve возвращает массивы длиной n+1; thresholds длиной n
    for prec, rec, thr in zip(precision_arr[:-1], recall_arr[:-1], thresholds):
        if rec >= recall_target:
            return float(thr), float(prec), float(rec)
    # Если ни один порог не даёт нужный recall — берём минимальный из thresholds
    return float(thresholds[0]), float(precision_arr[0]), float(recall_arr[0])


def plot_pr_curve(y_true, y_proba, threshold, save_path):
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(recall_arr, precision_arr, lw=2, label='PR curve')
    # Точка выбранного порога
    thr_idx = np.searchsorted(thresholds, threshold)
    plt.scatter(recall_arr[thr_idx], precision_arr[thr_idx],
                s=120, zorder=5, color='red', label=f'threshold={threshold:.2f}')
    plt.axvline(RECALL_TARGET, linestyle='--', color='grey',
                label=f'recall target={RECALL_TARGET}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (OOF probabilities)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  → PR-кривая сохранена: {save_path}")


def confusion_df(y_true, y_pred):
    """Confusion matrix как читаемый DataFrame."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index  =['Actual: No churn', 'Actual: Churn'],
        columns=['Pred: No churn',   'Pred: Churn'],
    )


# ─────────────────────────────────────────────
# Основной скрипт
# ─────────────────────────────────────────────
def main():
    make_dirs()

    # ── 1. Загрузка данных ──────────────────────────────────────────
    print("=" * 60)
    print("1. Загрузка данных")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH)
    save_new_customers(df)  
    print(f"   Размер датасета: {df.shape}")

    # Целевая переменная
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    print(f"   Распределение Churn: {y.value_counts().to_dict()}")
    print(f"   Доля оттока: {y.mean():.1%}")

    # ── 2. Train/test split ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Train/test split (80/20, стратифицированный)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"   Train: {X_train.shape[0]} строк | Test: {X_test.shape[0]} строк")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── 3. Baseline ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Baseline — DummyClassifier")
    print("=" * 60)
    results_rows = []
    for strategy in ('most_frequent', 'stratified'):
        dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
        row = cv_metrics(dummy, X_train, y_train, cv, label=f'Dummy ({strategy})')
        results_rows.append(row)
        print(f"   {row['Model']:30s}  Recall={row['Recall']}  ROC-AUC={row['Roc_auc']}")

    # ── 4. Три классификатора — 5-fold CV ───────────────────────────
    print("\n" + "=" * 60)
    print("4. Сравнение классификаторов (5-fold stratified CV)")
    print("=" * 60)

    classifiers = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', n_estimators=200, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE),
    }

    cv_results = {}
    for name, clf in classifiers.items():
        pipe = build_pipeline(clf)
        row  = cv_metrics(pipe, X_train, y_train, cv, label=name)
        results_rows.append(row)
        cv_results[name] = row
        print(f"   {name:25s}  Recall={row['Recall']}  ROC-AUC={row['Roc_auc']}")

    metrics_df = pd.DataFrame(results_rows)
    print("\n── Сводная таблица метрик ──")
    print(metrics_df.to_string(index=False))

    # ── 5. Выбор лучшей модели ──────────────────────────────────────
    # Выбираем по среднему recall (первое число до " ± ")
    best_name = max(
        cv_results,
        key=lambda n: float(cv_results[n]['Recall'].split(' ')[0])
    )
    print(f"\n   Лучшая модель по recall: {best_name}")

    # ── 6. Подбор гиперпараметров (GridSearchCV) ────────────────────
    print("\n" + "=" * 60)
    print("6. GridSearchCV на лучшей модели")
    print("=" * 60)

    best_clf_base = classifiers[best_name]
    best_pipe     = build_pipeline(best_clf_base)

    if best_name == 'Logistic Regression':
        param_grid = {
            'model__C':       [0.01, 0.1, 1, 10],
            'model__solver':  ['lbfgs', 'saga'],
        }
    elif best_name == 'Random Forest':
        param_grid = {
            'model__n_estimators': [100, 300],
            'model__max_depth':    [None, 10, 20],
            'model__min_samples_leaf': [1, 5],
        }
    else:  # Gradient Boosting
        param_grid = {
            'model__n_estimators':  [100, 300],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth':     [3, 5],
        }

    grid_search = GridSearchCV(
        best_pipe,
        param_grid,
        cv=cv,
        scoring='recall',
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"   Лучшие параметры : {grid_search.best_params_}")
    print(f"   Лучший CV recall : {grid_search.best_score_:.3f}")

    tuned_pipe = grid_search.best_estimator_

    # ── 7. Подбор порога на OOF-вероятностях ───────────────────────
    print("\n" + "=" * 60)
    print("7. Подбор порога (OOF, recall ≥ {})".format(RECALL_TARGET))
    print("=" * 60)

    oof_proba = cross_val_predict(
        tuned_pipe, X_train, y_train,
        cv=cv, method='predict_proba', n_jobs=-1,
    )[:, 1]

    threshold, thr_prec, thr_rec = find_threshold(y_train, oof_proba)
    print(f"   Замороженный порог : {threshold:.4f}")
    print(f"   OOF precision      : {thr_prec:.3f}")
    print(f"   OOF recall         : {thr_rec:.3f}")

    # Сохраняем порог чтобы predict.py его подхватил
    with open(os.path.join(RESULTS_DIR, 'threshold.txt'), 'w') as f:
        f.write(str(threshold))

    plot_pr_curve(
        y_train, oof_proba, threshold,
        save_path=os.path.join(PLOTS_DIR, 'pr_curve_oof.png'),
    )

    # ── 8. Финальная оценка на тест-сете ───────────────────────────
    print("\n" + "=" * 60)
    print("8. Финальная оценка на тест-сете (один раз!)")
    print("=" * 60)

    # Дообучаем tuned_pipe на всём трейне (grid_search уже сделал refit=True)
    test_proba = tuned_pipe.predict_proba(X_test)[:, 1]
    y_pred     = (test_proba >= threshold).astype(int)

    print("\n── Classification report ──")
    print(classification_report(y_test, y_pred, target_names=['No churn', 'Churn']))

    cm_df = confusion_df(y_test, y_pred)
    print("\n── Confusion matrix ──")
    print(cm_df)

    # Сохраняем confusion matrix как картинку
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['No churn', 'Churn'],
        colorbar=False, ax=ax,
    )
    ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=120)
    plt.close()
    print(f"  → Confusion matrix сохранена: {os.path.join(PLOTS_DIR, 'confusion_matrix.png')}")

    # Ключевые метрики для проверки требований задания
    from sklearn.metrics import precision_score, recall_score
    test_recall    = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_roc_auc   = roc_auc_score(y_test, test_proba)

    print(f"\n   Test recall    : {test_recall:.3f}  (требование ≥ 0.75)")
    print(f"   Test precision : {test_precision:.3f}  (требование ≥ 0.45)")
    print(f"   Test ROC-AUC   : {test_roc_auc:.3f}")

    if test_recall < 0.75:
        print("   ⚠️  ВНИМАНИЕ: recall ниже требования 0.75!")
    if test_precision < 0.45:
        print("   ⚠️  ВНИМАНИЕ: precision ниже требования 0.45!")

    # ── 9. Сохранение Pipeline ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("9. Сохранение Pipeline")
    print("=" * 60)

    joblib.dump(tuned_pipe, PIPELINE_PATH)
    print(f"   Pipeline сохранён: {PIPELINE_PATH}")

    # ── 10. results.md ──────────────────────────────────────────────
    n_flagged   = int(y_pred.sum())
    n_test      = len(y_test)
    n_true_pos  = int(((y_pred == 1) & (y_test == 1)).sum())
    scale_1000  = 1000 / n_test

    results_md = f"""# Model Results

## Метрика и обоснование
- **Целевая метрика:** recall (churn class)
- **Причина:** пропустить реального уходящего клиента в 5× дороже,
  чем позвонить лояльному клиенту. Majority-class accuracy даёт ~73%,
  ничего не предсказывая — это бесполезный baseline.

## Сводная таблица CV

{metrics_df.to_markdown(index=False)}

## Лучшая модель
- **Модель:** {best_name}
- **Лучшие гиперпараметры:** `{grid_search.best_params_}`
- **CV recall (tuned):** {grid_search.best_score_:.3f}

## Порог отсечения
- **Порог:** {threshold:.4f} (выбран на OOF-данных, цель recall ≥ {RECALL_TARGET})
- **OOF precision при пороге:** {thr_prec:.3f}
- **OOF recall при пороге:**    {thr_rec:.3f}

## Финальные метрики на тест-сете
| Метрика        | Значение | Требование |
|----------------|----------|------------|
| Recall (churn) | {test_recall:.3f}    | ≥ 0.75     |
| Precision      | {test_precision:.3f}    | ≥ 0.45     |
| ROC-AUC        | {test_roc_auc:.3f}    | —          |

## Бизнес-интерпретация
При пороге **{threshold:.2f}** модель помечает примерно
**{round(n_flagged * scale_1000):.0f} клиентов из 1000** как потенциальных уходящих.
Из них около **{round(n_true_pos * scale_1000):.0f}** — реальные уходящие клиенты.
Это означает ~{round(n_flagged * scale_1000):.0f} звонков маркетинга в неделю
(на 1000 клиентов), из которых ~{round(n_true_pos * scale_1000):.0f} предотвращают реальный отток.
"""

    results_path = os.path.join(RESULTS_DIR, 'results.md')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(results_md)
    print(f"   results.md сохранён: {results_path}")

    print("\n✓ Обучение завершено успешно.")


def save_new_customers(df_raw: pd.DataFrame, n: int = 5):
    """Сохраняет n случайных строк без колонки Churn для тестирования predict.py"""
    path = os.path.join('data', 'new_customers.csv')
    if not os.path.exists(path):
        sample = df_raw.drop(columns=['Churn']).sample(n, random_state=RANDOM_STATE)
        sample.to_csv(path, index=False)
        print(f"   new_customers.csv создан: {path}")

if __name__ == '__main__':
    main()