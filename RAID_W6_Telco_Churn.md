
### Overview

The goal of this project is to build a customer-churn prediction model for a telecom operator using a real, messy customer dataset. You will analyze the data, engineer features, build a leakage-free scikit-learn `Pipeline`, choose and justify a metric appropriate for class imbalance, compare several classifiers, tune a decision threshold against a business constraint, and translate your results for a non-technical stakeholder.

### Role Play

You are a junior ML engineer at a telecom company. The CMO believes the company is losing money to customer churn, but the marketing team can't tell *which* customers are at risk of leaving. Your job is to build a model the marketing team can use to call at-risk customers before they cancel. The cost of a false negative (missing a real churner) is roughly **5× higher** than the cost of a false positive (calling a happy customer who would have stayed). Your metric and your threshold choice must reflect this asymmetry.

### Learning Objectives

By the end of this project, you will be able to:

1. Perform exploratory data analysis on a mixed-type dataset (numeric, categorical, boolean).
2. Identify and correctly handle implicit missing values (e.g. empty strings in a numeric column).
3. Engineer features from raw billing and contract data.
4. Build a scikit-learn `Pipeline` with a `ColumnTransformer` that prevents data leakage.
5. Choose and justify an evaluation metric for imbalanced classes instead of default accuracy.
6. Compare classifiers using stratified cross-validation and light hyperparameter tuning.
7. Tune a decision threshold on validation data against a business recall target.
8. Evaluate the final model on a held-out test set that is touched only once.
9. Serialize the trained Pipeline and make predictions on new customers.
10. Document your process and results clearly for a non-technical stakeholder.

### Instructions

#### Data

The dataset is the **IBM Telco Customer Churn** dataset (~7000 rows, 21 columns). See **Resources** for the download links; place the file at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` and reference it by a relative path from the repo root.

- `Churn` is the target (`Yes`/`No` → 1/0).
- `TotalCharges` looks numeric but loads as `object` because some rows are empty strings. This is a deliberate trap: use `pd.to_numeric(df['TotalCharges'], errors='coerce')` and impute the resulting `NaN`s inside your Pipeline.
- Class balance is ~73% non-churn, ~27% churn. **Accuracy is meaningless here** — plan around the imbalance from day one.

#### 1. EDA and feature engineering

- Create a Jupyter Notebook to perform EDA. **This notebook is not evaluated.** It should at least contain:
  - `df.dtypes`, `df.shape`, `df.head()`, and a written note of which columns are mistyped.
  - Distribution of `Churn` (bar chart) and of each numeric column (histogram).
  - Missing-value analysis for both explicit `NaN` and implicit empties (e.g. `" "` in `TotalCharges`).
  - Cross-tabulation of `Churn` by at least three categorical columns (`Contract`, `PaymentMethod`, `InternetService`).
  - Two or three written insights, e.g. "month-to-month contracts churn at 43% versus 3% for two-year contracts".

- Engineer **at least two** features beyond the raw data, and justify each one in the `README.md`. Examples (you are not required to use these):
  - `tenure_bucket`: bins of `tenure` (0–12, 13–24, 25–48, 49+).
  - `charges_per_month_of_tenure`: `TotalCharges / max(tenure, 1)`.
  - `n_services`: count of *active* services. Note these columns are **not** all `Yes`/`No`: `InternetService ∈ {DSL, Fiber optic, No}`, and the add-on columns use `No internet service` as a third value. Count a service as active when its value is **neither `No` nor `No internet service`**. A naive `== "Yes"` count silently scores `InternetService` as 0 — that is the trap.

> All feature engineering must be implemented **inside the Pipeline** (a custom `BaseEstimator/TransformerMixin` transformer, or a `FunctionTransformer` for row-local features), so it is reproduced identically at inference and cannot leak. Do not engineer features in standalone pandas before fitting.

#### 2. Pipeline construction

Use a single scikit-learn `Pipeline` containing a `ColumnTransformer` for preprocessing followed by a classifier:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
    ]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])

pipe = Pipeline([
    ('prep',  preprocessor),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000)),
])
```

**Hard rule:** all fitting of `StandardScaler`, `SimpleImputer`, `OneHotEncoder` must happen **inside** the Pipeline, on the training fold only. Fitting any preprocessor on the full dataset before the split is data leakage — the most common way to fail this project.

#### 3. Model selection

Split the data so that the test set is never seen during model development:

```console
DATA
└─── WA_Fn-UseC_-Telco-Customer-Churn.csv
     └───── Train (80%)
     │          Stratified 5-fold CV:
     │                 Fold k → fit on k-1 folds, validate on the held-out fold
     │          (threshold is chosen here, on validation / out-of-fold data)
     └───── Test (20%)  ← scored only once, at the very end
```

**Rules:**

- Stratified train/test split (`stratify=y`, `test_size=0.2`, fixed `random_state`). The test set is touched only once, at the end.
- First establish a baseline with `DummyClassifier` (`most_frequent` and `stratified`). **Every model below must beat it** on recall and ROC-AUC. Quote the ~73% majority-class accuracy in the `README.md` to justify discarding accuracy.
- Train and compare **three** classifiers (swap only the final Pipeline step):
  - Logistic Regression with `class_weight='balanced'`
  - Random Forest
  - Gradient Boosting (`GradientBoostingClassifier`)
- Use 5-fold **stratified** cross-validation and report **mean and standard deviation** of recall, precision, F1, and ROC-AUC (churn class). Write the table in the `README.md`.
- **Light hyperparameter tuning on the best model only:** tune at least two hyperparameters with `GridSearchCV`/`RandomizedSearchCV` (or the manual loop from the Week 5 workshop). The search **must wrap the full Pipeline** (`'model__param'` naming) and set `scoring` explicitly (e.g. `'recall'` or `'average_precision'`) — never the accuracy default.
- Display the confusion matrix for the best model as a labelled DataFrame (True label / Predicted label).
- Save the trained Pipeline as `results/churn_pipeline.pkl` (`joblib.dump` or `mlflow.sklearn.log_model`).

> Advice: get the train → test → predict loop working with default hyperparameters first; tune only once everything runs end-to-end. If your CV scores look like 0.99, you have leakage — find it before tuning.

#### 4. Threshold tuning

The default 0.5 cutoff is not a business decision. **The threshold is a hyperparameter: choose it on validation data, never on the test set.**

- Generate out-of-fold churn probabilities on the **training set** (`cross_val_predict(..., method='predict_proba')`), or use an explicit validation split carved from train.
- Plot the precision-recall curve from those probabilities and find the threshold giving **recall ≥ 0.80** with the highest available precision. Freeze it.
- Apply the frozen threshold to the held-out test set **once**, and build the confusion matrix.
- Translate the result into business language, e.g. "At threshold 0.31 the model flags ~180 customers per 1000, of whom ~90 are real churners → ~180 marketing calls per week." Put this in `results/results.md`.

**Rules:**

- Test-set recall (churn class) ≥ **0.75** at your frozen threshold. Write the result in the `README.md`.
- Test-set precision (churn class) ≥ **0.45** at your frozen threshold. Write the result in the `README.md`.
- Choosing or adjusting the threshold using test-set data is leakage and fails the project.

#### 5. Predict

Once you are confident in the model, predict on new customers:

- Load the saved Pipeline.
- Read a CSV of new customers in the same schema as training data, **without** the `Churn` column. Provide a small `data/new_customers.csv` (5 rows taken from the raw data with `Churn` dropped) to test against.
- Set `customerID` aside, drop it before `predict_proba` (it is an identifier, not a feature), then re-attach it to the output by index.
- Save predictions to `results/predictions.csv` with columns `customerID, churn_pred, churn_proba`.
- The script must run from the repo root with `python scripts/predict.py` after `pip install -r requirements.txt`. If you use custom transformers, their class definitions must live in `scripts/` so they are importable when the Pipeline is unpickled.

### Project repository structure:

```console
project
│   README.md
│   requirements.txt
│
└───data
│   │   WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │   new_customers.csv
│
└───notebook
│   │   EDA.ipynb
│
└───scripts
│   │   preprocessing.py
│   │   train.py
│   │   predict.py
│
└───results
    │   plots
    │   predictions.csv
    │   churn_pipeline.pkl
    │   results.md
```

### Tips

1. Build the Pipeline first with a plain `LogisticRegression` and default hyperparameters. Get train → test → predict working end-to-end before you tune anything.
2. Class imbalance is not your enemy — your default metric (accuracy) is. Replace it on day one and anchor your argument on the ~73% majority-class baseline.
3. `TotalCharges` loads as `object` because of empty strings. Use `pd.to_numeric(..., errors='coerce')` and handle the `NaN`s with the `SimpleImputer` inside your Pipeline.
4. If your CV scores look like 0.99, you have leakage. Common culprits: a feature built from the target, or a scaler fitted on the full dataset.
5. Pick the decision threshold on validation / out-of-fold data, never on the test set. Tuning the threshold on test is the most common way to inflate results and is leakage.
6. Keep all feature engineering inside the Pipeline so `predict.py` reproduces it exactly. A transformer that only exists in your notebook will crash inference.
7. The `README.md` is graded. A clean README often raises the score more than an extra hyperparameter sweep.
8. Use Git from the first commit, not the last. Logical, incremental commits make your work easy to follow.
9. Don't rely on a single metric. Read the confusion matrix and the precision-recall curve, and say what each told you.
10. If you finish early, plot a learning curve to show whether more data would help.

### Resources

- IBM Telco Customer Churn dataset (Kaggle, login required): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Same dataset, direct download (no account): https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- scikit-learn `Pipeline` + `ColumnTransformer`: https://scikit-learn.org/stable/modules/compose.html
- scikit-learn model selection (`GridSearchCV`, `cross_val_predict`): https://scikit-learn.org/stable/modules/grid_search.html
- scikit-learn metrics (precision-recall curve, threshold): https://scikit-learn.org/stable/modules/model_evaluation.html
- Model persistence with `joblib`: https://scikit-learn.org/stable/model_persistence.html
- Chip Huyen, *Designing Machine-Learning Systems*, Chapter 6 — choosing the right metric.
- Imbalanced-learn (optional, for SMOTE): https://imbalanced-learn.org/stable/
