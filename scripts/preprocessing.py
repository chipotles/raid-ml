import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TotalChargesConverter(BaseEstimator, TransformerMixin):
    """Конвертирует TotalCharges из object в float (пустые строки → NaN)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Создаёт 4 новых признака:
      - tenure_bucket    : группы по сроку клиента (0-12, 13-24, 25-48, 49+)
      - charges_per_month: TotalCharges / tenure — нагрузка без влияния срока
      - n_services       : число активных услуг (не 'No' / 'No internet service')
      - is_new_customer  : 1 если tenure <= 12 месяцев
    """

    SERVICE_COLS = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]
    INACTIVE = {'No', 'No internet service', 'No phone service'}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['tenure_bucket'] = pd.cut(
            X['tenure'],
            bins=[0, 12, 24, 48, np.inf],
            labels=['0-12', '13-24', '25-48', '49+'],
            include_lowest=True,
        ).astype(str)

        X['charges_per_month'] = X['TotalCharges'] / X['tenure'].clip(lower=1)

        svc_cols = [c for c in self.SERVICE_COLS if c in X.columns]
        X['n_services'] = X[svc_cols].apply(
            lambda row: sum(str(v) not in self.INACTIVE for v in row), axis=1
        )

        X['is_new_customer'] = (X['tenure'] <= 12).astype(int)

        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Удаляет колонки-идентификаторы перед подачей в модель."""

    def __init__(self, columns=None):
        self.columns = columns or ['customerID']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=[c for c in self.columns if c in X.columns])


def get_feature_columns(df):
    """Возвращает (num_cols, cat_cols) для ColumnTransformer."""
    num_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'charges_per_month', 'n_services', 'is_new_customer',
    ]
    cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod',
        'tenure_bucket',
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    return num_cols, cat_cols


if __name__ == '__main__':
    test_df = pd.DataFrame({
        'customerID':       ['001', '002', '003'],
        'tenure':           [1, 24, 72],
        'MonthlyCharges':   [29.85, 56.95, 89.10],
        'TotalCharges':     ['29.85', ' ', '6401.40'],
        'PhoneService':     ['Yes', 'No', 'Yes'],
        'MultipleLines':    ['No phone service', 'No', 'Yes'],
        'InternetService':  ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity':   ['No', 'Yes', 'No internet service'],
        'OnlineBackup':     ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport':      ['No', 'No', 'No internet service'],
        'StreamingTV':      ['No', 'Yes', 'No internet service'],
        'StreamingMovies':  ['No', 'Yes', 'No internet service'],
        'Contract':         ['Month-to-month', 'One year', 'Two year'],
        'Partner':          ['Yes', 'No', 'Yes'],
        'Dependents':       ['No', 'Yes', 'No'],
        'gender':           ['Female', 'Male', 'Female'],
        'SeniorCitizen':    [0, 0, 1],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod':    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
    })

    out = ColumnDropper().transform(
            FeatureEngineer().transform(
                TotalChargesConverter().transform(test_df)
            )
          )

    print("tenure_bucket:    ", out['tenure_bucket'].tolist())
    print("n_services:       ", out['n_services'].tolist())
    print("is_new_customer:  ", out['is_new_customer'].tolist())
    print("charges_per_month:", out['charges_per_month'].round(2).tolist())
    print("customerID удалён:", 'customerID' not in out.columns)
    print("\n✓ OK")