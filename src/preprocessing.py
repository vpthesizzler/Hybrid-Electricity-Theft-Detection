import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(sgcc_path, freq_path):

    sgcc = pd.read_csv(sgcc_path)
    freq = pd.read_csv(freq_path)

    sgcc.columns = sgcc.columns.str.strip()
    freq.columns = freq.columns.str.strip()

    sgcc = sgcc.select_dtypes(include=["number"])
    freq = freq.select_dtypes(include=["number"])

    return sgcc, freq


def prepare_dataset(sgcc, freq):

    y = sgcc["label"]

    X = pd.concat(
        [
            sgcc.drop(columns=["label"]),
            freq.drop(columns=["label"], errors="ignore")
        ],
        axis=1
    )

    return X, y


def split_and_balance(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    return X_train, X_test, y_train, y_test
