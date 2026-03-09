import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):

    df = df.drop("customerID", axis=1)

    # Convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def handle_missing_values(df):

    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def feature_engineering(df):

    # Feature 1: Average monthly charge
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Feature 2: Support risk indicator
    df["support_risk"] = (
        (df["TechSupport"] == "No").astype(int) +
        (df["OnlineSecurity"] == "No").astype(int)
    )

    return df


def encode_categorical(df):

    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def prepare_target(df):

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def full_preprocessing_pipeline(data_path):

    df = load_data(data_path)

    df = clean_data(df)

    df = handle_missing_values(df)

    df = feature_engineering(df)

    df = encode_categorical(df)

    X, y = prepare_target(df)

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names
    }