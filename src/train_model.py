import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import full_preprocessing_pipeline


DATA_PATH = r"D:\Assignment\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = r"D:\Assignment\models"


# -----------------------------
# Save Confusion Matrix Image
# -----------------------------
def save_confusion_matrix(cm, model_name):

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn","Churn"],
        yticklabels=["No Churn","Churn"]
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    path = os.path.join(
        MODEL_DIR,
        f"{model_name.replace(' ','_')}_confusion_matrix.png"
    )

    plt.savefig(path)
    plt.close()


# -----------------------------
# Save Metrics Chart
# -----------------------------
def save_metrics_chart(acc, prec, rec, f1, model_name):

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

    plt.figure(figsize=(6,5))

    plt.bar(metrics.keys(), metrics.values())

    plt.ylim(0,1)

    for i,v in enumerate(metrics.values()):
        plt.text(i, v+0.02, f"{v:.2f}", ha="center")

    plt.title(f"{model_name} Metrics")

    path = os.path.join(
        MODEL_DIR,
        f"{model_name.replace(' ','_')}_metrics.png"
    )

    plt.savefig(path)
    plt.close()


# -----------------------------
# Save Classification Report
# -----------------------------
def save_classification_report(report, model_name):

    path = os.path.join(
        MODEL_DIR,
        f"{model_name.replace(' ','_')}_classification_report.txt"
    )

    with open(path, "w") as f:
        f.write(report)


# -----------------------------
# Training Function
# -----------------------------
def train_models(data_path):

    data = full_preprocessing_pipeline(data_path)

    X_train = data["X_train"]
    X_test = data["X_test"]

    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    scaler = data["scaler"]
    feature_names = data["feature_names"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
    }

    results = {}

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        # Train
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        cm = confusion_matrix(y_test, predictions)

        report = classification_report(y_test, predictions)

        # Console Output
        print("\nEvaluation Metrics")
        print("-------------------")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nConfusion Matrix")
        print(cm)

        print("\nClassification Report")
        print(report)

        # Save outputs
        save_confusion_matrix(cm, name)
        save_metrics_chart(acc, prec, rec, f1, name)
        save_classification_report(report, name)

        results[name] = {
            "model": model,
            "f1": f1
        }

    # -----------------------------
    # Select Best Model
    # -----------------------------
    best_model_name = max(results, key=lambda x: results[x]["f1"])
    best_model = results[best_model_name]["model"]

    print("\nBest Model Selected:", best_model_name)

    # -----------------------------
    # Save Model + Scaler + Features
    # -----------------------------
    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "churn_model.pkl")
    )

    joblib.dump(
        scaler,
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    joblib.dump(
        feature_names,
        os.path.join(MODEL_DIR, "features.pkl")
    )

    print("\nModel saved successfully!")


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":

    train_models(DATA_PATH)