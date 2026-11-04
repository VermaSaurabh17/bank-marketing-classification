
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


# Define the dataset, trained model and result locations.
DATA_PATH = Path("data/bank-full.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

RANDOM_STATE = 42


def load_test_data():
    # Load the dataset and exclude call duration to avoid target leakage.
    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.drop(columns=["duration"])

    # Separate input features and convert the target into binary values.
    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    # Recreate the same stratified split used during model training.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_test, y_test


def save_test_data(X_test, y_test):
    # Combine test features and actual target values for Streamlit evaluation.
    test_data = X_test.copy()
    test_data["y"] = y_test

    # Save the test dataset for uploading to the Streamlit application.
    test_data.to_csv("test_data.csv", index=False)

    print("\nTest dataset saved to: test_data.csv")
    print("Test dataset shape:", test_data.shape)

def evaluate_models():
    # Load the test dataset and create the results folder.
    X_test, y_test = load_test_data()
    save_test_data(X_test, y_test)
    
    
def save_test_data(X_test, y_test):
    # Combine test features and actual target values for Streamlit evaluation.
    test_data = X_test.copy()
    test_data["y"] = y_test

    # Save the test dataset for uploading to the Streamlit application.
    test_data.to_csv("test_data.csv", index=False)

    print("\nTest dataset saved to: test_data.csv")
    print("Test dataset shape:", test_data.shape)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "SVM": "svm.joblib"
    }

    results = []

    # Load each trained model and calculate the required evaluation metrics.
    for model_name, file_name in model_files.items():
        model = joblib.load(MODEL_DIR / file_name)

        y_pred = model.predict(X_test)

        # Use predicted probabilities or decision scores for AUC calculation.
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_score),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

        results.append(metrics)

        # Display the confusion matrix for the current model.
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        print(f"\n{model_name} - Confusion Matrix")
        print(cm)

        print(f"{model_name} - Evaluation Metrics")
        for metric_name, metric_value in metrics.items():
            if metric_name != "Model":
                print(f"{metric_name}: {metric_value:.4f}")

    # Save the evaluation results for comparison and reporting.
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_DIR / "model_comparison.csv", index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.round(4).to_string(index=False))

    print("\nResults saved to:", RESULT_DIR / "model_comparison.csv")


if __name__ == "__main__":
    evaluate_models()
