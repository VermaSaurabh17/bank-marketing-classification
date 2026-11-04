import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Define application paths and available models.
MODEL_DIR = Path("model")

MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# Define the expected input features and target.
REQUIRED_COLUMNS = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "campaign", "pdays", "previous", "poutcome"
]

TARGET_COLUMN = "y"

# Configure the Streamlit page.
st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Classification")
st.write(
    "Evaluate five machine learning classifiers "
    "using the Bank Marketing test dataset."
)

st.markdown("---")

# Allow the user to upload the test dataset.
st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

is_valid = False

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)

        # Validate the uploaded dataset structure.
        missing_columns = [
            col for col in REQUIRED_COLUMNS + [TARGET_COLUMN]
            if col not in test_data.columns
        ]

        if test_data.empty:
            st.error("The uploaded CSV contains no records.")

        elif missing_columns:
            st.error(f"Missing required columns: {missing_columns}")

        elif test_data[REQUIRED_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
            st.error("The dataset contains missing values in required columns.")

        elif not set(test_data[TARGET_COLUMN].unique()).issubset({0, 1}):
            st.error("Target column 'y' must contain only 0 and 1.")

        else:
            is_valid = True
            st.success("Test dataset uploaded and validated successfully.")

            st.write("Dataset shape:", test_data.shape)
            st.dataframe(test_data.head())

    except Exception as error:
        st.error(f"Unable to read the uploaded CSV: {error}")

# Display model selection and evaluation.
if uploaded_file is not None and is_valid:

    st.markdown("---")
    st.subheader("Model Selection")

    selected_model = st.selectbox(
        "Choose a classification model",
        list(MODELS.keys())
    )

    if st.button("Evaluate Model", type="primary"):

        model_path = MODEL_DIR / MODELS[selected_model]

        try:
            # Load the saved pipeline and generate predictions.
            model = joblib.load(model_path)

            X_test = test_data[REQUIRED_COLUMNS]
            y_test = test_data[TARGET_COLUMN]

            y_pred = model.predict(X_test)

            # Calculate AUC using the model's probability or decision scores.
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)

            auc = roc_auc_score(y_test, y_score)

            # Calculate all six evaluation metrics.
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": auc,
                "Precision": precision_score(
                    y_test, y_pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, y_pred, zero_division=0
                ),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }

            st.success(f"{selected_model} evaluation completed.")

            # Display evaluation metrics.
            st.subheader("Evaluation Metrics")

            metric_columns = st.columns(3)

            for index, (metric_name, metric_value) in enumerate(metrics.items()):
                metric_columns[index % 3].metric(
                    metric_name,
                    f"{metric_value:.4f}"
                )

            st.markdown("---")

            # Display confusion matrix and classification report.
            st.subheader("Model Performance Details")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.write("**Confusion Matrix**")

                cm = confusion_matrix(y_test, y_pred)

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual No", "Actual Yes"],
                    columns=["Predicted No", "Predicted Yes"]
                )

                st.dataframe(cm_df)

            with result_col2:
                st.write("**Classification Report**")

                report = classification_report(
                    y_test,
                    y_pred,
                    target_names=["No", "Yes"],
                    zero_division=0
                )

                st.text(report)

        except FileNotFoundError:
            st.error(f"Model file not found: {model_path}")

        except Exception as error:
            st.error(f"Model evaluation failed: {error}")
