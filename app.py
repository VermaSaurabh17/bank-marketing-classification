from io import BytesIO
from pathlib import Path
import hashlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_DATA_PATH = BASE_DIR / "test_data.csv"

FEATURES = [
    "age", "balance", "day", "campaign", "pdays", "previous",
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]

NUMERIC_FEATURES = [
    "age", "balance", "day", "campaign", "pdays", "previous",
]

CATEGORICAL_FEATURES = [
    feature for feature in FEATURES if feature not in NUMERIC_FEATURES
]

MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

st.set_page_config(
    page_title="DepositScope | Model Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1400px;
            padding: 1.5rem 2rem 2rem 2rem;
        }
        [data-testid="stSidebar"] {
            background-color: #F5F7FA;
            border-right: 1px solid #E2E8F0;
        }
        body, .stApp {
            font-family: Arial, sans-serif;
        }
        h1, h2, h3 {
            color: #183B56;
            font-weight: 600;
        }
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 3px solid #287D8E;
            border-radius: 6px;
            padding: 12px 14px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0;
            border-radius: 6px;
        }
        .app-subtitle {
            color: #64748B;
            font-size: 14px;
            margin-top: -10px;
        }
        .section-label {
            color: #64748B;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.7px;
            margin-bottom: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. Model loading and evaluation
# --------------------------------------------------

@st.cache_resource
def load_models():
    loaded_models = {}

    for name, filename in MODELS.items():
        model_path = MODEL_DIR / filename

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Check that train_models.py saves models into the 'model' folder."
            )

        loaded_models[name] = joblib.load(model_path)

    return loaded_models


@st.cache_data(show_spinner=False)
def evaluate_models(csv_bytes):
    data = pd.read_csv(BytesIO(csv_bytes))

    X = data[FEATURES]
    y = data["y"].astype(int)

    saved_models = load_models()
    results = {}
    predictions = {}

    for name, model in saved_models.items():
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            positive_index = list(model.classes_).index(1)
            y_score = probabilities[:, positive_index]

        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X)

            if list(model.classes_)[-1] != 1:
                y_score = -y_score

        else:
            raise AttributeError(
                f"{name} does not provide probability or decision scores for AUC."
            )

        results[name] = {
            "Accuracy": accuracy_score(y, y_pred),
            "AUC": roc_auc_score(y, y_score),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1 Score": f1_score(y, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y, y_pred),
        }

        predictions[name] = y_pred

    return data, results, predictions


# --------------------------------------------------
# 4. Validation and encoder category checks
# --------------------------------------------------

def validate_dataset(data):
    if data.empty:
        return "The uploaded CSV contains no records."

    required_columns = FEATURES + ["y"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        return "Missing required columns: " + ", ".join(missing_columns)

    if data[required_columns].isnull().any().any():
        return "The required columns contain missing values."

    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(data[column]):
            return f"Column '{column}' must contain numeric values."

        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            return f"Column '{column}' contains infinite or non-finite values."

    if not pd.api.types.is_numeric_dtype(data["y"]):
        return "Target column 'y' must contain numeric values 0 or 1."

    if not data["y"].isin([0, 1]).all():
        return "Target column 'y' must contain only 0 and 1."

    if data["y"].nunique() != 2:
        return (
            "The test dataset must contain both target classes (0 and 1) "
            "to calculate AUC."
        )

    return None


def get_encoder_categories(model):
    """Return fitted OneHotEncoder categories from a saved sklearn pipeline."""

    try:
        preprocessor = model.named_steps["preprocessor"]
    except (AttributeError, KeyError):
        return {}

    def collect_encoders(transformer, columns):
        if hasattr(transformer, "categories_"):
            return {
                feature: set(values.tolist())
                for feature, values in zip(columns, transformer.categories_)
            }

        if hasattr(transformer, "named_steps"):
            for step in transformer.named_steps.values():
                categories = collect_encoders(step, columns)

                if categories:
                    return categories

        if hasattr(transformer, "transformers_"):
            for _, nested_transformer, nested_columns in transformer.transformers_:
                if isinstance(
                    nested_columns,
                    (list, tuple, np.ndarray, pd.Index)
                ):
                    categories = collect_encoders(
                        nested_transformer,
                        list(nested_columns)
                    )

                    if categories:
                        return categories

        return {}

    if hasattr(preprocessor, "transformers_"):
        for _, transformer, columns in preprocessor.transformers_:
            if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                categories = collect_encoders(transformer, list(columns))

                if categories:
                    return categories

    return {}


def find_unseen_categories(data, models):
    if not models:
        return {}

    known_categories = get_encoder_categories(
        next(iter(models.values()))
    )

    if not known_categories:
        return {}

    unseen = {}

    for feature in CATEGORICAL_FEATURES:
        known_values = known_categories.get(feature)

        if known_values is None:
            continue

        unknown_values = (
            set(data[feature].unique().tolist()) - known_values
        )

        if unknown_values:
            unseen[feature] = sorted(
                str(value) for value in unknown_values
            )

    return unseen

# --------------------------------------------------
# 5. Sidebar: model selection only
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Model selection")

    selected_model = st.selectbox(
        "Select classifier",
        options=list(MODELS.keys())
    )

    st.divider()

    st.caption("Five pre-trained classification models")
    st.caption("Models are loaded only when evaluation is requested.")


# --------------------------------------------------
# 6. Main header and single upload location
# --------------------------------------------------

st.title("DepositScope")

st.markdown(
    '<p class="app-subtitle">'
    'Bank Marketing | Classification Evaluation Platform'
    '</p>',
    unsafe_allow_html=True,
)

st.write(
    "Evaluate and compare five machine learning classifiers "
    "for bank term deposit subscription prediction."
)

st.markdown("### Upload evaluation dataset")

upload_col, sample_col = st.columns([3, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload test dataset (CSV)",
        type=["csv"],
        help=(
            "Upload the 15 input features and target column "
            "y (0 = No, 1 = Yes)."
        ),
        key="main_csv_uploader",
    )

with sample_col:
    st.write("")
    st.write("")

    if SAMPLE_DATA_PATH.is_file():
        st.download_button(
            "Download sample CSV",
            data=SAMPLE_DATA_PATH.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Sample CSV unavailable.")

st.caption(
    "Required columns: 15 predictors and target y (0 = No, 1 = Yes). "
    "The app evaluates saved models and does not retrain them."
)

st.divider()


# --------------------------------------------------
# 7. Initial page when no file is uploaded
# --------------------------------------------------

if uploaded_file is None:
    st.subheader("Model evaluation workspace")

    st.info(
        "Upload a test CSV above to view the dataset overview "
        "and run model evaluation."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Classification Models", "5")
    col2.metric("Evaluation Metrics", "6")
    col3.metric("Input Features", "15")

    st.markdown("### Dataset information")

    st.write(
        "The application uses the UCI Bank Marketing dataset "
        "to predict whether a client will subscribe to a term deposit."
    )

    st.markdown(
        "[UCI Bank Marketing Dataset]"
        "(https://archive.ics.uci.edu/dataset/222/bank+marketing)"
    )

    st.stop()


# --------------------------------------------------
# 8. Read and validate uploaded data
# --------------------------------------------------

try:
    csv_bytes = uploaded_file.getvalue()
    uploaded_data = pd.read_csv(BytesIO(csv_bytes))

except Exception as error:
    st.error(f"Unable to read the uploaded CSV: {error}")
    st.stop()


validation_error = validate_dataset(uploaded_data)

if validation_error:
    st.error(validation_error)
    st.info(
        "Check the required columns and target format shown above, "
        "or download the sample CSV as a reference."
    )
    st.stop()


# --------------------------------------------------
# 9. Reset evaluation when uploaded file changes
# --------------------------------------------------

upload_fingerprint = hashlib.sha256(csv_bytes).hexdigest()

if st.session_state.get("upload_fingerprint") != upload_fingerprint:

    st.session_state["upload_fingerprint"] = upload_fingerprint

    st.session_state["evaluation_complete"] = False

    st.session_state.pop("evaluation_result", None)


# --------------------------------------------------
# 10. Dataset overview
# --------------------------------------------------

st.markdown(
    '<p class="section-label">01 · DATASET OVERVIEW</p>',
    unsafe_allow_html=True,
)

st.subheader("Evaluation dataset")

target = uploaded_data["y"].astype(int)

yes_count = int((target == 1).sum())
no_count = int((target == 0).sum())
total_count = len(target)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Test Records", f"{total_count:,}")
c2.metric("Input Features", len(FEATURES))
c3.metric("Subscribed (Yes)", f"{yes_count:,}")
c4.metric("Not Subscribed (No)", f"{no_count:,}")

d1, d2 = st.columns(2)

d1.metric("Yes class share", f"{yes_count / total_count:.1%}")
d2.metric("No class share", f"{no_count / total_count:.1%}")


# --------------------------------------------------
# 11. Uploaded data preview
# --------------------------------------------------

with st.expander("Preview uploaded data"):

    st.caption(
        f"Showing the first 10 records from {total_count:,} uploaded records."
    )

    st.dataframe(
        uploaded_data[FEATURES + ["y"]].head(10),
        use_container_width=True,
        height=320,
    )


# --------------------------------------------------
# 12. Dataset details
# --------------------------------------------------

with st.expander("Dataset details and source"):

    st.markdown("**Bank Marketing Dataset**")

    st.write(
        "The dataset contains direct marketing campaign information "
        "from a Portuguese banking institution. The objective is to "
        "predict whether a client will subscribe to a bank term deposit."
    )

    d1, d2, d3 = st.columns(3)

    d1.metric("Original Instances", "45,211")
    d2.metric("Original Predictors", "16")
    d3.metric("Target Classes", "2")

    details = pd.DataFrame({
        "Property": [
            "Subject Area",
            "Associated Task",
            "Feature Type",
            "Target Variable",
            "Target Classes",
        ],
        "Description": [
            "Business and Banking",
            "Binary Classification",
            "Categorical and Integer",
            "y",
            "Yes / No",
        ],
    })

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Experimental dataset**")

    st.write(
        "The experiment uses 15 predictor features and target variable y. "
        "The original dataset was divided using a stratified 80:20 training "
        "and testing split. The project's sample test CSV contains 9,043 records; "
        "your uploaded dataset currently contains "
        f"{total_count:,} records. Saved preprocessing pipelines and "
        "classifiers are used without retraining."
    )

    st.write(
        "**Preprocessing:** StandardScaler for numerical features and "
        "OneHotEncoder for categorical features."
    )

    st.markdown(
        "[View Bank Marketing Dataset – UCI Repository]"
        "(https://archive.ics.uci.edu/dataset/222/bank+marketing)"
    )


st.divider()

# --------------------------------------------------
# 13. Tabs: evaluation is explicitly triggered
# --------------------------------------------------

overview_tab, comparison_tab, evaluation_tab = st.tabs(
    ["Overview", "Model Comparison", "Model Evaluation"]
)


# --------------------------------------------------
# 14. Overview tab
# --------------------------------------------------

with overview_tab:

    st.markdown("### Dataset readiness")

    st.success(
        "The uploaded CSV passed the required schema and data validation checks."
    )

    st.write(
        "The dataset overview is available without loading model artifacts. "
        "Open **Model Comparison** and select **Run Evaluation** "
        "when you are ready."
    )

    st.caption(
        "Evaluation uses the uploaded test data. For meaningful generalization "
        "metrics, this data should be independent of the model training data."
    )


# --------------------------------------------------
# 15. Model comparison tab
# --------------------------------------------------

with comparison_tab:

    st.markdown(
        '<p class="section-label">02 · MODEL COMPARISON</p>',
        unsafe_allow_html=True,
    )

    st.subheader("Performance across five classifiers")

    ranking_metric = st.selectbox(
        "Rank models by",
        options=METRICS,
        index=4,
        help=(
            "Choose the metric used to rank and highlight "
            "the leading model."
        ),
    )

    run_clicked = st.button(
        "Run Evaluation",
        type="primary",
        use_container_width=False,
    )

    # Run evaluation only when the user clicks the button
    if run_clicked:

        try:
            with st.spinner(
                "Loading saved models and evaluating the test dataset..."
            ):

                saved_models = load_models()

                evaluation_result = evaluate_models(csv_bytes)

            st.session_state["evaluation_result"] = evaluation_result

            st.session_state["evaluation_complete"] = True

            st.session_state["unseen_categories"] = (
                find_unseen_categories(uploaded_data, saved_models)
            )

        except Exception as error:

            st.session_state["evaluation_complete"] = False

            st.error(f"Model evaluation failed: {error}")


    # Display results after evaluation
    if st.session_state.get("evaluation_complete", False):

        data, results, predictions = (
            st.session_state["evaluation_result"]
        )

        comparison_df = pd.DataFrame(results).T

        comparison_df.index.name = "Classification Model"

        comparison_display = comparison_df.sort_values(
            ranking_metric,
            ascending=False,
        ).copy()

        comparison_display.insert(
            0,
            "Rank",
            range(1, len(comparison_display) + 1),
        )

        best_model = comparison_display.index[0]

        best_score = comparison_display.iloc[0][ranking_metric]


        # Highlight the best-performing model
        def highlight_winner(row):

            if row.name == best_model:
                return [
                    "background-color: #E5F2EC; font-weight: bold"
                ] * len(row)

            return [""] * len(row)


        st.dataframe(
            comparison_display.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC": "{:.4f}",
            }).apply(highlight_winner, axis=1),
            use_container_width=True,
        )

        ranking_slug = ranking_metric.lower().replace(" ", "_")

        st.download_button(
            "Download comparison results (CSV)",
            data=comparison_display.to_csv(index=True).encode("utf-8"),
            file_name=f"model_comparison_by_{ranking_slug}.csv",
            mime="text/csv",
            key="download_comparison_csv",
        )

        st.success(
            f"Best by {ranking_metric}: "
            f"{best_model} ({best_score:.4f})"
        )

        st.info(
            "Evaluation complete. Open the **Model Evaluation** tab "
            "to inspect the selected classifier's metrics and confusion matrix."
        )

        st.caption(
            f"Models are ranked by {ranking_metric}. Consider all six metrics "
            "and the business cost of false positives and false negatives."
        )

        st.info(
            "Evaluation metrics estimate model generalization only when the "
            "uploaded test dataset is independent of the data used to train the models."
        )


        # Warn about categorical values not seen during training
        unseen_categories = st.session_state.get(
            "unseen_categories",
            {}
        )

        if unseen_categories:

            warning_lines = [
                f"**{feature}:** {', '.join(values)}"
                for feature, values in unseen_categories.items()
            ]

            st.warning(
                "The uploaded dataset contains categorical values not seen "
                "during training. The encoder ignores unseen categories, "
                "so evaluation can continue; check these values for spelling "
                "or data consistency:\n\n"
                + "\n\n".join(warning_lines)
            )

    else:

        st.info(
            "No evaluation has been run for this upload yet. "
            "Select a ranking metric and click **Run Evaluation** "
            "to load the saved models and calculate all six metrics."
        )


# --------------------------------------------------
# 16. Model evaluation tab
# --------------------------------------------------

with evaluation_tab:

    st.markdown(
        '<p class="section-label">03 · MODEL EVALUATION</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.get("evaluation_complete", False):

        st.info(
            "Run the evaluation from the **Model Comparison** tab "
            "to view the selected model's metrics, confusion matrix, "
            "and classification report."
        )

    else:

        data, results, predictions = (
            st.session_state["evaluation_result"]
        )

        st.subheader(selected_model)

        selected_metrics = results[selected_model]

        selected_predictions = predictions[selected_model]


        # Six evaluation metrics
        metric_row_1 = st.columns(3)

        for column, metric in zip(
            metric_row_1,
            ["Accuracy", "AUC", "Precision"],
        ):
            column.metric(
                metric,
                f"{selected_metrics[metric]:.4f}",
            )


        metric_row_2 = st.columns(3)

        for column, metric in zip(
            metric_row_2,
            ["Recall", "F1 Score", "MCC"],
        ):
            column.metric(
                metric,
                f"{selected_metrics[metric]:.4f}",
            )


        st.divider()

        # --------------------------------------------------
        # 17. Confusion matrix
        # --------------------------------------------------

        st.subheader("Confusion matrix")

        st.caption(
            "Rows represent actual classes and columns represent predicted "
            "classes. Percentages are calculated within each actual class."
        )

        cm = confusion_matrix(
            data["y"].astype(int),
            selected_predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        row_totals = cm.sum(axis=1, keepdims=True)

        percentages = np.divide(
            cm * 100,
            row_totals,
            out=np.zeros_like(cm, dtype=float),
            where=row_totals != 0,
        )

        annotations = np.empty((2, 2), dtype=object)

        for row in range(2):
            for column in range(2):

                annotations[row, column] = (
                    f"{cm[row, column]:,}\n"
                    f"{percentages[row, column]:.1f}%"
                )


        matrix_col, explanation_col = st.columns([1, 1.4])


        # Compact heatmap
        with matrix_col:

            fig, ax = plt.subplots(figsize=(4.2, 3.2))

            sns.heatmap(
                cm,
                annot=annotations,
                fmt="",
                cmap="Blues",
                linewidths=1.5,
                linecolor="white",
                cbar=False,
                xticklabels=["No (0)", "Yes (1)"],
                yticklabels=["No (0)", "Yes (1)"],
                annot_kws={
                    "fontsize": 10,
                    "fontweight": "semibold",
                    "color": "#183B56",
                },
                ax=ax,
            )

            ax.set_xlabel(
                "Predicted class",
                fontsize=9,
                labelpad=7,
            )

            ax.set_ylabel(
                "Actual class",
                fontsize=9,
                labelpad=7,
            )

            ax.set_title(
                selected_model,
                fontsize=10,
                pad=10,
            )

            ax.tick_params(
                axis="both",
                labelsize=9,
                length=0,
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=False,
            )

            plt.close(fig)


        # Numerical confusion matrix table
        with explanation_col:

            st.markdown("**Confusion matrix values**")

            cm_table = pd.DataFrame(
                cm,
                index=["Actual No (0)", "Actual Yes (1)"],
                columns=["Predicted No (0)", "Predicted Yes (1)"],
            )

            st.dataframe(
                cm_table,
                use_container_width=True,
            )

            st.caption(
                "The table provides exact counts for accessibility and "
                "for cases where heatmap colors are difficult to distinguish."
            )


        # --------------------------------------------------
        # 18. Prediction summary
        # --------------------------------------------------

        st.markdown("**Prediction summary**")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("True Negatives", f"{tn:,}")
        c2.metric("False Positives", f"{fp:,}")
        c3.metric("False Negatives", f"{fn:,}")
        c4.metric("True Positives", f"{tp:,}")


        st.divider()


        # --------------------------------------------------
        # 19. Classification report
        # --------------------------------------------------

        st.subheader("Classification report")

        report = classification_report(
            data["y"].astype(int),
            selected_predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).T

        st.dataframe(
            report_df.style.format({
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}",
            }),
            use_container_width=True,
        )


# --------------------------------------------------
# 20. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "DepositScope | Bank Marketing Classification | "
    "Five saved classifiers | Evaluation without retraining"
)