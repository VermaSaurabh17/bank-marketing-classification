
import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


# Define the dataset and model storage locations.
DATA_PATH = Path("data/bank-full.csv")
MODEL_DIR = Path("models")

RANDOM_STATE = 42


def load_data():
    # Load the dataset and exclude call duration to avoid target leakage.
    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.drop(columns=["duration"])

    # Separate input features and convert the target into binary values.
    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    # Split the data while preserving the target class proportions.
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )


def create_preprocessor(X_train):
    # Identify numerical and categorical columns.
    numerical_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # Apply scaling to numerical data and encoding to categorical data.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ],
        remainder="drop"
    )


def create_models():
    # Configure the five classifiers required for the assignment.
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform"
        ),
        "Naive Bayes": GaussianNB(),
        "SVM": LinearSVC(
            max_iter=5000,
            random_state=RANDOM_STATE
        )
    }


def train_models():
    # Load the training and testing datasets.
    X_train, X_test, y_train, y_test = load_data()

    # Create the folder for saving trained models.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = create_models()

    print("Training records:", X_train.shape[0])
    print("Testing records:", X_test.shape[0])

    print("\nStarting model training...\n")

    # Train and save each classifier with its own preprocessing pipeline.
    for model_name, classifier in models.items():
        preprocessor = create_preprocessor(X_train)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        file_name = model_name.lower().replace(" ", "_") + ".joblib"
        model_path = MODEL_DIR / file_name

        joblib.dump(pipeline, model_path)

        print(f"{model_name}:")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Saved model: {model_path}")


if __name__ == "__main__":
    train_models()
