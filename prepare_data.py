
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Load the Bank Marketing dataset and prepare it for model training.
DATA_PATH = Path("data/bank-full.csv")


def prepare_dataset():
    # Read the semicolon-separated CSV file.
    df = pd.read_csv(DATA_PATH, sep=";")

    # Exclude call duration to avoid target leakage.
    df = df.drop(columns=["duration"])

    # Separate input features and convert the target into binary values.
    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    # Identify numerical and categorical columns for preprocessing.
    numerical_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    # Split the data while preserving the target class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Display dataset details and verify the train-test split.
    print("Dataset shape:", df.shape)
    print("Input features shape:", X.shape)
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    print("\nNumerical features:", numerical_features)
    print("Number of numerical features:", len(numerical_features))

    print("\nCategorical features:", categorical_features)
    print("Number of categorical features:", len(categorical_features))

    print("\nTraining target distribution (%):")
    print((y_train.value_counts(normalize=True) * 100).sort_index())

    print("\nTesting target distribution (%):")
    print((y_test.value_counts(normalize=True) * 100).sort_index())
    
        # Build and fit preprocessing using only the training data.
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Display the transformed feature dimensions.
    print("\nTransformed training data shape:", X_train_transformed.shape)
    print("Transformed testing data shape:", X_test_transformed.shape)


def build_preprocessor(numerical_features, categorical_features):
    # Apply scaling to numerical columns and one-hot encoding to categorical columns.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


if __name__ == "__main__":
    prepare_dataset()
