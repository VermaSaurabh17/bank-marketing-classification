
import pandas as pd
from pathlib import Path

# Dataset location
DATA_PATH = Path("data/bank-full.csv")

def inspect_dataset():
    # Load dataset (UCI Bank Marketing uses semicolon separator)
    df = pd.read_csv(DATA_PATH, sep=";")

    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)
    print("\nFirst 5 records:")
    print(df.head())

    print("\n========== COLUMN INFORMATION ==========")
    print(df.info())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE RECORDS ==========")
    print("Duplicates:", df.duplicated().sum())

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["y"].value_counts())
    print("\nTarget percentage:")
    print(df["y"].value_counts(normalize=True) * 100)

    print("\n========== NUMERICAL FEATURES ==========")
    print(df.select_dtypes(include="number").columns.tolist())

    print("\n========== CATEGORICAL FEATURES ==========")
    print(df.select_dtypes(include="object").columns.tolist())

    print("\n========== NUMERICAL SUMMARY ==========")
    print(df.describe())

if __name__ == "__main__":
    inspect_dataset()
