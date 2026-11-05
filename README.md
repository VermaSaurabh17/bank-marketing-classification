# Bank Marketing Classification

This project uses machine learning to predict whether a customer will subscribe to a bank term deposit after a direct marketing campaign.

The work compares five classification algorithms on the UCI Bank Marketing dataset and includes a Streamlit app for evaluating the saved models against a test CSV.

## Dataset

**Source:** [UCI Machine Learning Repository – Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

| Item | Details |
|---|---:|
| Original records | 45,211 |
| Original columns | 17 |
| Input features used | 15 |
| Target | `y` |
| Training / test split | 80% / 20% |
| Test records | 9,043 |

The target is binary: `0` means the customer did not subscribe, and `1` means the customer subscribed.

### Data preparation

- Removed `duration`. It is only known after the call and would not be available when making a prediction before the call.
- Converted the target values to `0` (No) and `1` (Yes).
- Standardized the six numerical features with `StandardScaler`.
- Encoded the nine categorical features with `OneHotEncoder`.
- Used a stratified 80:20 train/test split with `random_state=42`.
- Fitted preprocessing on the training data and applied the fitted transformations to the test data.

The resulting model input contains 15 features: six numerical and nine categorical.

## Models and results

All five models use the same training and test split. The test set was used to calculate the following metrics.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8933 | 0.7717 | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree | 0.8921 | 0.7144 | 0.6281 | 0.1900 | 0.2917 | 0.3046 |
| K-Nearest Neighbors | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Gaussian Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.8932 | 0.7682 | 0.6513 | 0.1871 | 0.2907 | 0.3101 |

### Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the highest accuracy (89.33%), AUC (0.7717), and precision (0.6632). It produced relatively few false positives, but its recall was low (17.86%), meaning it missed many actual subscribers. |
| Decision Tree | Performance was close to Logistic Regression on accuracy (89.21%), with lower AUC (0.7144) and precision (0.6281), and slightly higher recall (19.00%). |
| K-Nearest Neighbors (KNN) | Recorded the lowest MCC (0.2833) among the five models. Its F1 score was 0.2911 and AUC was 0.7027 on this test split. |
| Gaussian Naive Bayes | Identified more subscribers than the other models, with the highest recall (46.41%), F1 score (0.4123), and MCC (0.3271). This came with lower precision (0.3708) and more false positives. |
| Support Vector Machine (SVM) | Achieved accuracy of 89.32%, AUC of 0.7682, and precision of 0.6513. Recall remained low at 18.71%, so it also missed many actual subscribers. |
| Overall winner for this dataset | **Gaussian Naive Bayes** ranked highest on recall, F1 score, and MCC, making it the strongest choice when identifying actual subscribers is the priority. Logistic Regression is a reasonable alternative when precision is more important. |

The target classes are imbalanced: about 88.3% of records are `No` and 11.7% are `Yes`. Accuracy alone can therefore be misleading. The preferred model depends on the business cost of missing a potential subscriber versus contacting a customer who does not subscribe.

## Streamlit application

The app is called **DepositScope**. It loads the saved model pipelines and evaluates an uploaded test CSV; it does not retrain the models.

The app provides:

- CSV upload and input validation.
- A dataset overview.
- Selection of one of the five saved classifiers.
- A comparison of all five models using six metrics.
- A confusion matrix, prediction counts, and classification report for the selected model.
- A CSV download of the model comparison results.

**Live app:** [Open DepositScope](https://2026ak05336-ml.streamlit.app/)

### Run locally

Install the packages:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in a browser.

## Repository

[GitHub – bank-marketing-classification](https://github.com/VermaSaurabh17/bank-marketing-classification)

The repository includes the application, training and evaluation scripts, saved model files, test data, dependencies, and this README.

## Project structure

```text
bank-marketing-classification/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── results/
│   └── model_comparison.csv
├── inspect_data.py
├── prepare_data.py
├── train_models.py
└── evaluate_models.py
```

The original dataset is kept locally and is not included in the public repository.

## Tools used

- Python 3.12
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- Joblib
