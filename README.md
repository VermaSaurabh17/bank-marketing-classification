# Bank Marketing Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to predict whether a bank customer will subscribe to a term deposit based on customer information and previous marketing campaign interactions.

This is a binary classification problem:

- **0:** No subscription
- **1:** Yes subscription

Five machine learning classification algorithms are implemented and evaluated using the same dataset and training/testing split.

The project also includes an interactive Streamlit application that allows users to upload test data, select a trained model, and view its evaluation results.

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

### Dataset Details

| Attribute           | Description           |
| ------------------- | --------------------- |
| Total records       | 45,211                |
| Original columns    | 17                    |
| Input features used | 15                    |
| Target variable     | y                     |
| Problem type        | Binary classification |
| Training data       | 80%                   |
| Testing data        | 20%                   |

The target variable indicates whether a customer subscribed to a term deposit.

### Data Preparation

The following preprocessing steps were performed:

- The `duration` feature was excluded because call duration is only known after the marketing call. It would not be available when predicting subscription before the call.
- The target variable was mapped as No = 0 and Yes = 1.
- Numerical features were standardized using `StandardScaler`.
- Categorical features were encoded using `OneHotEncoder`.
- The dataset was divided into 80% training and 20% testing data.
- Stratified sampling was used to preserve the target class distribution.
- Random state was set to 42 for reproducibility.
- Preprocessing was fitted only on the training data and then applied to the test data.

The final dataset contains 15 input features: 6 numerical features and 9 categorical features.

## 3. GitHub Repository Link

**Repository:**

https://github.com/VermaSaurabh17/bank-marketing-classification

The repository contains the Python source code, saved trained models, test dataset, dependencies, and project documentation.

## 4. Models Used and Performance Comparison

All five classifiers were trained and evaluated using the same dataset and test split.

### Evaluation Metrics

The following six evaluation metrics were calculated for each model:

1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model Name          | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ---------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression    |   0.8933 | 0.7717 |    0.6632 | 0.1786 |   0.2815 | 0.3065 |
| Decision Tree          |   0.8921 | 0.7144 |    0.6281 | 0.1900 |   0.2917 | 0.3046 |
| KNN                    |   0.8874 | 0.7027 |    0.5529 | 0.1975 |   0.2911 | 0.2833 |
| Naive Bayes            |   0.8452 | 0.7514 |    0.3708 | 0.4641 |   0.4123 | 0.3271 |
| Support Vector Machine |   0.8932 | 0.7682 |    0.6513 | 0.1871 |   0.2907 | 0.3101 |

### Observations on Model Performance

#### Logistic Regression

- Achieved the highest accuracy of 0.8933.
- Achieved the highest AUC of 0.7717.
- Achieved the highest precision of 0.6632.
- However, recall was relatively low at 0.1786.
- The model performs well in identifying negative cases but misses a significant proportion of actual subscribers.

#### Decision Tree

- Achieved an accuracy of 0.8921.
- Its F1 score was 0.2917.
- It achieved a precision of 0.6281 and recall of 0.1900.
- It performed reasonably well but did not outperform the other models on the main evaluation metrics.

#### K-Nearest Neighbors (KNN)

- Achieved an accuracy of 0.8874.
- Its AUC was 0.7027.
- Its F1 score was 0.2911.
- It showed comparatively lower overall performance among the five models.

#### Naive Bayes

- Achieved the highest recall of 0.4641.
- Achieved the highest F1 score of 0.4123.
- Achieved the highest MCC of 0.3271.
- It identified more actual subscribers than the other models.
- However, it also generated more false positives, resulting in lower precision of 0.3708.

#### Support Vector Machine (SVM)

- Achieved an accuracy of 0.8932.
- Its AUC was 0.7682.
- Its MCC was 0.3101.
- Its overall performance was comparable to Logistic Regression.
- However, its recall remained relatively low at 0.1871.

### Overall Winner for the Dataset

**Naive Bayes is the strongest baseline model based on F1 Score, Recall, and Matthews Correlation Coefficient.**

The dataset is imbalanced, with approximately:

- 88.3% negative records (No subscription).
- 11.7% positive records (Yes subscription).

Therefore, accuracy alone is not sufficient to identify the best classifier.

Naive Bayes achieved a better balance between precision and recall for identifying customers who subscribed to a term deposit. It also achieved the highest F1 score and MCC among the five models.

Logistic Regression performed best in terms of accuracy, AUC, and precision.

The final model selection would depend on the business objective and the relative cost of false positives and false negatives.

## 5. Streamlit Application

The project includes an interactive Streamlit application for evaluating the five trained classification models.

### Application Features

- Upload a test dataset in CSV format.
- Validate the uploaded dataset and required columns.
- Select one of the five trained classification models.
- Calculate and display all six evaluation metrics.
- Display the confusion matrix.
- Display the classification report.

The application loads previously trained models and evaluates the uploaded test dataset without retraining.

### Run the Application Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the application in a browser:

```text
http://localhost:8501
```

### Test Dataset

The application uses `test_data.csv`, containing 9,043 test records with 15 input features and the target variable.

The application does not train or modify models using the uploaded test data.

## 6. Project Structure

```text
bank-marketing-classification/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── results/
│   └── model_comparison.csv
│
├── inspect_data.py
├── prepare_data.py
├── train_models.py
└── evaluate_models.py
```

The original dataset is maintained locally and is not included in the public GitHub repository.

## 7. Technologies Used

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## 8. Conclusion

This project demonstrates an end-to-end machine learning classification workflow, including:

- Dataset exploration and preparation.
- Numerical and categorical feature preprocessing.
- Training five classification algorithms.
- Evaluation using six performance metrics.
- Comparison of model performance.
- Saving trained machine learning pipelines.
- Building an interactive Streamlit application.

The results highlight the importance of selecting evaluation metrics based on the classification problem rather than relying only on accuracy.

Naive Bayes demonstrated the strongest baseline performance for identifying positive subscription cases, while Logistic Regression achieved the highest accuracy, AUC, and precision.
