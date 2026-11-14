# 🛒 Will This Visitor Buy? — Real-Time Purchase Intent Prediction

**A Machine Learning web application that predicts whether an online shopping visitor is likely to make a purchase based on their browsing session behavior.**

🌐 **Live Demo:** [Try the Purchase Intent Predictor](https://purchase-intent-prediction-j7zjqdeud49csfmhwepqt7.streamlit.app/)

## 📌 Project Overview

This project aims to predict online shoppers' purchase intentions using session-based behavioral data.

The model analyzes browsing activity, page values, visitor type, traffic source, and other session features to estimate the probability of a purchase.

The trained model is deployed as an interactive web application using Streamlit.

## 🎯 Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Online Shoppers Purchasing Intention
- **Total Records:** 12,330 sessions
- **Target Variable:** `Revenue`
- **Task:** Binary Classification

## 🧠 Machine Learning Approach

The project includes:

- Data preprocessing and feature engineering.
- Handling missing values and categorical features.
- Class imbalance handling using **SMOTENC**.
- Model evaluation and hyperparameter tuning.
- Cross-validation and threshold optimization.
- Feature importance analysis.

**Final Model:** HistGradientBoostingClassifier inside an Imbalanced-learn Pipeline.

**Selected Classification Threshold:** 0.327

## 📊 Model Performance

Evaluation results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 89.13% |
| Precision | 63.13% |
| Recall | 71.73% |
| F1-score | 67.16% |
| ROC-AUC | 93.29% |
| Average Precision | 74.66% |

### Confusion Matrix

| | Predicted: No Purchase | Predicted: Purchase |
|---|---:|---:|
| Actual: No Purchase | 1,924 | 160 |
| Actual: Purchase | 108 | 274 |

### Feature Importance

The most influential features were:
1. PageValues
2. Month
3. ExitRates
4. ProductRelated_Duration
5. Administrative_Duration

## 🖥️ Application

The Streamlit application allows users to:

- Enter visitor session information.
- Get a predicted purchase probability.
- Identify whether the visitor shows purchase intent.
- View an interactive prediction result.

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Joblib
- Streamlit
- UCI ML Repository (`ucimlrepo`)

## 🚀 Run Locally

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Navigate to the project folder:

   ```bash
   cd Purchase-Intent-Prediction
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

## 🌐 Live Application

Try the deployed application:

**[Will This Visitor Buy? — Live Demo](https://purchase-intent-prediction-j7zjqdeud49csfmhwepqt7.streamlit.app/)**

---

**Developed as a Machine Learning project using Python and Streamlit.**
