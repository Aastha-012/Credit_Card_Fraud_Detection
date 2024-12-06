# Credit Card Fraud Detection Project

## Overview
This project is designed to detect fraudulent credit card transactions using machine learning techniques. It explores data preprocessing, feature engineering, and model training while handling imbalanced datasets.

---

## Features
- **Data Preprocessing**: Min-Max scaling, SMOTE for class imbalance.
- **Visualization**: Histogram, correlation heatmaps, and fraud vs. non-fraud distributions.
- **Machine Learning Models**: Comparison of logistic regression, random forest, and more.
- **Deployment**: Real-time prediction using Flask API.

---

## Dataset
The dataset used is an anonymized dataset from Kaggle:
- **Link**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Jupyter Notebook**:
    Open `notebooks/credit_fraud.ipynb` to explore the code and outputs.

4. **Run the Flask API**:
    ```bash
    python src/model_api.py
    ```
    Send a POST request to `http://127.0.0.1:5000/predict` with JSON data.

---

## Results
- Best performing model: Random Forest with an F1-score of **X.XX** and AUC-ROC of **X.XX**.
- Flask API deployed for real-time fraud detection.

---

## File Structure
