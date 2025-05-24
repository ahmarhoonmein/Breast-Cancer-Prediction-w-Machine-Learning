# Breast-Cancer-Prediction-w-Machine-Learning

This project aims to predict whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** using various machine learning classification models trained on the **Breast Cancer Wisconsin dataset** from Kaggle.

---

## Dataset

- **Source:** [Kaggle - Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- Contains 30 numeric features computed from digitized images of a breast mass.
- Each instance is labeled as either:
  - `M` → Malignant
  - `B` → Benign

---

## Project Workflow

1. **Data Collection**
   - Dataset downloaded using Kaggle API.
   - Loaded and inspected using `pandas`.

2. **Data Preprocessing**
   - Dropped irrelevant columns.
   - Encoded categorical labels (M → 1, B → 0).
   - Scaled features using `StandardScaler`.

3. **Model Training**
   Trained and evaluated the following classifiers:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Naive Bayes
   - Decision Tree
   - Random Forest

4. **Model Evaluation**
   - Accuracy score for each model.
   - Confusion matrices visualized using `seaborn`.

5. **Prediction on New Data**
   - Ability to input a new set of tumor features and receive a prediction from the model.

---

## Main Algorithm Used: Logistic Regression

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Jupyter Notebook

---
