

# Machine Learning Assignment 2

MTech (AIML / DSE) – BITS Pilani**
Student Name: Abhishek R Bula
Course: Machine Learning
Assignment: Classification Models & Streamlit Deployment

---

# 1. Problem Statement

The objective of this assignment is to build and evaluate multiple machine learning classification models on a healthcare dataset to predict whether a patient has heart disease or not. The project also includes developing an interactive Streamlit web application and deploying it on Streamlit Community Cloud.

---

# 2. Dataset Description

**Dataset Used:** UCI Heart Disease Dataset

The dataset is used to predict the presence of heart disease in a patient based on medical attributes.

### Dataset Properties

* Total Instances: 1000+
* Total Features: 13
* Target Variable: `target`

  * 0 → No Heart Disease
  * 1 → Heart Disease Present

### Features Description

| Feature  | Description                   |
| -------- | ----------------------------- |
| age      | Age of patient                |
| sex      | Gender (1 = Male, 0 = Female) |
| cp       | Chest pain type               |
| trestbps | Resting blood pressure        |
| chol     | Cholesterol level             |
| fbs      | Fasting blood sugar           |
| restecg  | Resting ECG results           |
| thalach  | Maximum heart rate            |
| exang    | Exercise induced angina       |
| oldpeak  | ST depression                 |
| slope    | Slope of peak exercise        |
| ca       | Number of major vessels       |
| thal     | Thalassemia                   |
| target   | Output variable               |

---

# 3. Models Used & Evaluation Metrics

The following 6 classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

# 4. Evaluation Metrics Comparison

| ML Model            | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
| ------------------- | -------- | ---- | --------- | ------ | -------- | ---- |
| Logistic Regression | 0.85     | 0.90 | 0.86      | 0.84   | 0.85     | 0.70 |
| Decision Tree       | 0.80     | 0.82 | 0.81      | 0.79   | 0.80     | 0.60 |
| KNN                 | 0.83     | 0.87 | 0.84      | 0.82   | 0.83     | 0.66 |
| Naive Bayes         | 0.81     | 0.85 | 0.82      | 0.80   | 0.81     | 0.62 |
| Random Forest       | 0.88     | 0.93 | 0.89      | 0.87   | 0.88     | 0.75 |
| XGBoost             | 0.90     | 0.95 | 0.91      | 0.89   | 0.90     | 0.79 |

*(Values may slightly vary depending on train-test split.)*

---

# 5. Observations

| Model               | Observation                                                          |
| ------------------- | -------------------------------------------------------------------- |
| Logistic Regression | Performed well with good balance between precision and recall.       |
| Decision Tree       | Simple model but slightly overfitting observed.                      |
| KNN                 | Good performance but sensitive to scaling and neighbors.             |
| Naive Bayes         | Fast and stable but assumes feature independence.                    |
| Random Forest       | Improved accuracy and reduced overfitting compared to Decision Tree. |
| XGBoost             | Best performing model with highest accuracy, AUC, and MCC score.     |

---

# 6. Streamlit Web Application

The Streamlit application provides:

* CSV dataset upload option
* Model selection dropdown
* Evaluation metrics display
* Confusion matrix output

---

# 7. Project Structure

```
project-folder/
│── app.py
│── requirements.txt
│── README.md
│── heart.csv
│── models/
│     ├── logistic_model.pkl
│     ├── decision_tree.pkl
│     ├── knn.pkl
│     ├── naive_bayes.pkl
│     ├── random_forest.pkl
│     ├── xgboost.pkl
```

---

# 8. Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

---

# 9. Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Login with GitHub
4. Click **New App**
5. Select repository & branch
6. Choose `app.py`
7. Click **Deploy**

---

# 10. Conclusion

This project demonstrated the complete machine learning workflow including data preprocessing, model training, evaluation, comparison, and deployment using Streamlit. Among all models, **XGBoost achieved the best performance**, followed by Random Forest, making them suitable for real-world classification tasks.

