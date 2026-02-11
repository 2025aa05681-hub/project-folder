import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.title("Heart Disease Prediction - ML Models")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Dataset must contain 'target' column")
    else:
        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model_name = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
        )

        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        st.subheader("Evaluation Metrics")

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("AUC:", roc_auc_score(y_test, y_prob))
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred))
        st.write("MCC:", matthews_corrcoef(y_test, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))
