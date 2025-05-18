import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set up Streamlit page
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file for churn prediction", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    # Drop 'customerID' if present
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Label Encoding for categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Feature and Target split
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Prediction
    y_pred = model.predict(X_test_scaled)

    # Results
    st.subheader("Model Evaluation")
    st.write("*Accuracy:*", accuracy_score(y_test, y_pred))
    st.write("*Confusion Matrix:*")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("*Classification Report:*")
    st.text(classification_report(y_test, y_pred))

    # Feature Importance
    st.subheader("Top 10 Feature Importances")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(10).sort_values()

    fig, ax = plt.subplots()
    top_features.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title('Top 10 Important Features for Customer Churn')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)
