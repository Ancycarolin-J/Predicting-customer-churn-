import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred),
            'conf_matrix': confusion_matrix(y_test, y_pred)
        }

    # Evaluation
    st.subheader("Model Evaluation")
    for name, result in results.items():
        st.markdown(f"### {name}")
        st.write("*Accuracy:*", result['accuracy'])
        st.write("*Confusion Matrix:*")
        st.write(result['conf_matrix'])
        st.write("*Classification Report:*")
        st.text(result['report'])

    # Feature Importance (Random Forest)
    st.subheader("Top 10 Feature Importances (Random Forest)")
    importances = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
    top_features = importances.nlargest(10).sort_values()

    fig, ax = plt.subplots()
    top_features.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title('Top 10 Important Features')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)

    # Custom Input Prediction
    st.subheader("Make a Custom Prediction")
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    selected_model = models[selected_model_name]

    input_data = {}
    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_.tolist()
            input_data[col] = st.selectbox(f"{col}", options)
        else:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    # Encode and scale input safely
    input_df = pd.DataFrame([input_data])
    for col in label_encoders:
        if col in input_df.columns:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except Exception as e:
                st.error(f"Encoding error in column '{col}': {e}")
        else:
            st.warning(f"Column '{col}' missing in input. Skipping encoding.")

    input_scaled = scaler.transform(input_df)

    prediction = selected_model.predict(input_scaled)[0]
    prediction_proba = selected_model.predict_proba(input_scaled)[0]

    st.write("### Prediction Result:")
    st.write("*Churn:*", "Yes" if prediction == 1 else "No")
    st.write("*Probability of Churn:*", round(prediction_proba[1]*100, 2), "%")
