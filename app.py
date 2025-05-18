

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc
)

# Set up Streamlit page
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("Customer Churn Prediction App")

# Generate or simulate a dataset
@st.cache_data
def load_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.round(np.random.uniform(20.0, 120.0, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(100.0, 8000.0, n_samples), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    return pd.DataFrame(data)

df = load_data()

# Preprocessing
df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'])
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_log_pred = log_model.predict(X_test_scaled)
y_log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
y_lin_pred = lin_model.predict(X_test_scaled)

# --- Streamlit Visualizations ---
st.subheader("Confusion Matrix - Logistic Regression")
cm = confusion_matrix(y_test, y_log_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("ROC Curve - Logistic Regression")
fpr, tpr, _ = roc_curve(y_test, y_log_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
st.pyplot(fig2)

st.subheader("Histogram of Predicted Churn Scores - Linear Regression")
fig3, ax3 = plt.subplots()
ax3.hist(y_lin_pred, bins=30, color='skyblue', edgecolor='black')
ax3.set_xlabel("Predicted Churn Score")
ax3.set_ylabel("Frequency")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc
)

# Set up Streamlit page
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("Customer Churn Prediction App")

# Generate or simulate a dataset
@st.cache_data
def load_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.round(np.random.uniform(20.0, 120.0, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(100.0, 8000.0, n_samples), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    return pd.DataFrame(data)

df = load_data()

# Preprocessing
df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'])
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_log_pred = log_model.predict(X_test_scaled)
y_log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
y_lin_pred = lin_model.predict(X_test_scaled)

# --- Streamlit Visualizations ---
st.subheader("Confusion Matrix - Logistic Regression")
cm = confusion_matrix(y_test, y_log_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("ROC Curve - Logistic Regression")
fpr, tpr, _ = roc_curve(y_test, y_log_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
st.pyplot(fig2)

st.subheader("Histogram of Predicted Churn Scores - Linear Regression")
fig3, ax3 = plt.subplots()
ax3.hist(y_lin_pred, bins=30, color='skyblue', edgecolor='black')
ax3.set_xlabel("Predicted Churn Score")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc
)

# Set up Streamlit page
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("Customer Churn Prediction App")

# Generate or simulate a dataset
@st.cache_data
def load_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.round(np.random.uniform(20.0, 120.0, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(100.0, 8000.0, n_samples), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    return pd.DataFrame(data)

df = load_data()

# Preprocessing
df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'])
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_log_pred = log_model.predict(X_test_scaled)
y_log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
y_lin_pred = lin_model.predict(X_test_scaled)

# --- Streamlit Visualizations ---
st.subheader("Confusion Matrix - Logistic Regression")
cm = confusion_matrix(y_test, y_log_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("ROC Curve - Logistic Regression")
fpr, tpr, _ = roc_curve(y_test, y_log_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
st.pyplot(fig2)

st.subheader("Histogram of Predicted Churn Scores - Linear Regression")
fig3, ax3 = plt.subplots()
ax3.hist(y_lin_pred, bins=30, color='skyblue', edgecolor='black')
ax3.set_xlabel("Predicted Churn Score")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc
)

# Set up Streamlit page
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("Customer Churn Prediction App")

# Generate or simulate a dataset
@st.cache_data
def load_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.round(np.random.uniform(20.0, 120.0, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(100.0, 8000.0, n_samples), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    return pd.DataFrame(data)

df = load_data()

# Preprocessing
df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'])
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_log_pred = log_model.predict(X_test_scaled)
y_log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
y_lin_pred = lin_model.predict(X_test_scaled)

# --- Streamlit Visualizations ---
st.subheader("Confusion Matrix - Logistic Regression")
cm = confusion_matrix(y_test, y_log_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("ROC Curve - Logistic Regression")
fpr, tpr, _ = roc_curve(y_test, y_log_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
st.pyplot(fig2)

st.subheader("Histogram of Predicted Churn Scores - Linear Regression")
fig3, ax3 = plt.subplots()
ax3.hist(y_lin_pred, bins=30, color='skyblue', edgecolor='black')
ax3.set_xlabel("Predicted Churn Score")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc
)

# Set up Streamlit page
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("Customer Churn Prediction App")

# Generate or simulate a dataset
@st.cache_data
def load_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.round(np.random.uniform(20.0, 120.0, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(100.0, 8000.0, n_samples), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    return pd.DataFrame(data)

df = load_data()

# Preprocessing
df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'])
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_log_pred = log_model.predict(X_test_scaled)
y_log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
y_lin_pred = lin_model.predict(X_test_scaled)

# --- Streamlit Visualizations ---
st.subheader("Confusion Matrix - Logistic Regression")
cm = confusion_matrix(y_test, y_log_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("ROC Curve - Logistic Regression")
fpr, tpr, _ = roc_curve(y_test, y_log_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
st.pyplot(fig2)

st.subheader("Histogram of Predicted Churn Scores - Linear Regression")
fig3, ax3 = plt.subplots()
ax3.hist(y_lin_pred, bins=30, color='skyblue', edgecolor='black')
ax3.set_xlabel("Predicted Churn Score")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

# The rest of the text is a comment and not Python code.
#---

#Step 3: Run the Streamlit App

#Navigate to the directory where your script is saved and run:

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc
)

# Set up Streamlit page
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("Customer Churn Prediction App")

# Generate or simulate a dataset
@st.cache_data
def load_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.round(np.random.uniform(20.0, 120.0, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(100.0, 8000.0, n_samples), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    return pd.DataFrame(data)

df = load_data()

# Preprocessing
df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'])
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_log_pred = log_model.predict(X_test_scaled)
y_log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
y_lin_pred = lin_model.predict(X_test_scaled)

# --- Streamlit Visualizations ---
st.subheader("Confusion Matrix - Logistic Regression")
cm = confusion_matrix(y_test, y_log_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("ROC Curve - Logistic Regression")
fpr, tpr, _ = roc_curve(y_test, y_log_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()
st.pyplot(fig2)

st.subheader("Histogram of Predicted Churn Scores - Linear Regression")
fig3, ax3 = plt.subplots()
ax3.hist(y_lin_pred, bins=30, color='skyblue', edgecolor='black')
ax3.set_xlabel("Predicted Churn Score")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

# The rest of the text is a comment and not Python code.
#---

#Step 3: Run the Streamlit App

#Navigate to the directory where your script is saved and run:

#streamlit run streamlit_app.py

# Navigate to the directory where your script is saved and run:

# streamlit run streamlit_app.py
