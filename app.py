import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Customer Churn Analysis and Prediction", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Analysis and Prediction")
st.caption("Streamlit interface for the original Python program. The dataset, charts, model logic, and prediction rules are kept consistent with the supplied code.")

DATA_FILE = Path(__file__).parent / "bank_customer_data-1.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load bank_customer_data-1.csv: {e}")
    st.info("Keep bank_customer_data-1.csv in the same folder as app.py.")
    st.stop()

required = {"Gender", "IsActiveMember", "Complain", "Balance", "Rating", "CustomerId"}
missing_cols = required - set(df.columns)
if missing_cols:
    st.error("Dataset is missing required columns: " + ", ".join(sorted(missing_cols)))
    st.stop()

# Keep the same data preparation and machine-learning logic as the supplied program.
df_model = df.copy()
df_model["Gender"] = df_model["Gender"].map({"Female": 0, "Male": 1})
df_model["Churn"] = (df_model["Rating"] <= 5).astype(int)
X = df_model[["Rating"]]
y = df_model["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

page = st.sidebar.radio("Navigation", ["Dashboard", "Data Analysis", "Predict Customer", "Model Details"])
st.sidebar.markdown("---")
st.sidebar.caption("Customer Churn Analysis and Prediction System")

if page == "Dashboard":
    st.subheader("Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Active Members", f"{int((df['IsActiveMember'] == 1).sum()):,}")
    c3.metric("Customer Complaints", f"{int((df['Complain'] == 1).sum()):,}")
    c4.metric("Model Accuracy", f"{accuracy:.2f}%")
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("#### Original program charts")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    active = df["IsActiveMember"].value_counts().reindex([1, 0], fill_value=0)
    ax[0].bar(["Yes", "No"], active.values, color=["green", "red"])
    ax[0].set_title("Active Members")
    ax[0].set_ylabel("Customers")
    complaint = df["Complain"].value_counts().reindex([1, 0], fill_value=0)
    ax[1].bar(["Yes", "No"], complaint.values, color=["orange", "blue"])
    ax[1].set_title("Customer Complaints")
    ax[1].set_ylabel("Customers")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    gender = df_model["Gender"].value_counts().sort_index()
    ax[0].pie(gender.values, labels=["Female", "Male"], autopct="%1.1f%%")
    ax[0].set_title("Gender Distribution")
    ax[1].hist(df["Balance"], bins=10, color="purple", edgecolor="black")
    ax[1].set_title("Balance Distribution")
    ax[1].set_xlabel("Balance")
    ax[1].set_ylabel("Customers")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Data Analysis":
    st.subheader("Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["First 5 Records", "Dataset Information", "Missing Values", "Statistical Summary"])
    with tab1:
        st.dataframe(df.head(), use_container_width=True)
    with tab2:
        st.write("Dataset Shape (rows, columns):", df.shape)
        st.write("Column names:")
        st.code(", ".join(df.columns))
    with tab3:
        st.dataframe(df.isnull().sum().rename("Missing Values"), use_container_width=True)
    with tab4:
        st.dataframe(df.describe(), use_container_width=True)
    st.markdown("#### Original program charts")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    active = df["IsActiveMember"].value_counts().reindex([1, 0], fill_value=0)
    ax[0].bar(["Yes", "No"], active.values, color=["green", "red"])
    ax[0].set_title("Active Members")
    ax[0].set_ylabel("Customers")
    complaint = df["Complain"].value_counts().reindex([1, 0], fill_value=0)
    ax[1].bar(["Yes", "No"], complaint.values, color=["orange", "blue"])
    ax[1].set_title("Customer Complaints")
    ax[1].set_ylabel("Customers")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    gender = df_model["Gender"].value_counts().sort_index()
    ax[0].pie(gender.values, labels=["Female", "Male"], autopct="%1.1f%%")
    ax[0].set_title("Gender Distribution")
    ax[1].hist(df["Balance"], bins=10, color="purple", edgecolor="black")
    ax[1].set_title("Balance Distribution")
    ax[1].set_xlabel("Balance")
    ax[1].set_ylabel("Customers")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Predict Customer":
    st.subheader("Predict Customer Churn")
    st.write("Enter the same inputs as the original terminal program.")
    with st.form("prediction_form"):
        customer_id = st.number_input("Enter Customer ID", min_value=0, step=1, value=1001)
        balance = st.number_input("Enter Account Balance (₹)", min_value=0.0, step=100.0, value=10000.0)
        rating = st.number_input("Enter Customer Rating (1–10)", min_value=1.0, max_value=10.0, step=1.0, value=7.0)
        submitted = st.form_submit_button("Predict Churn", type="primary")
    if submitted:
        new_customer = pd.DataFrame({"Rating": [rating]})
        prediction = model.predict(new_customer)[0]
        probability = model.predict_proba(new_customer)[0]
        stay_percentage = probability[0] * 100
        churn_percentage = probability[1] * 100
        st.markdown("### Customer Details")
        a, b, c = st.columns(3)
        a.metric("Customer ID", str(int(customer_id)))
        b.metric("Balance", f"₹{balance:,.2f}")
        c.metric("Rating", f"{rating:g}/10")
        a, b = st.columns(2)
        a.metric("Stay Probability", f"{stay_percentage:.2f}%")
        b.metric("Churn Probability", f"{churn_percentage:.2f}%")
        if prediction == 1:
            st.error("FINAL PREDICTION: CUSTOMER IS LIKELY TO CHURN.")
        else:
            st.success("FINAL PREDICTION: CUSTOMER IS LIKELY TO STAY.")
        st.caption("As in the original code, the model uses Rating only. Customer ID and Balance are displayed but are not model features.")

else:
    st.subheader("Model Details")
    st.metric("Model Accuracy", f"{accuracy:.2f}%")
    st.write("Algorithm: Logistic Regression")
    st.write("Train/test split: 80% training and 20% testing; random_state=42")
    st.write("Target used by the original program: Churn = 1 when Rating ≤ 5; otherwise 0.")
    st.write("Input feature used by the original model: Rating only.")
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]), use_container_width=True)
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    st.warning("The original code creates the Churn target from Rating itself. This interface preserves that logic; the displayed model accuracy should not be interpreted as evidence of real-world churn prediction performance.")

st.markdown("---")
st.caption("Customer Churn Analysis and Prediction System | Streamlit interface added separately; original terminal program is not overwritten.")
