import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.3rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin-bottom: .3rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #e7edf3;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📊 Customer Churn Analysis & Prediction</h1>
  <p>Analyze customer data, explore patterns, and estimate churn probability.</p>
</div>
""", unsafe_allow_html=True)

DATA_FILE = Path(__file__).parent / "bank_customer_data-1(1).csv"

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Choose a page", ["Dashboard", "Predict Customer", "Model Details"])
    st.divider()
    st.caption("Python • Pandas • Matplotlib • Scikit-learn")
    st.success("Bundled project dataset loaded automatically.")

try:
    df = pd.read_csv(DATA_FILE)
except Exception as exc:
    st.error(f"Could not load the project dataset. Check that {DATA_FILE.name} is in the same folder as app.py. Details: {exc}")
    st.stop()

required = ["CustomerId", "Gender", "Balance", "IsActiveMember", "Complain", "Rating"]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error("The dataset is missing required columns: " + ", ".join(missing))
    st.stop()

df = df.copy()
for col in ["Balance", "IsActiveMember", "Complain", "Rating", "CustomerId"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["Balance", "IsActiveMember", "Complain", "Rating", "CustomerId"])
df = df[df["Rating"].between(1, 10)]
if len(df) < 5:
    st.error("Not enough valid customer records to build the model.")
    st.stop()

# Match the target rule used in the submitted project report.
df["Churn"] = (df["Rating"] <= 5).astype(int)

X = df[["Rating"]]
y = df["Churn"]
if y.nunique() < 2:
    st.error("The dataset has only one class under the report's rating-based rule. Include ratings both <= 5 and > 5.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

def money(value):
    return f"₹{value:,.2f}"

if page == "Dashboard":
    st.header("Customer Overview")
    total = len(df)
    churn_count = int(df["Churn"].sum())
    stay_count = total - churn_count
    churn_rate = churn_count / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Likely Churn (rating rule)", f"{churn_count:,}")
    c3.metric("Likely to Stay (rating rule)", f"{stay_count:,}")
    c4.metric("Churn Rate", f"{churn_rate:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Churn vs Stay")
        counts = df["Churn"].map({0: "Likely to Stay", 1: "Likely to Churn"}).value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Number of customers")
        ax.tick_params(axis="x", rotation=10)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.subheader("Customer Rating Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["Rating"], bins=np.arange(0.5, 11.5, 1), edgecolor="black")
        ax.set_xlabel("Rating (1–10)")
        ax.set_ylabel("Number of customers")
        ax.set_xticks(range(1, 11))
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Customer Activity")
    active_counts = df["IsActiveMember"].map({1: "Active", 0: "Inactive"}).value_counts()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(active_counts.index, active_counts.values)
    ax.set_ylabel("Customers")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Customer Complaints")
    complaint_counts = df["Complain"].map({1: "Complaint", 0: "No Complaint"}).value_counts()
    st.bar_chart(complaint_counts)

    st.subheader("Account Balance Distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df["Balance"], bins=15, edgecolor="black")
    ax.set_xlabel("Balance (₹)")
    ax.set_ylabel("Customers")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("View customer records"):
        st.dataframe(df.drop(columns=["Churn"]), use_container_width=True)

elif page == "Predict Customer":
    st.header("Predict Customer Churn")
    st.write("Enter a customer ID, account balance, and rating. The current model uses Rating as its input feature, matching your report's code.")
    customer_ids = df["CustomerId"].astype("int64").tolist()
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            customer_id = st.number_input("Customer ID", min_value=1, value=int(customer_ids[0]), step=1)
        with col2:
            balance = st.number_input("Account Balance (₹)", min_value=0.0, value=0.0, step=1000.0)
        with col3:
            rating = st.slider("Customer Rating", min_value=1, max_value=10, value=7)
        submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

    if submitted:
        probs = model.predict_proba(pd.DataFrame({"Rating": [float(rating)]}))[0]
        class_positions = {int(cls): i for i, cls in enumerate(model.classes_)}
        churn_probability = float(probs[class_positions[1]]) * 100
        stay_probability = float(probs[class_positions[0]]) * 100
        prediction = int(model.predict(pd.DataFrame({"Rating": [float(rating)]}))[0])

        st.divider()
        st.subheader("Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Customer ID", str(int(customer_id)))
        r2.metric("Stay Probability", f"{stay_probability:.2f}%")
        r3.metric("Churn Probability", f"{churn_probability:.2f}%")
        st.progress(int(round(churn_probability)))
        if prediction == 1:
            st.error("Prediction: Customer is likely to churn.")
        else:
            st.success("Prediction: Customer is likely to stay.")
        st.caption(f"Entered balance: {money(balance)}")
        st.warning("This starter app follows your report's rule: Rating <= 5 is labelled as churn. Because the label is derived from Rating, the probabilities are a demonstration and not a validated real-world churn prediction.")

elif page == "Model Details":
    st.header("Model & Dataset Details")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Rows", len(df))
    c2.metric("Model Test Accuracy", f"{accuracy:.2f}%")
    c3.metric("Features Used", "Rating")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Target rule copied from report:** Rating <= 5 → Churn (1); Rating > 5 → Stay (0)")
    st.write("**Train/test split:** 80% training and 20% testing.")
    st.warning("The model target is calculated from Rating itself. For meaningful real-world machine learning, train against an actual historical churn label and use independent features such as balance, activity, and complaint history.")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted Stay", "Predicted Churn"])
    ax.set_yticks([0, 1], labels=["Actual Stay", "Actual Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Classification report"):
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
