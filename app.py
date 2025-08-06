import streamlit as st
import pickle
import pandas as pd

# Load model
with open("Loan.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.markdown(
    "<h2 style='text-align:center; color:#2c3e50;'>🏦 Loan Approval Predictor</h2>",
    unsafe_allow_html=True,
)

# Sidebar for inputs
st.sidebar.header("Enter Applicant Details")

inputs = {
    "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
    "Married": st.sidebar.selectbox("Married", ["Yes", "No"]),
    "Dependents": st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"]),
    "Education": st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"]),
    "Self_Employed": st.sidebar.selectbox("Self Employed", ["Yes", "No"]),
    "ApplicantIncome": st.sidebar.number_input(
        "Applicant Income", min_value=0.0, format="%.2f"
    ),
    "CoapplicantIncome": st.sidebar.number_input(
        "Coapplicant Income", min_value=0.0, format="%.2f"
    ),
    "LoanAmount": st.sidebar.number_input("Loan Amount", min_value=0.0, format="%.2f"),
    "Loan_Amount_Term": st.sidebar.number_input(
        "Loan Amount Term(in days)", min_value=0.0, format="%.2f"
    ),
    "Credit_History": st.sidebar.selectbox("Credit History", ["1.0", "0.0"]),
    "Property_Area": st.sidebar.selectbox(
        "Property Area", ["Urban", "Semiurban", "Rural"]
    ),
}

df = pd.DataFrame([inputs])

# ---------------------------
# MANUAL ENCODING
# ---------------------------
encoding_maps = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Credit_History": {"1.0": 1, "0.0": 0},
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
}

for col, mapping in encoding_maps.items():
    df[col] = df[col].map(mapping)

# ---------------------------
# FIXED MIN-MAX SCALING
# ---------------------------
scaling_values = {
    "ApplicantIncome": (1500, 81000),
    "CoapplicantIncome": (0.0, 42000),
    "LoanAmount": (10.0, 700.0),
    "Loan_Amount_Term": (12.0, 480.0),
}

for col, (min_val, max_val) in scaling_values.items():
    df[col] = (df[col] - min_val) / (max_val - min_val)

# ---------------------------
# Predict and display result
# ---------------------------
if st.button("💡 Predict Loan Approval"):
    prediction = model.predict(df)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"
    st.markdown(
        f"<h3 style='text-align:center; color:#006d77;'>Prediction: {result}</h3>",
        unsafe_allow_html=True,
    )

    with st.expander("See Details"):
        st.dataframe(df.style.highlight_max(axis=1, color="lightgreen"))

else:
    st.markdown(
        "<p style='text-align:center; color:gray;'>👈 Fill out the details and hit Predict!</p>",
        unsafe_allow_html=True,
    )

