import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

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

df = pd.DataFrame([inputs])  # Keep original for PDF

# Manual Encoding
encoding_maps = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Credit_History": {"1.0": 1, "0.0": 0},
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
}

encoded_df = df.copy()

for col, mapping in encoding_maps.items():
    encoded_df[col] = encoded_df[col].map(mapping)

# Scaling
scaling_values = {
    "ApplicantIncome": (1500, 81000),
    "CoapplicantIncome": (0.0, 42000),
    "LoanAmount": (10.0, 700.0),
    "Loan_Amount_Term": (12.0, 480.0),
}

for col, (min_val, max_val) in scaling_values.items():
    encoded_df[col] = (encoded_df[col] - min_val) / (max_val - min_val)

# Charts
st.subheader("📊 Income Distribution")

# Bar chart
st.write("**Applicant vs Coapplicant Income**")
bar_df = pd.DataFrame(
    {
        "Type": ["Applicant Income", "Coapplicant Income"],
        "Income": [inputs["ApplicantIncome"], inputs["CoapplicantIncome"]],
    }
)
st.bar_chart(bar_df.set_index("Type"))

# Pie chart
if inputs["ApplicantIncome"] == 0 and inputs["CoapplicantIncome"] == 0:
    st.warning("📌 Cannot plot pie chart because both incomes are zero.")
else:
    pie_values = [inputs["ApplicantIncome"], inputs["CoapplicantIncome"]]
    pie_labels = ["Applicant", "Coapplicant"]

    filtered_data = [
        (label, value) for label, value in zip(pie_labels, pie_values) if value > 0
    ]

    if filtered_data:
        labels, values = zip(*filtered_data)
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.warning("📌 Cannot plot pie chart because all income values are zero.")

# Predict and Display
if st.button("💡 Predict Loan Approval"):
    prediction = model.predict(encoded_df)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"
    st.markdown(
        f"<h3 style='text-align:center; color:#006d77;'>Prediction: {result}</h3>",
        unsafe_allow_html=True,
    )

    with st.expander("See Details"):
        st.dataframe(df.style.highlight_max(axis=1, color="lightgreen"))

    # PDF Report Generator using original inputs
    def create_pdf(original_inputs, prediction_result):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        title = Paragraph("Loan Prediction Report", styles["Title"])
        elements.append(title)
        elements.append(Spacer(1, 12))

        result_para = Paragraph(
            f"<b>Prediction:</b> {prediction_result}", styles["Heading2"]
        )
        elements.append(result_para)
        elements.append(Spacer(1, 12))

        table_data = [["Feature", "Value"]]
        for key, value in original_inputs.items():
            table_data.append([key, str(value)])

        table = Table(table_data, colWidths=[200, 200])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.gray),
                ]
            )
        )
        elements.append(table)

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf = create_pdf(inputs, result)
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf,
        file_name="loan_prediction_report.pdf",
        mime="application/pdf",
    )

else:
    st.markdown(
        "<p style='text-align:center; color:gray;'>👈 Fill out the details and hit Predict!</p>",
        unsafe_allow_html=True,
    )
