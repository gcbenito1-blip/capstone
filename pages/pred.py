import streamlit as st
import matplotlib.pyplot as plt
from utils import require_data, get_data
import joblib

st.set_page_config(
    page_title="Prediction Results",
    layout="wide",
    page_icon=":pencil2:"
)

with st.sidebar:
    st.title("NAT-lytics")
    st.subheader("Model Evaluation Dashboard")
    st.divider()
    st.page_link("app.py", label="Dataset Upload", icon=":material/upload:")
    st.page_link("pages/eda.py", label="Exploratory Data Analysis", icon=":material/analytics:")
    st.page_link("pages/pred.py", label="Prediction Results", icon=":material/psychology:")
    st.page_link("pages/model_eval.py", label="Model Evaluation", icon=":material/monitoring:")
    st.page_link("pages/reports.py", label="Reports & Evaluation", icon=":material/description:")

st.title("Prediction Results")
st.text("Learner-level MPS predictions and proficiency probabilities")

require_data()

df = get_data()
r_model = joblib.load("models/regression_model.joblib")
c_model = joblib.load("models/classification_model.joblib")

r_prediction = r_model.predict(df)
c_prediction = c_model.predict(df)

df['Regression Prediction']=r_prediction
df['Classification Prediction']=c_prediction
st.dataframe(df[['studentID', 'School', 'age', 'sex', 'BMI/nutrional status', 'mother tongue', 'Regression Prediction', 'Classification Prediction']])
# st.write(df["Classification Prediction"].value_counts())
p1, p2 = st.columns(2)
with p1:
    counts = df["Classification Prediction"].value_counts()
    labels = [f"{label} ({count})" for label, count in zip(counts.index, counts.values)]
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=labels, autopct="%1.1f%%")
    ax.set_title("Predicted Proficiency Distribution")

    st.pyplot(fig)
with p2:
    st.markdown("**Average MPS by School**")
    school_perf = df.groupby("School")["Regression Prediction"].mean()
    st.bar_chart(school_perf)

s1, s2 = st.columns(2)
with s1:
    bmi_perf = df.groupby("BMI/nutrional status")["Regression Prediction"].mean().sort_values()
    st.markdown("**Average MPS Predicted by BMI Category**")
    st.bar_chart(bmi_perf)
with s2:
    st.markdown("**Average MPS Predicted by Gender**")
    gender_perf = df.groupby("sex")["Regression Prediction"].mean()
    st.bar_chart(gender_perf)