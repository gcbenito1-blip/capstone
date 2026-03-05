import streamlit as st
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
st.write(df["Classification Prediction"].value_counts())