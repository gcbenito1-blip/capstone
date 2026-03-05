import streamlit as st
import pandas as pd
from regression import train_regression

st.set_page_config(
    page_title="Model Evaluation",
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

st.title("Model Evaluation")
st.text("Performance metrics and validation results for the prediction model")


train_df = pd.read_csv("train.csv")

results, r2, rmse, mae = train_regression(train_df)

st.dataframe(results)