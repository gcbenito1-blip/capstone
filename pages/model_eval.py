import streamlit as st
import pandas as pd
from regression import regression_model 
from classification import  classification_model
st.set_page_config(
    page_title="Model Evaluation",
    layout="wide",
    page_icon=":pencil2:"
)
with st.spinner("Loading Model Evaluation"):
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

    train_data = pd.read_csv("train.csv")

    results, r2, rmse, mae, fi = regression_model(train_data)
    st.markdown("**Regression Performance Metrics**")
    c1, c2, c3 = st.columns(3, border=True)
    with c1:
        st.metric("R² Score", round(r2,2))
    with c2:
        st.metric("RMSE", round(rmse,2)) 
    with c3:
        st.metric("MAE", round(mae,2))
    st.markdown("**Feature Importance Analysis For Regression Analysis**")
    st.text("Relative contribution of predictors to model performance")
    st.bar_chart(fi.set_index("Feature"))