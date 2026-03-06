import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import require_data, get_data
import joblib
import shap

st.set_page_config(
    page_title="Prediction Results",
    layout="wide",
    page_icon=":pencil2:"
)

with st.spinner("Loading Prediction"):
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
    X = df.drop(columns=["studentID"], errors="ignore")
    r_model = joblib.load("models/regression_model.joblib")

    r_prediction = r_model.predict(X)

    df['Regression Prediction']=r_prediction
    st.dataframe(df[['studentID', 'School', 'age', 'sex', 'BMI/nutrional status', 'mother tongue', 'Regression Prediction', 'Classification Prediction']], hide_index=True)
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

    def plot_contributions(model, X, student_index):
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        contrib = pd.Series(
            shap_values.values[student_index],
            index=X.columns
        ).sort_values()
        fig, ax = plt.subplots(figsize=(8,5))
        contrib.plot(kind="barh", ax=ax)
        ax.set_title("Predictor Contributions to Predicted MPS")
        ax.set_xlabel("Contribution Value")
        st.pyplot(fig)

    def plot_contributions(pipeline_model, X, student_index):
        preprocessor = pipeline_model.named_steps["preprocessor"]
        model = pipeline_model.named_steps["model"]
        # transform features
        X_transformed = preprocessor.transform(X)
        # get feature names after encoding
        feature_names = preprocessor.get_feature_names_out()
        # create explainer for the trained model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        contrib = pd.Series(
            shap_values[student_index],
            index=feature_names
        ).sort_values()
        fig, ax = plt.subplots(figsize=(8,5))
        contrib.tail(15).plot(kind="barh", ax=ax)
        ax.set_title("Predictor Contributions to Predicted MPS")
        ax.set_xlabel("SHAP Contribution")
        st.pyplot(fig)

    st.markdown("**Which predictors influenced each learner’s predicted MPS.**")
    student_id = st.selectbox("Select Student", df["studentID"])
    with st.spinner("Loading Feature Contribution with Regression Model"):
        student_index = df.index[df["studentID"] == student_id][0]
        plot_contributions(r_model, X, student_index)