import streamlit as st
from tabs import tab1, tab2, tab3, tab4
import warnings
import pandas as pd
from model import train_test_split_shared, regression_model, classification_model
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NAT-lytics Dashboard",
    page_icon="assets/logo.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "A National Achievement Test Exploratory and Predictive Tool"
    }
)

st.markdown(
    """
    <style>
    .sub-text {
        color: grey;
    }
    </style>
    """, unsafe_allow_html=True
)
# Title
st.header(":pencil2: NAT-lytics Model Evaluation and Prediction Dashboard", anchor=False)
st.markdown('National Achievement Test Analysis & Prediction System')

t1, t2, t3, t4= st.tabs([":material/upload: Upload Data", ":material/finance: Exploratory Data Analysis", ":material/target: Prediction Results", ":material/search_insights: Model Evaluation" ])
with st.sidebar:
    st.sidebar.title(":rocket: Quick Start Guide")
    with st.container(border=True, ):
        st.markdown("""
        1. Create your own dataset file following the provided template.
        1. Upload your own dataset.
        1. Click 'Generate Results'
        """, unsafe_allow_html=True)

with t1:
    st.header("Data Management & Upload", anchor=False)
    st.markdown("""<p class="sub-text">Upload student data to generate NAT predictions and analysis</p>""", unsafe_allow_html=True)

    tab1.render()

with t2:
    st.header("Exploratory Data Analysis", anchor=False)
    # st.markdown("""<p class="sub-text">Understanding the characteristics and quality of the uploaded dataset</p>""", unsafe_allow_html=True)
    tab2.render()

with t4: 
    st.header('Model Summary', anchor=False)
    st.markdown("""<p class="sub-text">Model Training Summary and Evaluation</p>""", unsafe_allow_html=True)
    df = pd.read_csv("cd.csv")

    # Shared split
    split_data = train_test_split_shared(df)

    # -----------------------
    # Regression
    # -----------------------
    reg_pipe, X_test_reg, y_test_reg, reg_results, reg_metrics = regression_model(df, split_data)

    st.markdown("**Regression Metrics**")
    r1, r2, r3 = st.columns(3, border=True)
    r1.metric("R²", reg_metrics["R2"], help="Ranges from 0-1, Value Closer to 1, the better. 1='Perfect Prediction'")
    r2.metric("Root Mean Squared Error", reg_metrics["RMSE"], help="Difference between the actual and predicted score. Bigger mistakes count more.")
    r3.metric("Mean Average Error", reg_metrics["MAE"], help="Difference between the actual and predicted score. Treats all mistakes equally.")
    with st.expander("Regression Predictions"):
        st.dataframe(reg_results)
        fig, ax = plt.subplots()
        ax.scatter(y_test_reg, reg_results["Predicted_mps"], alpha=0.7)
        ax.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
        ax.set_xlabel("Actual mps")
        ax.set_ylabel("Predicted mps")
        ax.set_title("Predicted vs Actual")
        st.pyplot(fig)

    # -----------------------
    # Classification
    # -----------------------
    clf_pipe, X_test_clf, y_test_clf, clf_results, clf_metrics = classification_model(df, split_data)

    st.markdown("**Classification Metrics**")
    c1, c2, c3, c4, c5 = st.columns(5, border=True)
    c1.metric("Accuracy", clf_metrics["Accuracy"])
    c2.metric("Precision", clf_metrics["Precision"])
    c3.metric("Recall", clf_metrics["Recall"])
    c4.metric("F1", clf_metrics["F1"])
    c5.metric("ROC AUC", clf_metrics["ROC_AUC"])

    with st.expander("Classification Predictions"):
        st.dataframe(clf_results)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(clf_pipe, X_test_clf, y_test_clf, ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

with t3:
    st.header('Prediction Results', anchor=False)
    st.markdown("""<p class="sub-text">Predicted NAT outcomes for the uploaded dataset</p>""", unsafe_allow_html=True)
    tab3.render()