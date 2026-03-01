import streamlit_shadcn_ui as ui
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tabs import tab1, tab2, tab3, tab4
import warnings

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
st.header(":pencil2: NAT-Lytics Model Evaluation and Prediction Dashboard", anchor=False)
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
    st.markdown("""<p class="sub-text">Understanding the characteristics and quality of the uploaded dataset</p>""", unsafe_allow_html=True)
    tab2.render()

with t4: 
    st.header('Model Summary', anchor=False)
    st.markdown("""<p class="sub-text">Model Training Summary and Evaluation</p>""", unsafe_allow_html=True)

    tab3.render()

with t3:
    st.header('Prediction Results', anchor=False)
    st.markdown("""<p class="sub-text">Predicted NAT outcomes for the uploaded dataset</p>""", unsafe_allow_html=True)

    tab4.render()