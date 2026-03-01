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

# Custom CSS
st.markdown("""
    <style>
    h1 a{
        display:none; 
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1
    }

    .card {
    background-color: white; /* Background color */
    border-radius: 8px; /* Rounded corners */
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); /* Shadow effect */
    margin: 20px; /* Space around cards */
    padding: 20px; /* Inner spacing */
    transition: transform 0.2s; /* Animation effect */
    }

    .card:hover {
        transform: scale(1.05); /* Scale up on hover */
    }

    .card-title {
        font-size: 24px; /* Title font size */
        color: #333; /* Title color */
    }

    .card-content {
        font-size: 16px; /* Content font size */
        color: #666; /* Content color */
    }

    .card-button {
        background-color: #4CAF50; /* Button background */
        border: none; /* No border */
        color: white; /* Button text color */
        padding: 10px 15px; /* Button padding */
        text-align: center; /* Center alignment */
        text-decoration: none; /* No underline */
        display: inline-block; /* Inline-block for button */
        margin-top: 10px; /* Space above button */
        border-radius: 5px; /* Rounded corners for button */
        cursor: pointer; /* Pointer on hover */
    }

    .sub-text{
        font-size: 12px;
        color: #6b7280;
        line-height: 1.4;
    }

    </style>
""", unsafe_allow_html=True) 

# Title
st.header(":pencil2: NAT-Lytics Model Evaluation and Prediction Dashboard", anchor=False)
st.markdown('National Achievement Test Analysis & Prediction System')

t1, t2, t3, t4= st.tabs([":material/upload: Upload Data", ":material/finance: Exploratory Data Analysis", ":material/search_insights: Model Evaluation", ":material/target: Prediction Results"])
with st.sidebar:
    st.sidebar.title(":rocket: Quick Start Guide")
    with st.container(border=True, ):
        st.markdown("""
        1. Create your own dataset file following the provided template.
        1. Upload your own dataset.
        1. Click 'Generate Results'
        """, unsafe_allow_html=True)

with t1:
    tab1.render()

with t2:
    tab2.render()

with t3:
    tab3.render()

with t4:
    tab4.render()