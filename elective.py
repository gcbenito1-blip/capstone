import streamlit as st
import streamlit_shadcn_ui as ui
from elective_tab import eTab1, eTab2, eTab3
import pandas as pd

with st.sidebar:
    st.sidebar.title(":rocket: Quick Start Guide")
    with st.container(border=True, ):
        st.markdown("""
        1. Create your own dataset file following the provided template.
        1. Upload your own dataset.
        1. Click 'Generate Results'
        """, unsafe_allow_html=True)
@st.cache_data
def load_data():
    df = pd.read_csv("data/egovph.csv")
    return df


df=load_data()

# Page Setup
st.set_page_config(
    page_title="eGovPH Review Insight",
    page_icon=":philippines:",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "Text Mining and Sentiment Analysis"
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
st.title(":material/bar_chart: eGovPH Review Insight", anchor=False)
st.markdown("<p class='sub-text'>Text Mining and Sentiment analysis for eGovPH Android App</p>", unsafe_allow_html=True)

with st.container(border=True):
    u1, u2, u3 = st.tabs([':material/dashboard: Overview', ':material/dictionary: Text Mining Analysis', ':material/cognition_2: Sentiment Analysis'])

with u1:
    eTab1.render(df=df)
with u2:
    eTab2.render(df=df)
with u3:
    eTab3.render(df=df)
