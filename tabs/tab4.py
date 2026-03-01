
import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd

df = pd.read_csv('data/1.csv')

def render():
    if not st.session_state.get("tab1_ready", False):
        st.warning("Upload Your Data First")
    else:
        st.dataframe(df)

