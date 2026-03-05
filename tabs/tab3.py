# Prediction Tab
import streamlit as st

def render():
    if not st.session_state.get("tab2_ready", False):
        st.warning("Upload Your Data First")
    else:
        test = st.session_state['dataset']
        st.dataframe(test)