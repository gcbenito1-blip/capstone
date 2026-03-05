import streamlit as st

def require_data():
    if not st.session_state.get("data_ready", False):
        st.warning("Upload dataset first from Dataset Upload page.")
        st.stop()

def get_data():
    return st.session_state.get("uploaded_data")