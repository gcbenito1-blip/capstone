# Prediction Tab
import streamlit as st
from model1 import load_models

def render():
    if not st.session_state.get("tab2_ready", False):
        st.warning("Upload Your Data First")
    else:
        test = st.session_state['dataset']
        reg_pipe, clf_pipe = load_models()
        reg_pipe = reg_pipe.predict(test)
        clf_pipe = clf_pipe.predict(test)

        st.write(reg_pipe)
        st.write(clf_pipe)