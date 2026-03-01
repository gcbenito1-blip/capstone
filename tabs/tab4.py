
import streamlit as st
import streamlit_shadcn_ui as ui

def render():
    st.header('Prediction Results', anchor=False)
    st.markdown("""<p class="sub-text">Predicted NAT outcomes for the uploaded dataset</p>""", unsafe_allow_html=True)

