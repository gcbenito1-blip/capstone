import streamlit as st

def render():
    st.header('Model Evaluation', anchor=False)
    st.text('Evaluating the accuracy and reliability of the predictive model')
    with st.container(width=500, border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("hello")
        with col2:
            st.button("button")

