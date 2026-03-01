
import streamlit as st

def render():
    if not st.session_state.get("tab1_ready", False):
        st.warning('Please Upload your dataset first')
        st.stop()
    else:
        st.header("Exploratory Data Analysis", anchor=False)
        st.text("Understanding the characteristics and quality of the uploaded dataset")

        with st.container(border=True):
            st.text("Filters")
            st.text("Filter the data to analyze specific segments")

            gender = st.selectbox(label='Gender', options=["Male","Female"])

        with st.container(border=True):
            st.write(gender)