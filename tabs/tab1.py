import streamlit as st
import pandas as pd
import streamlit_shadcn_ui as ui

def render():
    @st.cache_data
    def get_data():
        df = pd.read_csv("data/1.csv")
        df = df.iloc[0:0]
        return df.to_csv(index=False)
    template= get_data()


    # t1, t2 = st.tabs(tabs=["Bulk Analysis", "Individual Analysis"])
    # with t1:
    with st.container(border=True, ):
        st.write("**Step 1: Download Template**")
        st.markdown("""
        Download the standardized CSV template with required fields
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("""
            **Required Fields:**
            * Student Demographics(Gender, Age, Mother Tongue, BMI)
            * Final Ratings in 5 subjects from Grade 1 to Grade 6
            * Attendance Percentage
            """)


        st.download_button(
            label=":material/download: Download CSV Template",
            data=template,
            file_name="template.csv",
            mime="text/csv",
            type="secondary",
            width="stretch"
            )

    with st.container(border=True, ):
        is_disabled=True
        st.write("**Step 2: Upload Filled template**")

        dataset = st.file_uploader("Upload your completed CSV file",type="csv")
        if dataset:
            is_disabled = False
            df = pd.read_csv(dataset)

        tab1button = st.button(":material/query_stats: Upload Dataset", type="primary", key="tab1button", width="stretch", disabled=is_disabled)

        if tab1button and dataset is not None:
            st.session_state['tab1_ready'] = True
            st.session_state['uploaded_data'] = df
            st.success("Dataset Ready for Exploratory Analysis", icon="✅")