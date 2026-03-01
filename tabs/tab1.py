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

    st.header("Data Management & Upload", anchor=False)
    st.markdown("""<p class="sub-text"> Upload student data to generate NAT predictions and analysis</p>""", unsafe_allow_html=True)

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
        st.write("**Step 2: Upload Filled template**")

        dataset = st.file_uploader("Upload your completed CSV file",type="csv")
        tab1button = st.button(":material/query_stats: Generate Results", type="primary", key="tab1button", width="stretch")

        if tab1button and dataset is not None:
            df = pd.read_csv(dataset)
            st.session_state['tab1_ready'] = True
            st.session_state['uploaded_data'] = df
            st.success("Result Successfully Generated", icon="✅")