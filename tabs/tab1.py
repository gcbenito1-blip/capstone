import streamlit as st

def render():

    st.header("Data Management & Upload", anchor=False)
    st.text("Upload student data to generate NAT predictions and analysis")
    with st.container(border=True, ):
            st.write("**Step 1**")
            st.markdown("""
            This is step 1
            """, unsafe_allow_html=True)
            school = st.selectbox(
                "Please Select Your School",
                ("Urdaneta I Central School", "School 1", "School2"),
            )

    with st.container(border=True, ):
        st.write("**Step 2**")
        st.markdown("""
        This is step 2,
                    Download Template
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("""
            **Required Fields:**
            * Student Demographics(Gender, Age, Mother Tongue, BMI)
            * Final Ratings in 5 subjects from Grade 1 to Grade 6
            * Attendance Percentage
            """)
        template_download = st.download_button(label=":material/download: Download CSV Template", data="hello")

    with st.container(border=True, ):
        st.write("**Step 3**")
        st.markdown("""
        This is step 3
        """, unsafe_allow_html=True)

    dataset = st.file_uploader(label="Upload Data", type="csv")
    tab1button = st.button("Generate Results")

    if tab1button:
        st.session_state['tab1_ready'] = True
        st.success("Result Successfully Generated", icon="✅")
        st.button("Navigate to")