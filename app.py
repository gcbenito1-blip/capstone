import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="NAT-lytics",
    layout="wide",
    page_icon=":pencil2:"
)

with st.sidebar:
    st.title("NAT-lytics")
    st.subheader("Model Evaluation Dashboard")
    st.divider()
    st.page_link("app.py", label="Dataset Upload", icon=":material/upload:")
    st.page_link("pages/eda.py", label="Exploratory Data Analysis", icon=":material/analytics:")
    st.page_link("pages/pred.py", label="Prediction Results", icon=":material/psychology:")
    st.page_link("pages/model_eval.py", label="Model Evaluation", icon=":material/monitoring:")
    st.page_link("pages/reports.py", label="Reports & Evaluation", icon=":material/description:")
st.header("Dataset Upload & Preprocessing", anchor=False)
st.text("Upload historical academic and demographic records for analysis")
if "data_ready" not in st.session_state:
    st.session_state.data_ready = False

@st.cache_data
def get_data():
    df = pd.read_csv("actual_test.csv")
    df = df.iloc[0:0]
    return df.to_csv(index=False)
template= get_data()


with st.container(border=True, ):
    st.write("**:material/download: Download Data Template**")
    st.markdown("""
    Get the standardized CSV template with all required fields
    """, unsafe_allow_html=True)
    st.space("xxsmall")
    st.markdown("""
    Template includes:
    *   Student demographics (ID, School, Age, Sex, Mother Tongue, BMI/Nutritional Status)
    *   Academic records from Grade 1 to Grade 6
    *   Final ratings for Math, English, Science, Filipino, and Araling Panlipunan
    *   Sample data rows for reference
    """)

    st.download_button(
        label=":material/download: Download CSV Template",
        data=template,
        file_name="template.csv",
        mime="text/csv",
        type="secondary",
        width="stretch"
        )
    with st.container(border=True):
        st.markdown("""
            Valid values:
            * **Sex:** F, M
            * **Mother Tongue:** Filipino, Tagalog, Ilocano
            * **BMI/Nutritional Status:** Obese, Overweight, Normal, Wasted, Severely Wasted
            * **Grade Ratings:** 75-100 (passing grades)
        """)


with st.container(border=True, ):
    is_disabled=True
    st.write("**Upload Filled template**")

    dataset = st.file_uploader("Upload your completed CSV file",type="csv")
    if dataset:
        is_disabled = False
        df = pd.read_csv(dataset)

    tab1button = st.button(":material/query_stats: Upload Dataset", type="primary", key="tab1button", width="stretch", disabled=is_disabled)

    if tab1button and dataset is not None:
        st.session_state['uploaded_data'] = df
        st.session_state['data_ready'] = True
        st.success("Dataset Ready for Exploratory Analysis", icon="✅")