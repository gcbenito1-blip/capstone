import streamlit as st
import pandas as pd
import numpy as np

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
    if dataset is not None:

        df = pd.read_csv(dataset)

        st.success("Dataset successfully loaded")

        # -------------------------
        # DATASET SUMMARY
        # -------------------------
        st.subheader("Dataset Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Records", len(df))

        with col2:
            st.metric("Total Features", df.shape[1])

        # -------------------------
        # MISSING VALUES
        # -------------------------
        st.subheader("Missing Values")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) > 0:
            st.dataframe(missing.rename("Missing Count"))
        else:
            st.info("No missing values detected")

        # -------------------------
        # SCHOOL DISTRIBUTION
        # -------------------------
        st.subheader("School Distribution")

        school_counts = df["School"].value_counts()

        st.dataframe(school_counts)
        # -------------------------
        # DEMOGRAPHIC SUMMARY
        # -------------------------
        st.subheader("Demographic Summary")

        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            st.write("**Gender Distribution**")
            st.dataframe(df["sex"].value_counts())

        with dcol2:
            st.write("**Mother Tongue**")
            st.dataframe(df["mother tongue"].value_counts())

        with dcol3:
            st.write("**BMI Categories**")
            st.dataframe(df["BMI/nutrional status"].value_counts())
        # -------------------------
        # OUTLIER DETECTION
        # -------------------------
        st.subheader("Outlier Detection (Z-score > 3)")
        numeric_cols = df.select_dtypes(include=np.number).columns
        outlier_counts = {}
        for col in numeric_cols:
            z = (df[col] - df[col].mean()) / df[col].std()
            outlier_counts[col] = (z.abs() > 3).sum()
        st.dataframe(pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["Outliers"]))
        # -------------------------
        # PREPROCESSING PLAN
        # -------------------------
        st.subheader("Preprocessing Pipeline")

        st.markdown("""
        **The following preprocessing will be applied:**

        1. **Handle Missing Values**
           - Numerical → Median Imputation  
           - Categorical → Most Frequent Imputation

        2. **Outlier Handling**
           - Remove or cap outliers in numeric features (Z-score > 3)

        3. **Feature Engineering**
           - Categorical variables → One-Hot Encoding  
           - Numerical features → Standard Scaling

        4. **Exclude Student ID** from model features

        5. **Train-Test Split** for modeling
        """)

        # -------------------------
        # CONFIRM DATASET
        # -------------------------
        if st.button("Confirm Dataset and Continue", type="primary", width="stretch"):
            st.session_state["uploaded_data"] = df
            st.session_state["data_ready"] = True
            st.success(":check: Dataset ready for Exploratory Data Analysis")
            st.switch_page(
                "pages/eda.py",
            )