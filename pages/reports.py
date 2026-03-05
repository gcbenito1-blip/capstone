import streamlit as st
from utils import require_data, get_data

st.set_page_config(
    page_title="Reports and Evaluation",
    layout="wide",
    page_icon=":pencil2:"
)

with st.spinner("Loading Reports and Evaluation"):
    with st.sidebar:
        st.title("NAT-lytics")
        st.subheader("Model Evaluation Dashboard")
        st.divider()
        st.page_link("app.py", label="Dataset Upload", icon=":material/upload:")
        st.page_link("pages/eda.py", label="Exploratory Data Analysis", icon=":material/analytics:")
        st.page_link("pages/pred.py", label="Prediction Results", icon=":material/psychology:")
        st.page_link("pages/model_eval.py", label="Model Evaluation", icon=":material/monitoring:")
        st.page_link("pages/reports.py", label="Reports & Evaluation", icon=":material/description:")

    st.title("Report and Export")
    st.text("Download predictions, visualizations, and model performance reports")

    require_data()

    df = get_data()

    with st.container(border=True):
        st.markdown("**:material/download:** All exports are generated based on current filter settings and data loaded in the system.")

    # Comprehensive Reports
    with st.container(border=True):
        st.markdown("**Comprehensive Reports**")
        st.text("Complete documentation and analysis packages")

        with st.container(border=True):
            c1, c2 = st.columns([4, 1], vertical_alignment="center")
            with c1:
                st.markdown("**:material/docs: Complete Analysis Report**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c2:
                st.button("download", key="1")
        with st.container(border=True):
            c1, c2 = st.columns([4, 1], vertical_alignment="center", width="stretch")
            with c1:
                st.markdown("**:material/docs: Complete Analysis Report**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c2:
                st.button("download", key="2")
        with st.container(border=True):
            c1, c2 = st.columns([4, 1], vertical_alignment="center")
            with c1:
                st.markdown("**:material/docs: Complete Analysis Report**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c2:
                st.button("download", key="3")

    # Prediction Data
    with st.container(border=True):
        st.markdown("**Comprehensive Reports**")
        st.text("Complete documentation and analysis packages")

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown("::material/file")
            with c2:
                st.markdown("**Complete Analysis Report**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c3:
                st.markdown("download")
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown("::material/file")
            with c2:
                st.markdown("**Executive Summary**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c3:
                st.markdown("download")
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown("::material/file")
            with c2:
                st.markdown("**Full Dataset Export**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c3:
                st.markdown("download")


    # Model Performance 
    with st.container(border=True):
        st.markdown("**Comprehensive Reports**")
        st.text("Complete documentation and analysis packages")

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown("::material/file")
            with c2:
                st.markdown("**Complete Analysis Report**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c3:
                st.markdown("download")
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown("::material/file")
            with c2:
                st.markdown("**Executive Summary**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c3:
                st.markdown("download")
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown("::material/file")
            with c2:
                st.markdown("**Full Dataset Export**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c3:
                st.markdown("download")

    # Visualizations
    with st.container(border=True):
        st.markdown("**Comprehensive Reports**")
        st.text("Complete documentation and analysis packages")

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown(":material/docs:")
            with c2:
                st.markdown("**Complete Analysis Report**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c3:
                st.markdown("download")
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown(":material/docs:")
            with c2:
                st.markdown("**Executive Summary**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c3:
                st.markdown("download")
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown(":material/docs:")
            with c2:
                st.markdown("**Full Dataset Export**")
                st.text("All data including raw inputs, predictions, and metadata")
            with c3:
                st.markdown("download")