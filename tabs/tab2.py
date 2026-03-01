import streamlit_shadcn_ui as ui
import streamlit as st

def render():
    st.header("Exploratory Data Analysis", anchor=False)
    st.markdown("""<p class="sub-text">Understanding the characteristics and quality of the uploaded dataset</p>""", unsafe_allow_html=True)

    if not st.session_state.get("tab1_ready", False):
        st.warning(':material/warning: Please Upload your dataset first')
        st.stop()
    else:
        with st.container(border=True):
            st.text("Filters")
            st.markdown("""<p class="sub-text">Filter the data to analyze specific segments</p>""", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                with st.container():
                    age = st.selectbox(label="Age",
                    options=['1', '2', '3'],
                    key='1' 
                              )
            with col2:
                with st.container():
                    gender = st.selectbox(label="Gender",
                    options=['1', '2'],
                    key='2' 
                              )
            with col3:
                with st.container():
                    grade = st.selectbox(label="Grade Average",
                    options=['1', '2'],
                    key='3' 
                              )
            with col4:
                with st.container():
                    bmi= st.selectbox(label="BMI",
                    options=['1', '2'],
                    key='4'
                              )

        with st.container(border=True):
            st.write(age)
            st.write(gender)
            st.write(grade)
            st.write(bmi)