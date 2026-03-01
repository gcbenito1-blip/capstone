import streamlit as st
import streamlit_shadcn_ui as ui

def render():
    st.header('Model Evaluation', anchor=False)
    st.markdown("""<p class="sub-text">Evaluating the accuracy and reliability of the predictive model""", unsafe_allow_html=True)
    st.text('')
    if not st.session_state.get("tab1_ready", False):
        st.warning('Please Upload your dataset first')
        st.stop()
    else:
        st.markdown("**Regression Performance**")
        r1, r2, r3 = st.columns(3)
        with st.container():
            with r1:
                st.metric(
                    border=True,
                    label="Root Mean Squared Error",
                    help="This is a helper for this metric",
                    value=420,
                )
            with r2:
                with st.container(border=True):
                    st.metric(
                        label="Root Mean Squared Error",
                        help="This is the meaning of this metric",
                        value=420,
                    )
            with r3:
                st.metric(
                    border=True,
                    label="RMSE",
                    help="This is a helper for this metric",
                    value=420,
                )



