
import streamlit as st
import streamlit_shadcn_ui as ui

num = 50

def render():
    st.header('Prediction Results', anchor=False)
    st.markdown("""<p class="sub-text">Predicted NAT outcomes for the uploaded dataset</p>""", unsafe_allow_html=True)

    # trigger_btn = ui.button(text="Trigger Button", key="trigger_btn")
    # ui.alert_dialog(
    #     show=trigger_btn, 
    #     title="Alert Dialog", 
    #     description="This is an alert dialog", 
    #     confirm_label="OK", 
    #     cancel_label="Cancel", 
    #     key="alert_dialog1"
    # )
