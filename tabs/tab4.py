
import streamlit as st
import streamlit_shadcn_ui as ui

num = 50

def render():
    st.header('Tab 4 content', anchor=False)

    trigger_btn = ui.button(text="Trigger Button", key="trigger_btn")

    ui.alert_dialog(
        show=trigger_btn, 
        title="Alert Dialog", 
        description="This is an alert dialog", 
        confirm_label="OK", 
        cancel_label="Cancel", 
        key="alert_dialog1"
    )

    @st.dialog(title="Hello")
    def hello1():
        st.write("hello")

    @st.dialog(title="New")
    def hello2():
        st.write("hello hello")

    @st.dialog(title="three")
    def hello3():
        st.write("hello hello hello")

    n1=st.button("1")
    n2=st.button("2")
    n3=st.button("3")
    if n1:
        hello1()
    elif n2:
        hello2()
    elif n3:
        hello3()
