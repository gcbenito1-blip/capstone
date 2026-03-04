import streamlit as st
import streamlit_shadcn_ui as ui
import matplotlib.pyplot as plt
import pandas as pd


def render(df):
    st.header(":material/dashboard: Dataset Overview", anchor=False)
    df = pd.read_csv('data/egovph.csv')
    df['at'] = pd.to_datetime(df['at']).dt.date
    df = df.sort_values('at')

    st.write(df.dtypes)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Number of rows",
            value=df.shape[0],
        )
    with col2:
        st.metric(
            "Number of columns",
            value=df.shape[1],
        )

    new = df.groupby(df['at']).size()
    pie = df['score'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie, labels=pie.index)
    ax.set_title("Score Distribution")
    st.pyplot(fig)
    st.line_chart(data=new, y_label="Number of Post")
