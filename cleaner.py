import pandas as pd
import streamlit as st
import joblib
import pickle

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

df1 = pd.read_csv("data/1.csv")
df2 = pd.read_csv("data/2.csv")
df3 = pd.read_csv("data/3.csv")

st.set_page_config(
    layout="wide",
    page_title="Dataset cleaner"
)

t1,t2,t3 = st.tabs(['dataset 1', 'dataset 2', 'dataset 3'], width='stretch')

def label(df, prof):
    mapping = {
        "Low Proficient": 39,
        "Nearly Proficient": 62.5,
        "Proficient": 82
    }

    if prof not in mapping:
        raise ValueError("Invalid proficiency label")

    df.copy()

    df['birthdate'] = pd.to_datetime(df[["month", "day", "year"]])
    df['proficiency'] = prof
    df['mps'] = mapping[prof]
    today = pd.Timestamp.today()
    df["age"] = (
        today.year - df["birthdate"].dt.year
        - (
            (today.month < df["birthdate"].dt.month) |
            (
                (today.month == df["birthdate"].dt.month) &
                (today.day < df["birthdate"].dt.day)
            )
        )
    )
    df = df.drop(columns=["month", "year", "day", "birthdate"]).copy()

    return df

df1 = label(df1, "Low Proficient")
df2 = label(df2, "Nearly Proficient")
df3 = label(df3, "Proficient")

with t1:
    st.dataframe(df1)
with t2:
    st.dataframe(df2)
with t3:
    st.dataframe(df3)

df = pd.concat([df1,df2,df3])

st.dataframe(df)
st.write("Data Shape",df.shape)
st.write("Data Types", df.dtypes)
missing = df.isnull().sum().sum()

numeric_cols = df.select_dtypes(include=["int64", "float64", "int32"]).columns
numeric_imputer = SimpleImputer(strategy="median")

preprocessor = ColumnTransformer([
    ("num", numeric_imputer, numeric_cols),
])

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

st.write("Missing values", missing)
st.write(df['proficiency'].value_counts())

st.header('Cleaned dataset')
st.dataframe(df)
st.write(df.dtypes)
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Cleaned Dataset",
    data=csv,
    file_name="cleaned_dataset.csv",
    mime="text/csv"
)