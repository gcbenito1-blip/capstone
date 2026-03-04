import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# -----------------------
# Load Data
# -----------------------
df = pd.read_csv("cd.csv")

# -----------------------
# Remove non-feature columns
# -----------------------
student_ids = df["studentID"]
X = df.drop(columns=["studentID", "mps", "proficiency"])

# -----------------------
# Column Types
# -----------------------
numeric_cols = X.select_dtypes(include=["int64", "float64", "int32"]).columns
categorical_cols = ["sex"]

# -----------------------
# Preprocessing
# -----------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# -----------------------
# Full Clustering Pipeline
# -----------------------
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=10))
])

# -----------------------
# Fit Model
# -----------------------
pipe.fit(X)

# -----------------------
# Assign Clusters
# -----------------------
clusters = pipe.named_steps["kmeans"].labels_

results = pd.DataFrame({
    "studentID": student_ids,
    "Cluster": clusters
})

st.dataframe(results)
st.bar_chart(results["Cluster"].value_counts())