import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error


st.set_page_config(
    layout="wide",
    page_title="Regression Model",
    page_icon=":material/trending_up:"
)
df = pd.read_csv('cd.csv')
# -----------------------
# Feature / Target Split
# -----------------------
df_model = df.drop(columns=["proficiency"])
std_id = df_model['studentID']

X = df_model.drop(columns=["mps", "studentID"])
y = df_model["mps"]

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
# Full Pipeline
# -----------------------
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge())
])

# -----------------------
# Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, std_id,
    test_size=0.2, 
    random_state=42
)

# -----------------------
# Train
# -----------------------
pipe.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
pred = pipe.predict(X_test)
 
results = pd.DataFrame({
    "studentID": id_test,
    "Actual_mps": y_test,
    "Predicted_mps": pred
})

results["Error"] = results["Actual_mps"] - results["Predicted_mps"]

results = results.reset_index(drop=True)

r2 = round(r2_score(y_test, pred), 2)
rmse = round(root_mean_squared_error(y_test, pred), 2)
mae = round(mean_absolute_error(y_test, pred), 2)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("R2 metric", value=r2, border=True)
with c2:
    st.metric("Root Mean Squared Error", value=rmse, border=True)
with c3:
    st.metric("Mean Absolute Error", value=mae, border=True)
