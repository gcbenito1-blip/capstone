import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 

# -----------------------
# Load Data
# -----------------------
df = pd.read_csv("cd.csv")

st.set_page_config(
    layout='wide',
    page_title="Classification Model",
    page_icon=":material/category:"
)
# -----------------------
# Remove target-independent columns
# -----------------------
X = df.drop(columns=["proficiency", "mps"])  # mps removed, proficiency is target
y = df["proficiency"]
student_ids = X["studentID"]  # save for reference
X = X.drop(columns=["studentID"])  # remove from features

# -----------------------
# Column Types
# -----------------------
numeric_cols = X.select_dtypes(include=["int64", "float64", "int32"]).columns
categorical_cols = ["sex"]

# -----------------------
# Preprocessing Pipelines
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
    ("model", RandomForestClassifier(random_state=42))
])

# -----------------------
# Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, student_ids,
    test_size=0.2,
    random_state=42,
    stratify=y  # maintain class distribution
)

# -----------------------
# Train Model
# -----------------------
pipe.fit(X_train, y_train)

# -----------------------
# Predict
# -----------------------
pred = pipe.predict(X_test)

# -----------------------
# Results DataFrame
# -----------------------
results = pd.DataFrame({
    "studentID": id_test,
    "Actual_proficiency": y_test,
    "Predicted_proficiency": pred
}).reset_index(drop=True)


# -----------------------
# Evaluation
# -----------------------

# Binary or multi-class: specify average
acc = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average="weighted")  # weighted average per class
recall = recall_score(y_test, pred, average="weighted")
f1 = f1_score(y_test, pred, average="weighted")

# For ROC AUC in multi-class, use predicted probabilities
proba = pipe.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")

c1, c2, c3, c4, c5 = st.columns(5, border=True)
with c1:
    st.metric("Accuracy:", round(acc, 2), help="Helper")
with c2:
    st.metric("Precision:", round(precision, 2), help='Helper')
with c3:
    st.metric("Recall:", round(recall, 2), help='Helper')
with c4:
    st.metric("F1-score:", round(f1, 2), help='Helper')
with c5:
    st.metric("ROC AUC:", round(roc_auc, 2), help='Helper')
st.header("Prediction Results")
st.dataframe(results)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)