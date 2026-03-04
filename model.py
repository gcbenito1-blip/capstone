# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    r2_score, root_mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# -----------------------
# Shared Train/Test Split
# -----------------------
def train_test_split_shared(df, test_size=0.2, random_state=42):
    """
    Returns a dictionary containing X_train, X_test, y_train, y_test for regression,
    and X_train_clf, X_test_clf, y_train_clf, y_test_clf for classification,
    ensuring both use the same student split.
    """
    std_id = df['studentID']
    
    # Split indices only
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state
    )
    
    split_data = {
        "train_idx": train_idx,
        "test_idx": test_idx
    }
    
    return split_data

# -----------------------
# Regression Pipeline
# -----------------------
def regression_model(df, split_data):
    df_model = df.drop(columns=["proficiency"])
    std_id = df_model['studentID']

    X = df_model.drop(columns=["mps", "studentID"])
    y = df_model["mps"]

    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32"]).columns
    categorical_cols = ["sex"]

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

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ])

    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    id_test = std_id.loc[test_idx]

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    results = pd.DataFrame({
        "studentID": id_test,
        "Actual_mps": y_test,
        "Predicted_mps": pred
    })
    results["Error"] = results["Actual_mps"] - results["Predicted_mps"]
    results = results.reset_index(drop=True)

    metrics = {
        "R2": round(r2_score(y_test, pred), 2),
        "RMSE": round(root_mean_squared_error(y_test, pred), 2),
        "MAE": round(mean_absolute_error(y_test, pred), 2)
    }

    return pipe, X_test, y_test, results, metrics

# -----------------------
# Classification Pipeline
# -----------------------
def classification_model(df, split_data):
    X = df.drop(columns=["proficiency", "mps"])
    y = df["proficiency"]
    student_ids = X["studentID"]
    X = X.drop(columns=["studentID"])

    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32"]).columns
    categorical_cols = ["sex"]

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

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    id_test = student_ids.loc[test_idx]

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)

    results = pd.DataFrame({
        "studentID": id_test,
        "Actual_proficiency": y_test,
        "Predicted_proficiency": pred
    }).reset_index(drop=True)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, pred), 2),
        "Precision": round(precision_score(y_test, pred, average="weighted"), 2),
        "Recall": round(recall_score(y_test, pred, average="weighted"), 2),
        "F1": round(f1_score(y_test, pred, average="weighted"), 2),
        "ROC_AUC": round(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"), 2)
    }

    return pipe, X_test, y_test, results, metrics