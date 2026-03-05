import pandas as pd
import numpy as np
import os
import joblib

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
# Proficiency mapping
# -----------------------
def map_score_to_proficiency(score):
    if score >= 90:
        return "Highly Proficient"
    elif score >= 75:
        return "Proficient"
    elif score >= 50:
        return "Nearly Proficient"
    elif score >= 25:
        return "Low Proficient"
    else:
        return "Not Proficient"

# -----------------------
# Shared Split
# -----------------------
def train_test_split_shared(df, test_size=0.2, random_state=42):
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=random_state
    )
    return {"train_idx": train_idx, "test_idx": test_idx}

# -----------------------
# Regression Pipeline
# -----------------------
def regression_model(df, split_data):

    df_model = df.drop(columns=["proficiency"])
    std_id = df_model["studentID"]

    X = df_model.drop(columns=["mps", "studentID"])
    y = df_model["mps"]

    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32"]).columns
    categorical_cols = ["sex"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ]), categorical_cols)
    ])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ])

    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    id_test = std_id.loc[test_idx]

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    results = pd.DataFrame({
        "studentID": id_test,
        "Actual_mps": y_test,
        "Predicted_mps": pred
    })

    results["Error"] = results["Actual_mps"] - results["Predicted_mps"]
    results["Predicted_proficiency_from_score"] = \
        results["Predicted_mps"].apply(map_score_to_proficiency)

    metrics = {
        "R2": round(r2_score(y_test, pred), 2),
        "RMSE": round(root_mean_squared_error(y_test, pred), 2),
        "MAE": round(mean_absolute_error(y_test, pred), 2)
    }

    perm = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=42)

    cat_features = pipe.named_steps["preprocessor"] \
        .named_transformers_["cat"] \
        .named_steps["encoder"] \
        .get_feature_names_out(categorical_cols)

    all_features = np.concatenate([numeric_cols, cat_features])

    feat_importance = pd.DataFrame({
        "Feature": all_features,
        "Importance": perm.importances_mean
    }).sort_values(by="Importance", ascending=False)

    return pipe, X_test, y_test, results.reset_index(drop=True), metrics, feat_importance


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

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ]), categorical_cols)
    ])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
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
        "ROC_AUC": round(
            roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"), 2
        )
    }

    cat_features = pipe.named_steps["preprocessor"] \
        .named_transformers_["cat"] \
        .named_steps["encoder"] \
        .get_feature_names_out(categorical_cols)

    all_features = np.concatenate([numeric_cols, cat_features])

    feat_importance = pd.DataFrame({
        "Feature": all_features,
        "Importance": pipe.named_steps["model"].feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return pipe, X_test, y_test, results, metrics, feat_importance


# -----------------------
# Combine Results
# -----------------------
def combined_results(reg_results, clf_results):
    combined = reg_results.merge(
        clf_results[["studentID", "Predicted_proficiency"]],
        on="studentID",
        how="left"
    )
    combined = combined.rename(
        columns={"Predicted_proficiency": "Predicted_proficiency_classification"}
    )
    combined["Agreement"] = (
        combined["Predicted_proficiency_from_score"]
        == combined["Predicted_proficiency_classification"]
    )
    return combined

# -----------------------
# Export / Load Models
# -----------------------
def export_models(reg_pipe, clf_pipe, path="saved_models"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(reg_pipe, os.path.join(path, "regression_model.joblib"))
    joblib.dump(clf_pipe, os.path.join(path, "classification_model.joblib"))

def load_models(path="saved_models"):
    reg_pipe = joblib.load(os.path.join(path, "regression_model.joblib"))
    clf_pipe = joblib.load(os.path.join(path, "classification_model.joblib"))
    return reg_pipe, clf_pipe