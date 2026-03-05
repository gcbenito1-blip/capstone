import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def classification_model(df):

    tdf = df.copy()

    # -----------------------
    # Create target variable
    # -----------------------
    conditions = [
        tdf["School MPS"] == 66.82,
        tdf["School MPS"] == 73.08,
        tdf["School MPS"] == 75.71
    ]

    choices = ["low proficient", "nearly proficient", "proficient"]

    tdf["proficiency"] = np.select(conditions, choices, default="Unknown")

    # -----------------------
    # Features / Target
    # -----------------------
    X = tdf.drop(columns=["School MPS", "proficiency"])
    y = tdf["proficiency"]

    student_ids = X["studentID"]
    X = X.drop(columns=["studentID"])

    # -----------------------
    # Column Types
    # -----------------------
    numeric_cols = X.select_dtypes(include=["number"]).columns

    categorical_cols = [
        "sex",
        "BMI/nutrional status",
        "mother tongue"
    ]

    # -----------------------
    # Pipelines
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

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # -----------------------
    # Train/Test Split
    # -----------------------
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        student_ids,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------
    # Train
    # -----------------------
    pipe.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/classification_model.joblib")
    # -----------------------
    # Predict
    # -----------------------
    pred = pipe.predict(X_test)

    # -----------------------
    # Results
    # -----------------------
    results = pd.DataFrame({
        "studentID": id_test,
        "Actual_proficiency": y_test,
        "Predicted_proficiency": pred
    }).reset_index(drop=True)

    # -----------------------
    # Metrics
    # -----------------------
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="weighted")
    recall = recall_score(y_test, pred, average="weighted")
    f1 = f1_score(y_test, pred, average="weighted")

    proba = pipe.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")


    model = pipe.named_steps["model"]

    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()

    importances = model.feature_importances_

    feat_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    return results, acc, precision, recall, f1, roc_auc, feat_importance