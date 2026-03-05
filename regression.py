import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import joblib


def regression_model(df):

    tdf = df.copy()

    # -----------------------
    # Features / Target
    # -----------------------
    X = tdf.drop(columns=["School MPS"])
    y = tdf["School MPS"]

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
        ("model", RandomForestRegressor(random_state=42))
    ])

    # -----------------------
    # Train/Test Split
    # -----------------------
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        student_ids,
        test_size=0.2,
        random_state=42
    )

    # -----------------------
    # Train
    # -----------------------
    pipe.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/regression_model.joblib")
    # -----------------------
    # Predict
    # -----------------------
    pred = pipe.predict(X_test)

    # -----------------------
    # Results
    # -----------------------
    results = pd.DataFrame({
        "studentID": id_test,
        "Actual_MPS": y_test,
        "Predicted_MPS": pred
    }).reset_index(drop=True)

    # -----------------------
    # Metrics
    # -----------------------
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(root_mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)

    model = pipe.named_steps["model"]

    # get transformed feature names
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()

    importances = model.feature_importances_

    feat_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    return results, r2, rmse, mae, feat_importance