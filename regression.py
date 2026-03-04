import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error


def regression_model():
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

    st.header("Regression Model Metrics", anchor=False)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("R2 metric", value=r2, border=True, help="Range: 0 to 1. \nHigher the better, 1 = perfect prediction")
    with c2:
        st.metric("Root Mean Squared Error", value=rmse, border=True, help="Score difference from actual vs. predicted. Average error emphasizing worst-case mistakes")
    with c3:
        st.metric("Mean Absolute Error", value=mae, border=True, help="Score difference from actual vs. predicted. Average error you can expect normally")
    
    st.markdown("**Regression Model Fitting Data**", unsafe_allow_html=True)
    with st.expander(expanded=False, label="See Prediction Results From Test Data"):
        st.dataframe(results)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(y_test, pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') 
        ax.set_xlabel("Actual mps") 
        ax.set_ylabel("Predicted mps") 
        ax.set_title("Predicted vs Actual") 
        st.pyplot(fig)
