# regression.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import numpy as np

def train_regression(df):
    """
    Train a basic regression model on a dataframe and return predictions with evaluation metrics.
    
    Args:
        df (pd.DataFrame): Dataset with 'studentID' as index and 'School MPS' as target.
    
    Returns:
        pd.DataFrame: DataFrame with 'Actual' and 'Predicted' School MPS, indexed by studentID.
    """
    # Ensure studentID is index
    if 'studentID' in df.columns:
        df = df.set_index('studentID')
    
    # Features and target
    X = df.drop(columns=["School MPS"])
    y = df["School MPS"]
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')
    
    # Regression pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluation metrics
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(root_mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    # Return predictions with actual values
    results = pd.DataFrame({
        "Actual School MPS": y_test,
        "Predicted School MPS": predictions
    }, index=X_test.index)
    
    return results, r2, rmse, mae