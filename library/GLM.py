import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_global_linear_model(
                                target="test_acc"
                              , features=["learning_rate", "batch_size", "dropout", "optimizer_type"], 
                              csv_path="cleaned_results.csv"
                              ):
    """
    Train a global linear model on the given CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    target : str
        Column name you want to predict.
    features : list[str]
        Columns used to predict the target.

    Returns
    -------
    model : sklearn Pipeline
        The trained model.
    feature_names : list[str]
        The expanded list of feature names (after one-hot encoding).
    coef : list[float]
        Coefficients of the linear model.
    intercept : float
        Intercept of the linear model.
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Split X / y
    X = df[features]
    y = df[target]

    # Identify types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(), categorical_features)
        ]
    )

    # Build full pipeline
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("regressor", LinearRegression())
    ])

    # Train
    model.fit(X, y)

    # Extract feature names
    feature_names = numeric_features

    if categorical_features:
        cat_names = model.named_steps["preprocess"] \
                         .named_transformers_["cat"] \
                         .get_feature_names_out(categorical_features)
        feature_names = feature_names + list(cat_names)

    # Extract coefficients
    coef = list(model.named_steps["regressor"].coef_)
    intercept = model.named_steps["regressor"].intercept_

    return model, feature_names, coef, intercept
