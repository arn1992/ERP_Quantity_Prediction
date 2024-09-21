import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processing import *
from utils.feature_selection import *
from utils.display import *

# Function to train and evaluate the model
def train_evaluate_model(X_train, y_train, X_test, y_test, selected_features, model_type='XGBoost'):
    if model_type == 'XGBoost':
        print("Training XGBoost model...")
        model = XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.05)
    elif model_type == 'RandomForest':
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == 'GradientBoosting':
        print("Training Gradient Boosting model...")
        model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    else:
        raise ValueError("Unsupported model type!")

    model.fit(X_train[:, selected_features], y_train)

    print("Model trained. Making predictions on the test set...")
    y_pred = model.predict(X_test[:, selected_features])

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    return model


# Main function to execute the full pipeline
def main():
    # Load and inspect data
    df = load_data('./dataset/data.csv')

    # Remove outliers
    df = remove_outliers(df)

    # Encode categorical features
    categorical_columns_onehot = ['Style_No', 'Pack_Qty', 'Product', 'Fabric_Type', 'Order_Category']
    categorical_columns_label = ['Country']
    df = encode_features(df, categorical_columns_onehot, categorical_columns_label)

    # Display Pearson correlation
    df = display_correlation(df, target_column='Quantity')

    # Drop highly correlated features
    df = drop_highly_correlated_features(df, threshold=0.8)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Apply polynomial features
    X_train_poly, X_test_poly, poly = apply_polynomial_features(X_train_scaled, X_test_scaled, degree=2)

    # Perform recursive feature elimination
    selector = feature_selection(X_train_poly, y_train)
    selected_features = selector.support_
    print(selected_features)

    # Train and evaluate models with selected features
    models = ['XGBoost', 'RandomForest', 'GradientBoosting']
    for model_type in models:
        model = train_evaluate_model(X_train_poly, y_train, X_test_poly, y_test, selected_features,
                                     model_type=model_type)

        # Plot feature importance
        plot_feature_importance(model, selected_features)


# Entry point
if __name__ == "__main__":
    main()
