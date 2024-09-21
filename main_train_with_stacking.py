from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from utils.data_processing import *
from utils.feature_selection import *
from utils.display import *


# Function to train stacking model
def stacking_model(X_train, y_train):
    print("Training stacking model...")

    # Define base estimators: RandomForest and XGBoost
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
    ]

    # Define stacking regressor with GradientBoosting as the final estimator
    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    )

    # Train the stacking model
    stacking_reg.fit(X_train, y_train)

    print("Stacking model trained.")

    return stacking_reg


# Function to train and evaluate the model
def train_evaluate_model(X_train, y_train, X_test, y_test, model_type='XGBoost', selected_features=None):
    if model_type == 'XGBoost':
        print("Training XGBoost model...")
        model = XGBRegressor(random_state=42, n_estimators=200, objective='reg:squarederror')
    elif model_type == 'RandomForest':
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == 'GradientBoosting':
        print("Training Gradient Boosting model...")
        model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    elif model_type == 'Stacking':
        print("Training Stacking model...")
        model = stacking_model(X_train, y_train)
    else:
        raise ValueError("Unsupported model type!")

    # Use the selected feature indices for slicing the dataset
    model.fit(X_train[:, selected_features], y_train)

    print("Model trained. Making predictions on the test set...")
    y_pred = model.predict(X_test[:, selected_features])
    # model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")

    return model


# Modified train_evaluate_model function to accept selected features
# Function to train and evaluate the model
def train_evaluate_model_2(X_train, y_train, X_test, y_test, model_type='XGBoost', selected_features=None):
    if model_type == 'XGBoost':
        print("Training XGBoost model...")
        model = XGBRegressor(random_state=42, n_estimators=200, booster='gbtree')
    elif model_type == 'RandomForest':
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == 'GradientBoosting':
        print("Training Gradient Boosting model...")
        model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    elif model_type == 'Stacking':
        print("Training Stacking model...")
        model = stacking_model(X_train, y_train)
    else:
        raise ValueError("Unsupported model type!")

    # Fit the model using selected features if provided
    if selected_features is not None:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")

    return model


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
    df = drop_highly_correlated_features(df, threshold=0.6)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Select best features (change this method according to your preference)
    # 1. Select top k features
    X_train_selected, selected_features, selector = select_k_best_features(X_train_scaled, y_train, k=10)
    # Transform the test set using the same selector
    X_test_selected = selector.transform(X_test_scaled)
    print("X_train_selected shape:", X_train_selected.shape)
    print("X_test_selected shape:", X_test_selected.shape)
    print("Selected features:", selected_features)

    # Train and evaluate models with selected features
    models = ['XGBoost', 'RandomForest', 'GradientBoosting', 'Stacking']
    for model_type in models:
        model = train_evaluate_model_2(X_train_selected, y_train, X_test_selected, y_test, model_type,
                                       selected_features)

    # OR
    # 2. Perform recursive feature elimination
    '''selector, selected_features = feature_selection_rfe(X_train_scaled, y_train)
    # Train and evaluate models with selected features
    models = ['XGBoost', 'RandomForest', 'GradientBoosting', 'Stacking']
    for model_type in models:
        model = train_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, model_type, selected_features)'''

    # Apply polynomial features
    '''X_train_poly, X_test_poly, poly = apply_polynomial_features(X_train_scaled, X_test_scaled, degree=2)

    # Perform recursive feature elimination
    selector, selected_features = feature_selection(X_train_poly, y_train)
    print('selected feature: ', selected_features)

    # Train and evaluate the model
    model = train_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test,  model_type='XGBoost')
    model = train_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test,
                                 model_type='RandomForest')
    model = train_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, model_type='GradientBoosting')
    model = train_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, model_type='Stacking')

    # Plot feature importance
    #plot_feature_importance(model, selected_features)'''


# Entry point
if __name__ == "__main__":
    main()
