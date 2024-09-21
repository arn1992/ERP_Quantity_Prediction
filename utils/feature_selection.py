from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# Function to scale features
def scale_features(X_train, X_test):
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled.")
    return X_train_scaled, X_test_scaled


# Function to apply polynomial features
def apply_polynomial_features(X_train_scaled, X_test_scaled, degree=2):
    print(f"Applying polynomial features of degree {degree}...")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    print("Polynomial features applied.")
    return X_train_poly, X_test_poly, poly


# Function to apply SelectKBest feature selection
def select_k_best_features(X_train_scaled, y_train, k=10):
    print(f"Selecting top {k} features using SelectKBest...")
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    selected_feature_indices = selector.get_support(indices=True)
    print(f"Top {k} features selected.")
    return X_train_selected, selected_feature_indices, selector


# Function to perform recursive feature elimination (RFECV)
def feature_selection_rfe(X_train_scaled, y_train):
    print("Performing feature selection using RFECV...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    selector = RFECV(estimator=rf, step=1, cv=5)
    selector.fit(X_train_scaled, y_train)
    selected_feature_indices = np.where(selector.support_)[0]  # Get the indices of selected features
    print("Feature selection complete.")
    return selector, selected_feature_indices


# Function to perform recursive feature elimination (RFECV)
def feature_selection(X_train_poly, y_train):
    print("Performing feature selection using RFECV...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    selector = RFECV(estimator=rf, step=1, cv=5)
    selector.fit(X_train_poly, y_train)
    selected_feature_indices = np.where(selector.support_)[0]  # Get the indices of selected features
    print("Feature selection complete.")
    return selector, selected_feature_indices
