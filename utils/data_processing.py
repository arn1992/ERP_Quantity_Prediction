import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    print("Data loaded. First 10 rows:\n", df.head(10))
    print("\nUnique values per column:\n", df.apply(lambda x: len(x.unique())))
    print("\nMissing values per column:\n", df.isna().sum())
    print(df.info())
    print(df.describe())
    print("Data loading and inspection complete.")
    return df


# Function to remove outliers using the IQR method
def remove_outliers(df):
    print("Removing outliers using the IQR method...")
    numeric_cols = df.select_dtypes(include='number').columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f"Outliers removed. Remaining rows: {df_out.shape[0]} / {df.shape[0]}")
    return df_out


# Function to encode categorical features
def encode_features(df, onehot_columns, label_columns):
    print("Encoding categorical features...")
    encoder = OneHotEncoder(drop='first', sparse=False)
    label_encoder = LabelEncoder()

    # One-hot encoding
    encoded_onehot = encoder.fit_transform(df[onehot_columns])
    encoded_df = pd.DataFrame(encoded_onehot, columns=encoder.get_feature_names_out(onehot_columns))

    # Label encoding
    for col in label_columns:
        df[col] = label_encoder.fit_transform(df[col])

    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1).drop(columns=onehot_columns)
    print("Categorical features encoded.")
    return df


# Function to split data into training and test sets
def split_data(df, target_column='Quantity'):
    print(f"Splitting data into training and test sets, with target column '{target_column}'...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    print("Data split into training and test sets.")
    return X_train, X_test, y_train, y_test

# Function to display Pearson correlation heatmap
def display_correlation(df, target_column):
    print("Calculating Pearson correlation...")
    correlation_matrix = df.corr()

    # Pearson correlation with the target
    print("\nPearson correlation of features with target:\n")
    corr_with_target = correlation_matrix[target_column].sort_values(ascending=False)
    print(corr_with_target)

    # Plot correlation heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f",
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.75})
    plt.title("Correlation Matrix Heatmap", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

    print("Pearson correlation analysis complete.")
    return df


# Function to remove one feature from highly correlated pairs
def drop_highly_correlated_features(df, threshold=0.9):
    print(f"Removing one feature from pairs with Pearson correlation >= {threshold}...")
    correlation_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= threshold)]

    # Drop those features
    df_dropped = df.drop(columns=to_drop)
    print(f"Features dropped due to high correlation: {to_drop}")

    # Plot correlation heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_dropped.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f",
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.75})
    plt.title("Correlation Matrix After Dropping Highly Correlated Features", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

    print("Highly correlated feature removal complete.")
    return df_dropped