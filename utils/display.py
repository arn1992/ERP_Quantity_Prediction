import numpy as np
import matplotlib.pyplot as plt


# Function to plot feature importance
def plot_feature_importance(model, selected_features):
    print("Plotting feature importance...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances from Model")
    plt.bar(range(len(importances)), importances[indices], color="r", align="center")
    plt.xticks(range(len(importances)), selected_features[indices], rotation=90)
    plt.tight_layout()
    plt.show()
    print("Feature importance plot displayed.")
