"""
Machine Learning Pipeline for Calories Burnt Prediction
Compares Linear Regression, Random Forest, and XGBoost Regressor.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def load_and_preprocess(filepath="data/calories_data.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    print(f"[*] Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    # Encode Gender: male -> 0, female -> 1
    if "Gender" in df.columns and df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

    # Separate features and target
    drop_cols = ["User_ID", "Calories"] if "User_ID" in df.columns else ["Calories"]
    X = df.drop(columns=drop_cols)
    y = df["Calories"]

    print(f"[+] Loaded {len(df)} samples with features: {list(X.columns)}")
    return df, X, y

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"  Mean Absolute Error (MAE): {mae:.3f} kcal")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.3f} kcal")
    print(f"  R-squared (R2 Score):      {r2:.4f}")

    return {
        "model": model,
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred
    }

def main():
    os.makedirs("assets", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    df, X, y = load_and_preprocess()

    # Correlation Heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix - Exercise & Calorie Features", fontsize=12, weight="bold")
    plt.tight_layout()
    corr_path = "assets/correlation_matrix.png"
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"[+] Saved correlation heatmap to {corr_path}")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ]

    if HAS_XGBOOST:
        models.append(("XGBoost Regressor", XGBRegressor(n_estimators=200, learning_rate=0.08, random_state=42)))
    else:
        models.append(("Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=150, random_state=42)))

    results = []
    best_result = None

    for name, model in models:
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(res)
        if best_result is None or res["r2"] > best_result["r2"]:
            best_result = res

    # Plot Actual vs. Predicted for best model
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_result["y_pred"], alpha=0.6, color="#2563eb", edgecolors="w", s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Calories Burnt (kcal)", fontsize=11)
    plt.ylabel("Predicted Calories Burnt (kcal)", fontsize=11)
    plt.title(f"Actual vs. Predicted Calories ({best_result['name']} - R2: {best_result['r2']:.4f})", weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    scatter_path = "assets/actual_vs_predicted.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"[+] Saved actual vs predicted scatter plot to {scatter_path}")

    # Save Best Model
    model_path = "model/calorie_model.pkl"
    joblib.dump(best_result["model"], model_path)
    print(f"[+] Best performing model ({best_result['name']}) saved to {model_path}")

if __name__ == "__main__":
    main()
