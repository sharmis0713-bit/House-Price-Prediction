# ============================================================
# House Price Prediction using Linear Regression
# SyntecHub - Project 1
# ============================================================

# STEP 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

print("All libraries imported successfully!")

# ============================================================
# STEP 2: Load and explore the dataset
# ============================================================
print("\n--- Loading Dataset ---")
housing = fetch_california_housing()

# Convert to a pandas DataFrame (like an Excel table)
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # This is what we want to predict

print(f"Dataset shape: {df.shape}")  # Rows x Columns
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# ============================================================
# STEP 3: Exploratory Data Analysis (EDA) - Visualise the data
# ============================================================
print("\n--- Creating EDA Plots ---")

# Plot 1: Distribution of house prices
plt.figure(figsize=(8, 5))
plt.hist(df['MedHouseVal'], bins=50, color='steelblue', edgecolor='white')
plt.title('Distribution of House Prices')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.show()
print("Saved: price_distribution.png")

# Plot 2: Correlation heatmap - shows which features affect price the most
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("Saved: correlation_heatmap.png")

# ============================================================
# STEP 4: Select features and target variable
# ============================================================
print("\n--- Selecting Features ---")

# X = input features (what the model uses to predict)
# y = target (what we want to predict)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

print(f"Features: {list(X.columns)}")
print(f"Target: MedHouseVal")

# ============================================================
# STEP 5: Split into training and testing sets
# ============================================================
# We train on 80% of data and test on the remaining 20%
# random_state=42 means we get the same split every time we run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ============================================================
# STEP 6: Train the Linear Regression model
# ============================================================
print("\n--- Training Model ---")
model = LinearRegression()
model.fit(X_train, y_train)  # This is where the model "learns"
print("Model trained successfully!")

# ============================================================
# STEP 7: Evaluate the model
# ============================================================
print("\n--- Evaluating Model ---")
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"RMSE  : {rmse:.4f}")
print(f"R²    : {r2:.4f}")
print(f"\nInterpretation:")
print(f"  - RMSE of {rmse:.2f} means predictions are off by ~${rmse*100000:.0f} on average")
print(f"  - R² of {r2:.2f} means the model explains {r2*100:.1f}% of price variation")

# Plot 3: Actual vs Predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()
print("Saved: actual_vs_predicted.png")

# ============================================================
# STEP 8: Interpret coefficients
# ============================================================
print("\n--- Feature Coefficients ---")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
print(coef_df.to_string(index=False))

# ============================================================
# STEP 9: Save the model
# ============================================================
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/house_price_model.pkl')
print("\nModel saved to model/house_price_model.pkl")

# ============================================================
# STEP 10: Example predictions
# ============================================================
print("\n--- Example Predictions ---")
sample = X_test.iloc[:5]
predictions = model.predict(sample)
actual = y_test.iloc[:5].values

print(f"\n{'Sample':<10}{'Predicted ($100k)':<22}{'Actual ($100k)':<18}")
print("-" * 50)
for i, (pred, act) in enumerate(zip(predictions, actual)):
    print(f"{i+1:<10}{pred:<22.2f}{act:<18.2f}")