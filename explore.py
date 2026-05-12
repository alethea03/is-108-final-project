import pandas as pd

# Load dataset
df = pd.read_csv("data/churn.csv")

# Basic info
print("=== SHAPE ===")
print(df.shape)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== COLUMN INFO ===")
print(df.info())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== TARGET DISTRIBUTION ===")
print(df["Churn"].value_counts())