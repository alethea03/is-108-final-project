from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test, scaler = load_and_preprocess("data/churn.csv")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:\n", y_train.value_counts())

