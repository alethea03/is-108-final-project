from preprocessing import load_and_preprocess
from models import train_knn, train_svm, train_ann

X_train, X_test, y_train, y_test, scaler = load_and_preprocess("data/churn.csv")

knn = train_knn(X_train, y_train)
print("KNN trained ✓")

svm = train_svm(X_train, y_train)
print("SVM trained ✓")

ann = train_ann(X_train, y_train)
print("ANN trained ✓")

print("\nAll models ready!")