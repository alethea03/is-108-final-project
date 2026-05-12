from preprocessing import load_and_preprocess
from models import train_knn, train_svm, train_ann
from evaluation import evaluate_model

X_train, X_test, y_train, y_test, scaler = load_and_preprocess("data/churn.csv")

knn = train_knn(X_train, y_train)
svm = train_svm(X_train, y_train)
ann = train_ann(X_train, y_train)

knn_results = evaluate_model(knn, X_test, y_test, "KNN")
svm_results = evaluate_model(svm, X_test, y_test, "SVM")
ann_results = evaluate_model(ann, X_test, y_test, "ANN")