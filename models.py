from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model
def train_svm(X_train, y_train):
    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_ann(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu',max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model