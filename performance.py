import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle


X_num_train = pd.read_csv("X_num_train.csv")
X_num_test  = pd.read_csv("X_num_test.csv")
y_num_train = pd.read_csv("y_num_train.csv").squeeze()
y_num_test  = pd.read_csv("y_num_test.csv").squeeze()

X_cat_train = pd.read_csv("X-cat_train.csv")
X_cat_test  = pd.read_csv("X-cat_test.csv")
y_cat_train = pd.read_csv("y-cat_train.csv").squeeze()
y_cat_test  = pd.read_csv("y-cat_test.csv").squeeze()


with open("gaussian_nb_model.pkl", "rb") as f:
    gnb = pickle.load(f)

with open("categorical_nb_model.pkl", "rb") as f:
    cnb = pickle.load(f)


def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    print(f"\n=== {name} Performance ===")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print("\n-- Train Set --")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Precision:", precision_score(y_train, y_train_pred, average='weighted'))
    print("Recall:", recall_score(y_train, y_train_pred, average='weighted'))
    print("F1 Score:", f1_score(y_train, y_train_pred, average='weighted'))
    
    print("\n-- Test Set --")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_test_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))
    

evaluate_model(gnb, X_num_train, y_num_train, X_num_test, y_num_test, "GaussianNB (Numerical)")

evaluate_model(cnb, X_cat_train, y_cat_train, X_cat_test, y_cat_test, "CategoricalNB (Categorical)")
