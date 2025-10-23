import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle

X_num_train = pd.read_csv("X_num_train.csv")
X_num_test  = pd.read_csv("X_num_test.csv")
y_train = pd.read_csv("y_num_train.csv").squeeze()
y_test  = pd.read_csv("y_num_test.csv").squeeze()

X_cat_train = pd.read_csv("X-cat_train.csv")
X_cat_test  = pd.read_csv("X-cat_test.csv")

with open("gaussian_nb_model.pkl", "rb") as f:
    gnb = pickle.load(f)

with open("categorical_nb_model.pkl", "rb") as f:
    cnb = pickle.load(f)

with open("meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

gnb_train_pred = gnb.predict_proba(X_num_train)
cnb_train_pred = cnb.predict_proba(X_cat_train)
meta_X_train = np.hstack((gnb_train_pred, cnb_train_pred))

gnb_test_pred = gnb.predict_proba(X_num_test)
cnb_test_pred = cnb.predict_proba(X_cat_test)
meta_X_test = np.hstack((gnb_test_pred, cnb_test_pred))

y_train_pred = meta_model.predict(meta_X_train)
y_test_pred  = meta_model.predict(meta_X_test)

def evaluate_model(y_true, y_pred, dataset_name):
    print(f"\n=== Meta-Model Performance: {dataset_name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate_model(y_train, y_train_pred, "Train Set")
evaluate_model(y_test, y_test_pred, "Test Set")
