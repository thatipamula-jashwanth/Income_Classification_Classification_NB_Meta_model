import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
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

gnb_train_pred = gnb.predict_proba(X_num_train)
cnb_train_pred = cnb.predict_proba(X_cat_train)

import numpy as np
meta_X_train = np.hstack((gnb_train_pred, cnb_train_pred))
gnb_test_pred = gnb.predict_proba(X_num_test)
cnb_test_pred = cnb.predict_proba(X_cat_test)
meta_X_test = np.hstack((gnb_test_pred, cnb_test_pred))


meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(meta_X_train, y_num_train)  


with open("meta_model.pkl", "wb") as f:
    pickle.dump(meta_model, f)

print(" Meta-model trained and saved as 'meta_model.pkl'")
