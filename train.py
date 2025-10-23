import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
import pickle


X_num_train = pd.read_csv("X_num_train.csv")
y_num_train = pd.read_csv("y_num_train.csv").squeeze()

X_cat_train = pd.read_csv("X-cat_train.csv")
y_cat_train = pd.read_csv("y-cat_train.csv").squeeze()

gnb = GaussianNB()
gnb.fit(X_num_train, y_num_train)

with open("gaussian_nb_model.pkl", "wb") as f:
    pickle.dump(gnb, f)
print(" GaussianNB model trained and saved as 'gaussian_nb_model.pkl'")

cnb = CategoricalNB()
cnb.fit(X_cat_train, y_cat_train)

with open("categorical_nb_model.pkl", "wb") as f:
    pickle.dump(cnb, f)
print(" CategoricalNB model trained and saved as 'categorical_nb_model.pkl'")
