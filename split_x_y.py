import pandas as pd
from sklearn.model_selection import train_test_split

df_cat = pd.read_csv("cat_encoded_dataset.csv")
df_num = pd.read_csv("cleaned_numerical.csv")


target_col = 'income'

X_cat = df_cat.drop(columns=[target_col])
y_cat = df_cat[target_col]

X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
    X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

print(" Categorical dataset split done.")
print(f"Train shape: {X_cat_train.shape}, Test shape: {X_cat_test.shape}")


X_num = df_num.drop(columns=[target_col])
y_num = df_num[target_col]

X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(
    X_num, y_num, test_size=0.2, random_state=42, stratify=y_num
)

print(" Numerical dataset split done.")
print(f"Train shape: {X_num_train.shape}, Test shape: {X_num_test.shape}")

X_num_train.to_csv("X_num_train.csv", index=False)
X_num_test.to_csv("X_num_test.csv", index=False)
y_num_train.to_csv("y_num_train.csv", index=False)
y_num_test.to_csv("y_num_test.csv", index=False)

X_cat_train.to_csv("X-cat_train.csv",index = False)
X_cat_test.to_csv("X-cat_test.csv",index = False)
y_cat_test.to_csv("y-cat_test.csv",index = False)
y_cat_train.to_csv("y-cat_train.csv",index = False)