import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("adult_categorical.csv")

target_col = 'income' 
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

if target_col in categorical_cols:
    categorical_cols.remove(target_col)

encoder = OrdinalEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

print(df[categorical_cols].head())

df.to_csv("cat_encoded_dataset.csv", index=False)
print(f" Encoded dataset saved as 'cat_encoded_dataset.csv'")
