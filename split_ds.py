import pandas as pd
import numpy as np

df = pd.read_csv('adult_cleaned.csv')

categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = df.select_dtypes(include=['int64','float']).columns.tolist()

target_column = 'income'
if target_column in categorical_columns:
    categorical_columns.remove(target_column)
if target_column in numerical_columns:
    numerical_columns.remove(target_column)

df_categorical = df[categorical_columns + [target_column]]
df_numerical = df[numerical_columns + [target_column]]

df_categorical.to_csv('adult_categorical.csv', index = False)
df_numerical.to_csv('adult_numerical.csv', index = False)

print(f"Categorical features ({len(categorical_columns)}): {categorical_columns}")
print(f"Numerical features ({len(numerical_columns)}): {numerical_columns}")
print("\n Datasets saved as 'categorical_features.csv' and 'numerical_features.csv'")