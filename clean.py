import pandas as pd
import numpy as np 

df = pd.read_csv('adult.csv')

print(f"original shape: {df.shape}")

placeholders = ['?', 'NA', 'N/A', 'na', 'n/a', 'unknown', 'Unknown','-','-','']

df.replace(placeholders,np.nan, inplace=True)

print("\n missing values before dropping:")
print(df.isnull().sum())

df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)

print(f"\nshape after dropping: {df.shape}")
print("\n missing values after dropping:")
print(df.isnull().sum())

df.to_csv('adult_cleaned.csv', index=False)
print('\ncleaned data saved to adult_cleaned.csv')
