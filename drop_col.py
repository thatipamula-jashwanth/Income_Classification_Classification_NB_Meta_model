import pandas as pd

df = pd.read_csv("adult_numerical.csv")

threshold = 0.9

mostly_zero_cols = [col for col in df.columns if (df[col] == 0).mean() >= threshold]

df.drop(columns=mostly_zero_cols, inplace=True)

print(f" Dropped columns with >=90% zeros: {mostly_zero_cols}")

df.to_csv("cleaned_numerical", index=False)
print(" Cleaned dataset saved as 'cleaned_numerical.csv")
