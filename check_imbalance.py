import pandas as pd 

df = pd.read_csv('adult_cleaned.csv')

target_column = 'income'

class_counts = df[target_column].value_counts()
print('numbers of samples per class :\n',class_counts)

class_percent = df[target_column].value_counts(normalize=True) * 100
print('\npercentage of samples per class :\n', class_percent)

threeshold = 40
imbalanced_classes = class_percent[class_percent < threeshold]
if not imbalanced_classes.empty:
    print("\n Imbalanced classes detected:\n", imbalanced_classes)
else:
    print("\n Classes are reasonably balanced")

