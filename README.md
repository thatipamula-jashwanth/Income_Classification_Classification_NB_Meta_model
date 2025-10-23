# Income Classification Using Naive Bayes Meta-Model

## Project Overview
This project predicts whether an individual's income is `<=50K` or `>50K` using the **Census Income dataset**. The dataset contains **mixed types of features** (numerical, categorical) and is **imbalanced**. To handle this, we trained **two Naive Bayes models** separately for numerical and categorical features and then combined them into a **meta-model** using Logistic Regression.  

The meta-model leverages the strengths of both models to improve performance, especially for the minority class (`>50K`).  

---

## Dataset
- **Source:** UCI Census Income Dataset (Adult dataset)
- **Features:**  
  - Numerical: `age`, `fnlwgt`, `education.num`, `hours.per.week`, `capital.loss`  
  - Categorical: `workclass`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country`
- **Target:** `income` (`<=50K` or `>50K`)
- **Rows:** 30,162  
- **Imbalance:**  
  - `<=50K`: 75.1%  
  - `>50K`: 24.9%  

---

## Data Cleaning & Preprocessing
1. Removed rows with missing values and placeholders.  
2. Dropped features with **all zeros**  
3. Split features into **numerical** and **categorical** datasets  
4. Categorical features were **integer encoded** for CategoricalNB.  
5. Target column (`income`) used as-is for both models.  
6. Split data: **Train (80%)**, **Test (20%)**.

---

## Models Used
1. **Gaussian Naive Bayes** → trained on numerical features  
   - Captures patterns from continuous variables
2. **Categorical Naive Bayes** → trained on categorical features  
   - Captures patterns from categorical variables
3. **Meta-model (Logistic Regression)** → trained on **predicted probabilities** from the two base models  
   - Combines both models for better prediction performance  

**Saved models:**  
- `gaussian_nb_model.pkl`  
- `categorical_nb_model.pkl`  
- `meta_model.pkl`  

---

## Evaluation Metrics

### GaussianNB (Numerical Features)
**Train Set**  
- Accuracy: 0.7691  
- Precision: 0.7466  
- Recall: 0.7691  
- F1 Score: 0.7095  

**Test Set**  
- Accuracy: 0.7737  
- Precision: 0.7641  
- Recall: 0.7737  
- F1 Score: 0.7116  

---

### CategoricalNB (Categorical Features)
**Train Set**  
- Accuracy: 0.7891  
- Precision: 0.8174  
- Recall: 0.7891  
- F1 Score: 0.7979  

**Test Set**  
- Accuracy: 0.7970  
- Precision: 0.8229  
- Recall: 0.7970  
- F1 Score: 0.8051  

---

### Meta-Model (GaussianNB + CategoricalNB)
**Train Set**  
- Accuracy: 0.8245  
- Precision: 0.8163  
- Recall: 0.8245  
- F1 Score: 0.8182  

**Test Set**  
- Accuracy: 0.8336  
- Precision: 0.8256  
- Recall: 0.8336  
- F1 Score: 0.8267  
