# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 18:32:48 2026

@author: krish
"""

# =============================================================
# Diabetes Health Indicators – BRFSS 2015
# Exploratory Data Analysis (EDA) & Data Preprocessing Pipeline
# Data Analytics–First | Extensible to Data Science & ML
# =============================================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Choose ONE dataset depending on analysis goal
# Multiclass target (0,1,2)
df = pd.read_csv(r"C:\Users\krish\OneDrive\Documents\Projects\Resume_Projects\Diabetes_Health_Indicators\diabetes_012_health_indicators_BRFSS2015.csv")

# Binary balanced dataset (optional)
df = pd.read_csv(r"C:\Users\krish\OneDrive\Documents\Projects\Resume_Projects\Diabetes_Health_Indicators\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# Binary imbalanced dataset (optional)
df = pd.read_csv(r"C:\Users\krish\OneDrive\Documents\Projects\Resume_Projects\Diabetes_Health_Indicators\diabetes_binary_health_indicators_BRFSS2015.csv")

print(df.shape)
df.head()


# Data Understanding
print("\nData Info:\n")
df.info()

print("\nMissing Values:\n")
df.isna().sum()

print("\nDuplicate Rows:", df.duplicated().sum())


# 4. Target Variable Distribution 
if 'Diabetes_012' in df.columns:
    target_col = 'Diabetes_012'
elif 'Diabetes_binary' in df.columns:
    target_col = 'Diabetes_binary'

print("\nTarget Distribution:\n")
print(df[target_col].value_counts(normalize=True) * 100)

sns.countplot(x=target_col, data=df)
plt.title('Target Variable Distribution')
plt.show()


# 5. Descriptive Statistics 
print("\nNumerical Summary:\n")
df.describe().T


# 6. Categorical Conversion (0/1 → No/Yes) 
# Identify binary columns (excluding target)
binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != target_col]

# Mapping dictionary
binary_map = {0: 'No', 1: 'Yes'}

# Create a categorical copy for reporting & dashboards
df_cat = df.copy()

for col in binary_cols:
    df_cat[col] = df_cat[col].map(binary_map)

# Optional: Map target variable for interpretation
if target_col == 'Diabetes_binary':
    df_cat[target_col] = df_cat[target_col].map({0: 'No Diabetes', 1: 'Prediabetes/Diabetes'})

elif target_col == 'Diabetes_012':
    df_cat[target_col] = df_cat[target_col].map({
        0: 'No Diabetes',
        1: 'Prediabetes',
        2: 'Diabetes'
    })

# Preview categorical dataset
df_cat.head()


# 7. Univariate Analysis
num_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# 8. Bivariate Analysis (Target vs Features)
for col in num_cols:
    if col != target_col:
        plt.figure(figsize=(6,3))
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f'{col} vs {target_col}')
        plt.show()


# 9. Correlation Analysis
plt.figure(figsize=(14,10))
corr = df[num_cols].corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()


# 10. Outlier Check (IQR Method)
outlier_summary = {}

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
    outlier_summary[col] = outliers

outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outlier_Count'])
outlier_df.sort_values(by='Outlier_Count', ascending=False)


# 11. Feature Scaling (ONLY if Required) 
# NOTE:
# - Scaling is NOT required for tree-based models
# - Required for Logistic Regression, KNN, SVM, PCA

scale_cols = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
scale_cols = [col for col in scale_cols if col in df.columns]

scaler = StandardScaler()
df_scaled = df.copy()

if scale_cols:
    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])


# 12. Final Dataset Split (Ready for ML)
X = df_scaled.drop(columns=[target_col]) # for Data Science 
y = df_scaled[target_col] # for Data Science 

print("\nFinal Feature Matrix Shape:", X.shape) # for Data Science 
print("Final Target Shape:", y.shape) # for Data Science 


# 13. Save Processed Files (Optional)
# Analytics-friendly categorical version
df_cat.to_csv(r"C:\Users\krish\OneDrive\Documents\Projects\Resume_Projects\Diabetes_Health_Indicators\Cleaned_diabetes_categorical.xlsx", index=False)

# ML-ready scaled version
df_scaled.to_csv(r"C:\Users\krish\OneDrive\Documents\Projects\Resume_Projects\Diabetes_Health_Indicators\Cleaned_BRFSS_diabetes_scaled.xlsx", index=False)

# =============================================================
# END OF DATA ANALYTICS PIPELINE
# =============================================================
