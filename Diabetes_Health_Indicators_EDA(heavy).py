# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 16:33:28 2025

@author: krish
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from scipy import stats

df = pd.read_csv(r"C:\Users\krish\OneDrive\Documents\Projects\Resume_Projects\Diabetes_Health_Indicators\diabetes_binary_health_indicators_BRFSS2015.csv")

df.shape
df.head()
df.info()
df.describe()

df.dtypes
df.duplicated().sum()

df.drop_duplicates(inplace=True)
df.shape

df.isna().sum()

plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot – Outlier Detection")
plt.show()

df.hist(figsize=(18,12), bins=30)
plt.suptitle("Histograms of All Variables")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(df['BMI'], kde=True)
plt.title("BMI Distribution")
plt.show()

Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['BMI'] = np.where(df['BMI'] > upper, upper,
                     np.where(df['BMI'] < lower, lower, df['BMI']))

plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Scatter Plot 
plt.figure(figsize=(8,5))
plt.scatter(df['Age'], df['BMI'], alpha=0.3)
plt.xlabel("Age")
plt.ylabel("BMI")
plt.title("Age vs BMI Relationship")
plt.show()

# Trend Line Aggregation 
age_bmi = df.groupby('Age')['BMI'].mean().reset_index()

plt.figure(figsize=(8,5))
plt.plot(age_bmi['Age'], age_bmi['BMI'])
plt.xlabel("Age")
plt.ylabel("Average BMI")
plt.title("Average BMI Trend Across Age Groups")
plt.show()

# DownSample [Optional] 
sample_df = df.sample(5000, random_state=42)

plt.figure(figsize=(8,5))
plt.scatter(sample_df['Age'], sample_df['BMI'], alpha=0.4)
plt.xlabel("Age")
plt.ylabel("BMI")
plt.title("Age vs BMI (Sampled)")
plt.show()


df['Diabetes_binary'].value_counts(normalize=True)
sns.countplot(x='Diabetes_binary', data=df)
plt.title("Diabetes Distribution")
plt.show()

df.var()
(df.var() == 0).sum()


stats.probplot(df['BMI'], dist="norm", plot=plt)
plt.title("QQ Plot – BMI")
plt.show()

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

minmax = MinMaxScaler()
df_minmax = pd.DataFrame(minmax.fit_transform(df), columns=df.columns)

robust = RobustScaler()
df_robust = pd.DataFrame(robust.fit_transform(df), columns=df.columns)

print(df.head())

df.to_csv(r"C:\Users\krish\OneDrive\Documents\Projects\Resume_Projects\Diabetes_Health_Indicators\diabetes_binary_health_indicators_BRFSS2015.csv", index=False)

print(df)


# END EDA 

