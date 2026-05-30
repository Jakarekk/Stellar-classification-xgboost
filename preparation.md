# %% [markdown]
# # Comparison of the gradient boosting models
# XGBoost, LightGBM, CatBoost

# %% [markdown]
# File1 - Data preparation and Division into sets

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Data preparation

# %%
df = pd.read_csv('star_classification.csv')
df.head(10)

# %% [markdown]
# ### Removal of the categorical variables

# %%
df.columns

# %%
df = df.drop(columns=['obj_ID','run_ID','rerun_ID','cam_col','field_ID','spec_obj_ID','plate','MJD','fiber_ID'])

# %%
df.columns

# %% [markdown]
# ### Dealing with  Nans and duplicates 

# %% [markdown]
# In this dataset lack of data or duplicates are errors.

# %%
df.dropna(inplace=True)

# %%
df.drop_duplicates(inplace=True)

# %%
df.info()

# %%
print(df.duplicated().sum())

# %% [markdown]
# ### Check the correction of target

# %%
sns.histplot(df['class'])

# %% [markdown]
# Data is too unbalanced. 
# Records describing galaxies must be reduced.

# %% [markdown]
# Method 1:
# Data removed randomly

# %%
galaxies = df[df['class'] == 'GALAXY']

# %%
galaxies_to_keep = galaxies.sample(n=40000, random_state=19)

# %% [markdown]
# Replace all galaxies from the original dataset with galaxies_to_keep

# %%
df2 = df.drop(df[df['class'] == 'GALAXY'].index)

# %%
df_random_galaxies = pd.concat([galaxies_to_keep, df2])

# %%
sns.histplot(df_random_galaxies['class'])

# %% [markdown]
# Method 2:
# Removal of the most common data in features.
# 
# None of the outliers will be missed.
# 
# Personally preferred.

# %%
features1 = ['u', 'g', 'r', 'i', 'z']
features2 = ['alpha', 'delta']
features3 = ['redshift']

# %%
sns.boxplot(data=df[df['class'] == 'GALAXY'][features1])

# %%
features = features1 + features2 + features3

# %%
mean_val = df[df['class'] == 'GALAXY' ] [features].mean()

# %%
df['distance_from_mean'] = abs(df[df['class'] == 'GALAXY' ][features] - mean_val).sum(axis=1)

# %%
df_common_galaxies = df.sort_values(by =  'distance_from_mean', ascending= False).iloc[19445:]

# %%
sns.histplot(df_common_galaxies['class'])

# %%
df_common_galaxies.drop(columns = ['distance_from_mean'], inplace = True)

# %% [markdown]
# Check if sets have the same size

# %%
df_common_galaxies[df_common_galaxies['class'] == 'GALAXY'].count()

# %%
df_random_galaxies[df_random_galaxies['class'] == 'GALAXY'].count()

# %% [markdown]
# ### Comparison of the data before and after removing galaxies

# %%
sns.boxplot(df[df['class'] == 'GALAXY'][features1])

# %%
sns.boxplot(df_common_galaxies[df_common_galaxies['class'] == 'GALAXY'][features1])

# %% [markdown]
# Removing the most common records did not significantly change the data - the approach can be used

# %%
sns.boxplot(df_random_galaxies[df_random_galaxies['class'] == 'GALAXY'] [features1])

# %% [markdown]
# Both approaches did not change its representativnes

# %% [markdown]
# ### Data validation

# %% [markdown]
# ### Choose type of the data:

# %%
df = df_common_galaxies

# %%
#df = df_random_galaxies

# %% [markdown]
# ### Correction for the STAR  & QSO type:

# %%
sns.boxplot(df[df['class'] == 'STAR'][features1])

# %%
sns.boxplot(df[df['class'] == 'QSO'][features1])

# %%
df.sort_values(ascending= True, by = 'u')

# %% [markdown]
# u, g, z equals -9999 is an error, has to be removed

# %%
df.drop( df[df['u'] < 0].index, inplace = True)

# %%
df.sort_values(ascending= True, by = 'u')

# %%
sns.boxplot(df[df['class'] == 'STAR'][features1])

# %% [markdown]
# Galaxies, Stars and QSO's data is now correct.

# %% [markdown]
# ### Label Encoding

# %%
le = LabelEncoder()
df['target'] = le.fit_transform(df['class'])
df = df.drop('class', axis = 1)

# %%
df.tail(8)

# %% [markdown]
# ### The last check - the dataset readiness

# %%
df.isnull().sum()

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# ### Correlation

# %%
corr = df.corr()
plt.figure(figsize=(10, 8)) 
sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5)

# %% [markdown]
# ### Division into sets

# %%
X = df[['u','g','r','i','z','redshift','alpha','delta']]
Y = df[['target']]

# %%
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 42
)

# %%
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full, Y_train_full, test_size=0.25, random_state=42
)

# %%
print(X.shape)
print(X_test.shape)
print(X_val.shape)
print(X_train.shape)

# %% [markdown]
# ### Export of ready sets

# %%
X_test.to_csv('sets/X_test.csv', index=False)
Y_test.to_csv('sets/Y_test.csv', index=False)

# %%
X_val.to_csv('sets/X_val.csv', index=False)
Y_val.to_csv('sets/Y_val.csv', index=False)

# %%
X_train.to_csv('sets/X_train.csv', index=False)
Y_train.to_csv('sets/Y_train.csv', index=False)


