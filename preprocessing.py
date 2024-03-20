#%%
# Import required packages #
import pandas as pd
import numpy as np

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

# Set path #
path= "C:\\Users\\qiyu\\OneDrive - Chalmers\\Paper 3\\Data"
os.chdir(path)

pd.set_option('mode.chained_assignment', None)

# Load data #
# Read in file with pickle #
pickle_file_path = "C:\\Users\\qiyu\\OneDrive - Chalmers\\Paper 3\\Data\\All_predicted.pkl"
buildings = pd.read_pickle(pickle_file_path)
buildings

drop = buildings['Construction year'] < 1850
buildings = buildings[~drop]

#%%
test = buildings.loc[buildings['Year binned'] == '1920-29']

import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=test, x="Construction year")
plt.show()

#%%
# Compute pmf of year cohort and then apply it to the NaN values #
def year_prob(df):
    df1 = df[df['Construction year'].notna()]
    other = df[df['Construction year'].isna()]
    pmf = pd.DataFrame(df1['Construction year'].value_counts(normalize=True).sort_index().reset_index())
    pmf.columns = ['Construction year', 'Probability']

    num_samples = len(other)
    new_samples = np.random.choice(pmf['Construction year'], size=num_samples, p=pmf['Probability'])
    new_df = pd.DataFrame({'Generated_Values': new_samples})
    other= other.reset_index()
    other['Construction year'] = new_df

    combined = pd.concat([df1, other], ignore_index=True)
    column_to_drop = 'index'
    combined = combined.drop(column_to_drop, axis=1)

    return combined

test = buildings.loc[buildings['Year binned'] == '1910-19']
test1 = year_prob(test)
test1

import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=test1, x="Construction year")
plt.show()
#%%
# Get unique values from the 'Values' column
from sklearn.preprocessing import LabelEncoder

df = buildings
le=LabelEncoder()
df['Year binned'] = le.fit_transform(df['Year binned'])

unique_values = df['Year binned'].unique()

# Initialize an empty list to store DataFrames
dfs = []

# Apply the function using a for loop
for i in unique_values:
    df1 = df.loc[df['Year binned'] == i]
    result_df = year_prob(df1)
    dfs.append(result_df)

# Concatenate all the resulting DataFrames
final_result = pd.concat(dfs, ignore_index=True)

test2 = pd.DataFrame(final_result)
test2