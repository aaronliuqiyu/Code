# Temporary #
from helper import stock_change
apartment_change = stock_change(apartment_fs)
apartment_change = pd.DataFrame(apartment_change)
apartment_change = apartment_change.rename(columns = {0:'Usable floor space'}) 
apartment_change

#%%
apartment_change_10 = apartment_change.iloc[-7:].cumsum()
new_index_values = range(2016, 2023)
apartment_change_10.index = new_index_values
apartment_change_1 = apartment_change_10 + apartment_fs.iloc[-7:]
print(apartment_change_1)

#%%
# Using past 7 years new construction to create a dummy new construction dataframe #
apartment_change_2 = apartment_change_1 + apartment_change_10

apartment_change_3 = apartment_change_2 + apartment_change_10

apartment_change_4 = apartment_change_3 + apartment_change_10
print(apartment_change_4)
#test = pd.concat([apartment_change_1, apartment_change_2, apartment_change_3, apartment_change_4], ignore_index=True)
#test
#%%
# Define a new range of index values
new_index_values = range(2023, 2051)
test.index = new_index_values

apartment_fs = pd.concat([apartment_fs, test], axis=0)
print(apartment_fs)
