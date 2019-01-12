# 1. Enumerate the (key, value) pair of a Series:
111 Series.iteritems()


# 2. update column value in pandas with condition

## with a constant value
df.loc[row_index, col_index] = new_value
df.loc[df['Meter_Number'] == '8096662 58-5', 'Meter_Number'] = '8096662-58.5'

## OR with a function
mask = define a subset of the rows of a dataframe 
newColumnValue = a Pandas series of length of mask == True 
temp.loc[mask, 'newColumn'] = newColumnValue.apply(lambda x: function of x)


# 3. reorder a list
mylist = ['a', 'b', 'c', 'd', 'e']
myorder = [3, 2, 0, 1, 4]
mylist = [mylist[i] for i in myorder]


# 4. unique count of A by B
df = df.groupby('B')['A'].nunique() # series
# or
temp = df.groupby(['B']).agg({'A': 'nunique'}).reset_index() # dataframe
temp.columns = ['B', 'count']

# 5. Use plotly in Jupyter Notebook
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
# initiate notebook for offline plot
init_notebook_mode(connected=True)

# 6.  Create a new pandas column
## with Map function to one column:
sac.loc[:, 'over_200k'] = sac['price'].map(lambda x: 0 if x > 200000 else 1)
## with Apply function to the entire dataframe (not a set of columns)
df.apply(lambda x: x['a'] + x['b'] if x['a'] < 100 else x['b'], axis = 1)

## with loc
df.loc[:, 'Revenue_Year'] = df['Revenue_Month'].dt.year

# 7. 
