#1. Enumerate the (key, value) pair of a Series:
Series.iteritems()


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


# 4. groupby function
# For a Pandas dataframe, we can groupby on 
# 1) Index 
df.groupby(function, axis = 0)
# 2) ColumnName
df.groupby(function, axis = 1)
# 3) Particular Column(s)
df.groupby(['colA', 'colB']).['colC'].count()
df = df.groupby('B')['A'].nunique() 

# We can also apply mulitple functions to multiple columns
df.groupby(['B']).agg({'A': 'nunique'}).reset_index() # dataframe
df.groupby(['A', 'B']).agg({'C': ['nunique', 'sum'], 'D': lambda x: x*4}).reset_index()

# It's useful to check the GroupBy object attributes
df.group(XXX).groups # dictionary object


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
## with Apply function to one column:
df.salary.apply(cut_word) # cut_word is a defined function

## with loc
df.loc[:, 'Revenue_Year'] = df['Revenue_Month'].dt.year


# 7. Describe the columns of a dataframe
df.info() # with column names & types, # of null values, # of rows & columns


# 8. Read in Chinese characters:
#载入文件，因为文件不是utf-8格式的，所以使用gbk的编码读取
laGouDf = pd.read_csv('./DataAnalyst.csv',encoding='gbk')


# 9. Show labels in Chinese characters when using matplotlib
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
laGou_clean.boxplot(column='avgSalary',by= 'city',figsize=(9,7)) 


# 10. Use plotly in Jupyter Notebook
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
# initiate notebook for offline plot
init_notebook_mode(connected=True)


# 11. Create a dataframe from a list
df_dates = pd.DataFrame({'stay_date':datelist})


# 12.
