### Upgrade all PIP packages
pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

# delete variables
reset_selective var_name
# check variables
dir(), globals(), locals()


# Remove an element in a list by value:
listA = [x for x in listA if x!= valueToRemove]
listA.remove(valueToRemove) # this way will only delete the first occurance of the value


# Convert unicode to string    
x.encode('utf-8')


# read and write to text files
filename = 'review' + str(y) + '.txt'
text_file = open(filename, "rw")
lines = text_file.readlines()
text_file.write('input something')
text_file.close()


# Sample from a Data Frame
df.sample(n, axis = 1)


# Read a dataset into a DataFrame
columnNames = ['id', 'name', 'sales', 'salary'] 
df = pd.read_csv('./../../assets/datasets/salary.dat', delim_whitespace = True, names = columnNames)

df = pd.read_csv('./../../assets/datasets/salary.csv')

df = pd.read_csv('url of csv file')

# Export a DataFrame into csv file
DataFrame.to_csv('greenness.csv', sep='\t', encoding='utf-8', na_values = "?")





# EDA using Pandas
# Describe/Check a DataFrame
df.describe()
df.plot()

(df.count()/(df.count().max())).sort_values(ascending = True)
df.dtypes.sort_values()
df.dropna().info()
len(df.dropna())/float(len(df))

# Describe/Check a DataSeries
df.col.value_counts()
df.col.value_counts()/len(df)
np.any(np.isnan(df.col))



# Plotting for EDA:
1. 
g = sns.FacetGrid(data, row="contact", col="y", margin_titles=True)
g.map(sns.distplot, "age")

2. 
sns.kdeplot(data.query("y == 'no'").age, shade=True, alpha=.2, label='No', color='salmon')
sns.kdeplot(data.query("y == 'yes'").age, shade=True, alpha=.2, label='Yes',color='dodgerblue')
plt.show()

3. 
sns.factorplot(
    x='SchoolHoliday',
    y='Sales',
    data=store1_data, 
    kind='box'
)

4. # violin plot
ax = sns.violinplot(x="day", y="total_bill", data=tips,
                     inner=None, color=".8")
ax = sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)



# DataFrame Concatenation
pd.concat((df1, df2), axis=1)
DF_A.append(DF_B)

# Sort a DF by value in a column
df.sort_values('Date', inplace=True)
# Set DF index to be a column
df.set_index('Date', inplace=True)

# change a field from index to a column
df.reset_index(level = 0, inplace = True)

# sort a DF by index
DF.sort_index()

# Drop the null values from a DataFrame
DF.dropna(inplace = True)


# Find the Pearson Correlation between different columns in a DataFrame:
DataFrame.corr()

# Enumerate the (key, value) pair of a Series:
Series.iteritems()

# Filter on a String
Series.str.contains("Ormond Wychwood")

# Find the position of a character in a string
string.find("/", startPosition, endPosition)
# Find the last occurance of a char in a string
string.rfind("/", startPosition, endPosition)

# Find the last occurance of a char in a string

# Create dummy variables
categories = ["sx", "rk", "dg"]
for category in categories:
    series = df[category]
    dummies = pd.get_dummies(series, prefix=category)
    df = pd.concat([df, dummies], axis=1)
print df.columns

# String Contains Function can filter on multiple items once
subset = ['VEHICLE THEFT','BURGLARY','DRUG/NARCOTIC']
sf_crime['Category'].str.contains('|'.join(subset))

# Return a flatterned array
arrayName.ravel()

#Convert DataFrame to Numpy Array
data = data_df.as_matrix(columns=None)


# Create a new pandas column 
## with Map function:
sac.loc[:, 'over_200k'] = sac['price'].map(lambda x: 0 if x > 200000 else 1)
## with Apply function:
df.apply(lambda x: x['a'] + x['b'] if x['a'] < 100 else x['b'], axis = 1) 

## with loc
df.loc[:, 'Revenue_Year'] = df['Revenue_Month'].dt.year

# Cartesian Product of two lists:
['{}-{}'.format(x, y) for x in attributes for y in dimensions]


# Pandas Pivot Table syntax
1. pd.pivot_table(df,index=["Manager","Rep"],values=["Price","Quantity"],
       columns=["Product"],aggfunc=[np.sum],fill_value=0)
# columns is optional
2. df.groupby('B').aggregate({'D':np.sum, 'E':np.mean})
3. table = pd.pivot_table(df,index=["Manager","Status"],columns=["Product"],values=["Quantity","Price"],
               aggfunc={"Quantity":len,"Price":[np.sum,np.mean]},fill_value=0)

# Random Number Generation:
# Using Random_State to generate a list of random numbers with reproducibility
from numpy.random import RandomState
prng = RandomState(1234)
ran = prng.randint(1, 1000001, size=100000)

# set a seed for reproducibility
np.random.seed(12345)
# random integer
np.random.randn(10, 4)

# create a Series of booleans in which roughly half are True
nums = np.random.rand(len(data))
mask_large = nums > 0.5

# DateTime and Calendar import calendar
from datetime import date
my_date = date.today()
print calendar.day_name[my_date.weekday()]

from datetime import datetime
from datetime import timedelta
lesson_date = datetime(2016, 3, 5, 23, 31, 1, 844089)

# Convert string to DateTime
datetime.datetime.strptime(x,'%Y-%m-%d')

# DateTime index
# you can filter on year, year-month etc. but not month-only
store1_data['2015']

# Convert a series to Datetime
dataDF.reviewDate = dataDF.reviewDate.map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
OR
data['Date'] = pd.to_datetime(data['Date'])

# Pandas Plot Setup
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
%pylab inline

# MatplotLib Setup
# import matplotlib as mpl
import matplotlib.pyplot as plt
# Setup matplotlib to display in notebook:
%matplotlib inline

# MatplotLib Plot formatting examples
1. 
plt.legend(fontsize=10)
plt.xticks(fontsize = 20)
plt.title('% of Evil Deeds by Alignment', fontsize = 20)
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
ax.legend(loc = "lower right", fontsize = 14)
ax.set_ylabel("r^2 value", fontsize = 20)
ax.set_xlabel('degree of polynomial', fontsize = 20)
2. 
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from RM")
plt.ylabel("Actual Values MEDV")
plt.show()
# 3. Add a dashed line to indicate the relationship between measured and predicted y's
fig, ax = plt.ubplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# 4. 
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(coeffs)
fig.colorbar(cax)
ax.set_xticklabels(['']+list(coeffs.columns), rotation=45)
ax.set_yticklabels(['']+list(coeffs.index)) 

# Pandas Plot formatting examples
color = dict(boxes='DarkOrange', whiskers='DarkOrange', medians='DarkBlue', caps='Black')
DF.plot(kind='box', color=color, sym='r+',  subplots=True, figsize=(16, 4), sharey = True)
DF.T.plot(kind = 'bar', figsize=(12, 6), rot=0 )  # set the x label horizontal

# Seaborn Plot formatting examples
import seaborn as sb
%matplotlib inline 

sb.heatmap(feature_correlation_matrix)

a = sb.factorplot(x = 'alchemy_category', y = 'label', data = data,  size = 12, kind = 'bar')
a.set_xticklabels(rotation=30)

# Set Operation
# 1. difference of sets
list(set(list1) - set(list2))

# 2. intersection of sets
set(list1).intersection(list2)

# 3. union of sets
list(set(list1) + set(list2))





# Time Series
Series.resample()
Series.asfreq()
resample can take funcs as argements

# Filter on Time Series Index
data.loc[data.index < '2014-01-01 00:00:00']


# Sort by the Value in the (Key, Value) pair in a dictionary
Import Operator
sorted(DictName.iteritems(), key=operator.itemgetter(1))
max(DictName.iteritems(), key=operator.itemgetter(1))

# Indexing/Slicing Pandas DataFrame
'ix' is the most general indexer
df.ix[ indexNameArray, colNameArray ]
df.ix[rowNumberArray, colNumberArray]
df.ix[ rowNumberRange, colNumberRange ]

if indexName is a number, df.ix[] will interpret it as a rowNumber
You can use:
	df.ix[ df.index.isin(indexNameArray), [colNameArray/colNumberArray]  ]
or  df.ix[indexMask,  [colNameArray/colNumberArray]  ]
or  df.loc[indexNameArray, [colNameArray/colNumberArray] ]

df.iloc [ rowNumberArray, colNumberArray ]
df.iloc [ rowNumberRange, colNumberRange ]

df.loc [ indexNameArray, colNameArray]

df[indexMask], df.ix[indexMask, colMask]
df[colMask] is not correct

df[numberRange] selects rows
df[numberArray] selects columns based on the column numbers
df[colNameArray]

# Web Scraping from HTML:
import urllib2
import json
from bs4 import BeautifulSoup
r = requests.get(url, auth=('user', 'pass'))
soup = BeautifulSoup(r.content)
OR:
soup = BeautifulSoup(urllib2.urlopen('http://www.omdbapi.com/?y=&plot=short&r=json&t=').read(), "lxml")
tables = soup.find_all('table', width = "98%")
for idx, table in enumerate(tables):
    for idx2, row in enumerate(table.find_all('tr')):
	if idx2 > 0:
	    for idx3, column in enumerate(row.find_all(['td','th'])):
		if idx3 == 0:
		    functions on the column to pull the web contents
		    column.text
		    column.find('a').content[0]['src']
		    column.find_all('span', class = "sorttext")

# Web Scraping from JSON:
raw =[]
soup = BeautifulSoup(urllib2.urlopen('http://www.omdbapi.com/?y=&plot=short&r=json&t='+bond_movie).read(), "lxml")
parsed_json = json.loads(soup.text) # JSON.LOADS returns a dictionary which you can add or append to a list
raw += [parsed_json] # or raw.append(parsed_json)
df = pd.DataFrame(raw)



# Error Metrics:
from sklearn.metrics import mean_squared_error, mean_absolute_error
print "RMSE:", mean_squared_error(ys, predictions)
print "MAE:", mean_absolute_error(ys, predictions)

## Linear Regression Models:
# SK-Learn Linear Regression with RMSE (OLS method):
from sklearn import linear_model
lm = linear_model.LinearRegression() # this function returns a linear model, each dependent variable is at degree 1
X = df[["x0", "x1", "x2"]]
y = df["y"]
model = lm.fit(X, y)
predictions = lm.predict(X)
print "r^2:", model.score(X,y)  
print "Coefficients:", model.coef_, model.intercept_


# Stats Model LAE (Actually a Quantile Regression Model):
# The LAD model is a special case of quantile regression where q=0.5
import statsmodels.formula.api as smf
mod = smf.quantreg('y ~ x', df) 
# 'y~x0+x1+x2' if you have multiple variables
res = mod.fit(q=0.5) 
print(res.summary())

# Stats Model OLS (RMSE method):
import statsmodels.api as sm
X = np.array(xs).transpose()
X = sm.add_constant(X)
mod = sm.OLS(ys, X)
res = mod.fit()
predictions = res.predict(X)
print res.summary()

# NumPy Model for linear regression with Polynomials: 
coef = np.polyfit(xs, ys, deg=2) # xs is the independent variable, ys is the dependent variable
predictions = np.polyval(coef, xs)
plt.scatter(xs, ys) plt.plot(xs, predictions)
from sklearn import metrics
metrics.r2_score(ys, predictions) # calcuate R^2 value using SK-Learn


# Two functions to calculate the R^2 score
from sklearn import metrics
lm = linear_model.LinearRegression()
predictions = lm.predict(X)
lm.metrics.r2_scores(y, predictions)

from sklearn import metrics
lm = linear_model.LinearRegression()
model = lm.fit(X, y)
lm.score(X, y) or model.score(X,y)



# Split of Train and Test data
1. 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
def getTestScore(percent):
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = percent)
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
return model.score(X_test, y_test) 

2.
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(Y, 1, test_size=0.33, random_state=0)


# Cross Validation 
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
lm = linear_model.LinearRegression()  # initialize a model
scores = cross_val_score(lm, df, y, cv=5)  
# the first paremeter tells SK-Learn what type of model to cross validate, it doesn't return the fitted model though
# the df must be a 2-D array or a DataFrame, not a Data Series. You can reshape to (len(Series), 1) before applying
print "Cross-validated scores:", scores
predictions = cross_val_predict(lm, X, y, cv=5)  # X, y cannot be Series, need to be 2-D arrays
plt.scatter(y, predictions)
accuracy = metrics.r2_score(y, predictions)
print "Cross-Predicted Accuracy:", accuracy

# To Save/Serialize your Cross-Validated model
lm = linear_model.LinearRegression()
scores = cross_validation.cross_val_score(lm, X, y, cv=5)
# To fit your estimator
lm.fit(X, y)
# To serialize
import pickle
with open('our_estimator.pkl', 'wb') as fid:
    pickle.dump(lm, fid)
# To deserialize estimator later
with open('our_estimator.pkl', 'rb') as fid:
    lm = pickle.load(fid)


# Ridge-Regularization with built-in Cross-Validation
1. 
rlmcv = linear_model.RidgeCV()
# Fit the polynomial again with ridge regularization
X = np.vander(xs, 4)
y = ys
ridge_model = rlmcv.fit(X, y)
predictions = ridge_model.predict(X)
print "r^2:", ridge_model.score(X, ys)
print "alpha:", rlmcv.alpha_

2.
lm = linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1))   #  the returned lm model will be the model with the best alpha
model = lm.fit(X_train_vander, y_train)




# Loading data from SciKit-Learn package
from sklearn import datasets
boston = datasets.load_boston()
print boston.DESCR
df = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["MEDV"])

# SK-Learn normalization
from sklearn import preprocessing
xs = df["NOX"]
ys = df["TAX"]
xs = preprocessing.normalize(xs, norm='l1')
ys = preprocessing.normalize(ys, norm='l1')
plt.scatter(xs, ys, color='r')
plt.xlabel("NOX L1 Normalized")
plt.ylabel("TAX L1 Normalized")
plt.show()

# SK-Learn Min-Max Scaling
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
xs = scaler.fit_transform(df[["NOX"]])
ys = scaler.fit_transform(df[["TAX"]])

# SK-Learn Standardization
from sklearn import preprocessing
xs = preprocessing.scale(df["NOX"])
ys = preprocessing.scale(df["TAX"])

# Try-Catch:
try:
	pass
except Exception as e:
	print e


# KNN syntax 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
neigh = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2, metric='minkowski')
neigh.fit(data, target)
neigh.score(data, target)
test = data[50:60]
predict = neigh.predict(test) # model prediction
actual = list(target[50:60]) # actuals
print predict
print actual


# Empty String, List, Dictionary are like "False" when you use "If Else" statement:
	if not num_string: return 0
	else: return locale.atof(num_string)

# Replace Strange Characters with Empty String
	import re
	stringName = re.sub("[^0-9]", "", stringName)

# Convert String into Number:
	import locale
	locale.setlocale(locale.LC_NUMERIC, '')
	SeriesName = SeriesName.apply(locale.atof)

# Regular Expression

import re
m = re.search('...', stringy)
print m.group()

m = re.findall('\s+(\wo\w*)', data)
print m


# Find functions
'in' -- python
'isin' -- pandas
'$in' -- MongoDB

# Convert MongoDB data into a Pandas DataFrame
from pymongo import MongoClient
client = MongoClient("mongodb://guest:abc123@ds063946.mlab.com:63946/class_sample")
db = client.class_sample

import pandas as pd
movies = db.movies
movies_df = pd.DataFrame([x for x in movies.find({})])
movies_df.columns = ['year','movie','role','id']


## Classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    y_pred = gs_logreg.predict(X_test)
    y_score = gs_logreg.decision_function(X_test)
    y_proba_score = gs_logreg.predict_proba(X_test)
    
    print "Classification Report: ", (classification_report(y_test, y_pred)), "\n"
        
    cm = confusion_matrix(y_test, y_pred)
    idx = ['Negative', 'Positive']
    col = ['Predicted Negative', 'Predicted Positive']
    print pd.DataFrame(cm, index=idx, columns=col), "\n"

    FPR, TPR, thresholds = roc_curve(y_test, y_proba_score[:,1])
    ROC_AUC = auc(FPR, TPR)
    accuracy_score(y_test, y_pred)
    
    print "AUC is:", ROC_AUC, "\n"
    print "Accuracy Score is: ", accuracy_score(y_test, y_pred), "\n"

    # Plot of a ROC curve for class 1 (positive review)
    plt.figure(figsize=[7,7])
    plt.plot(FPR, TPR, label='ROC curve (area = %0.4f)' % ROC_AUC, linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC for Evergreen Website Prediction', fontsize=18)
    plt.legend(loc="lower right")
    plt.show()

modelFunc(df,y)





# Logistic Regression with CV and Regularization syntax: 
sss = StratifiedShuffleSplit(Y, 1, test_size=0.33, random_state=0)
for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
# fit model with five folds and lasso regularization
# use Cs=15 to test a grid of 15 distinct parameters
# remeber: Cs describes the inverse of regularization strength
logreg_cv = LogisticRegressionCV(Cs=15, cv=5, penalty='l1', scoring='accuracy', solver='liblinear')
logreg_cv.fit(X_train, Y_train)
# find best C per class
best_C = {logreg_cv.classes_[i]:x for i, (x, c) in enumerate(zip(logreg_cv.C_, logreg_cv.classes_))}
# fit regular logit model to 'DRUG/NARCOTIC' and 'BURGLARY' classes
# use lasso penalty
logreg_1 = LogisticRegression(C=best_C['DRUG/NARCOTIC'], penalty='l1', solver='liblinear')
logreg_2 = LogisticRegression(C=best_C['BURGLARY'], penalty='l1', solver='liblinear')
logreg_1.fit(X_train, Y_train)
logreg_2.fit(X_train, Y_train)
# build confusion matrices for the models above
Y_1_pred = logreg_1.predict(X_test)
conmat_1 = confusion_matrix(Y_test, Y_1_pred, labels=logreg_1.classes_)
conmat_1 = pd.DataFrame(conmat_1, columns=logreg_1.classes_, index=logreg_1.classes_)



# Decision Tree syntax
# regressor:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# classifier:
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
classtree = DecisionTreeClassifier(max_depth=3)
classtree.fit(X_train,y_train)
y_pred = classtree.predict(X_test)
y_proba_score = classtree.predict_proba(X_test)
accuracy_score(y_test, y_pred)

# Graph the Decision Tree
dot_data = export_graphviz(classtree, out_file=None, 
                         feature_names=X.columns,  
                         class_names='Label',  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 



# Hierarchical Clustering
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

X = df.as_matrix(columns=None)
Z = linkage(X, 'ward')
c, coph_dists = cophenet(Z, pdist(X))

plt.title('Truncated Dendrogram')
plt.xlabel('Index Numbers')
plt.ylabel('Distance')
dendrogram(
    Z,
    truncate_mode='lastp',  
    p=15,    ### now 15 clutster
    show_leaf_counts=False,  
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  
)
plt.show()

max_d = 15
clusters = fcluster(Z, max_d, criterion='distance')
clusters

plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')
plt.show()





# Create Dummay Variable Matrix (Design Matrix) for Categorical Variables using Pastry:
X = patsy.dmatrix('~ C(hour) + C(DayOfWeek) + C(PdDistrict)', DataFrame)
y = DataFrame['Category'].values





## Plotting ROC function

# generic curve plotting function
def auc_plotting_function(rate1, rate2, rate1_name, rate2_name, curve_name):
    AUC = auc(rate1, rate2)
    # Plot of a ROC curve for class 1 (has_cancer)
    plt.figure(figsize=[11,9])
    plt.plot(rate1, rate2, label=curve_name + ' (area = %0.2f)' % AUC, linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(rate1_name, fontsize=18)
    plt.ylabel(rate2_name, fontsize=18)
    plt.title(curve_name + ' for house price > 200,000', fontsize=18)
    plt.legend(loc="lower right")
    plt.show()

# plot receiving operator characteristic curve
def plot_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_plotting_function(fpr, tpr, 'False Positive Rate', 'True Positive Rate', 'ROC')

# Database connection
1. 
import sqlite3
sqlite_db = 'test_db2.sqlite'
conn = sqlite3.connect(sqlite_db)
c = conn.cursor()
result = c.execute('you sql command')
fetched = results.fetchall()
    for x in fetched:
        print x

2.
from sqlalchemy import create_engine
import psycopg2
# an rds database on the AWS, east coast
connect_param = 'postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com:5432/titanic'
engine_titanic = create_engine(connect_param)

	query = pd.read_sql("SELECT * FROM pg_catalog.pg_tables WHERE schemaname='public'", con=engine_titanic)
	print query

	conn = engine_titanic.connect()
	result = conn.execute('SELECT "Cabin", sum("Age") as total_age FROM train group by "Cabin" HAVING sum("Age") > 75 ')
	for x in result.fetchall():
	    print x


sns.facetgrid()
sns.kdeplot()


# Multi-threading using Futures package
# 1. 
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # executor is a pool of threads
     future_to_url = {executor.submit(load_url, url): url for url in URLS}    # dictionary of futures
     for future in concurrent.futures.as_completed(future_to_url):
	try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))

# 2.  Why is there no With statement here? 
pool = ThreadPoolExecutor(5)
futures = []
for x in range(5):
    futures.append(pool.submit(functionName, arg1, arg2, ..., argn))
for x in as_completed(futures):
    print(x.result())

3. 
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(functionName, range(10))  
    # this function returns an iterator whose content can only be viewed by a loop, like a file 
    for x in results:
        print x

# Retrieve data from newsgroup20
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=('headers', 'footers', 'quotes'))

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=('headers', 'footers', 'quotes'))




##### Fei's Pipeline

## Import packages
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV

## Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Feature Scaling
X = df.drop('acceptability', axis=1)
le = LabelEncoder()
le.fit(df['acceptability'])
y = le.transform(df['acceptability'])
names = le.inverse_transform([0,1,2,3])

## Cross Validation
def do_cv(model, X, y, cv):
    scores = cross_val_score(estimator=model,
                         X=X,
                         y=y,
                         cv=cv,
                         n_jobs=-1)
    print "Model:\n", model, "\n"
    print'CV accuracy for model: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))

## Classification Report & Confusion Matrix
def do_cm_cr(model, X, y, names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print "Classification Report: \n", (classification_report(y_test, y_pred)), "\n"

    confmat = confusion_matrix(y_test, y_pred)
    idx = names
    col = ["Predited " + x for x in names]
    print "Confusion Matrix: \n"
    print pd.DataFrame(confmat, index=idx, columns=col), "\n"

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    plt.tight_layout()
    plt.show()

def do_grid_search(model, X, y, parameters):
    gs = GridSearchCV(estimator=model, 
                      param_grid=parameters, 
                      scoring='accuracy', 
                      cv=5,
                      n_jobs=-1)
    gs = gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_params_)










# Hierarchical Clustering
# pass the triangular form (NOT the Square Form) of distance mastrix to hierarchy.linkage()
# if n data points, tri-matrix is a C(n,2) vector and linkage returns a (n-1)*4 matrix

# 1. Given data X is the distance matrix of data points
from scipy.spatial import distance as ssd
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
# X is the distance matrix
# ssd.squareform(X) is the upper triangular matrix of X

Z = hierarchy.linkage(ssd.squareform(X), method="complete")
# Z is the summary matrix of the clustering result: in each cluster - furthest two points and the no. of elements

fig, ax = plt.subplots(figsize=(12,8))
dn = hierarchy.dendrogram(Z, labels=X.index)




# 2. df is a m*n matrix that contains m points, each is a n-dimensional point 
from scipy.spatial.distance import pdist, squareform

# pdist(df, metric='euclidean') is the upper triangular matrix of row_dist
# row_clusters is the summary matrix of the clutering result
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1) 
                    for i in range(row_clusters.shape[0])])

from scipy.cluster.hierarchy import dendrogram
# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters, 
                       labels=labels,
                       # make dendrogram black (part 2/2)
                       # color_threshold=np.inf
                       )
plt.tight_layout()
plt.ylabel('Euclidean distance')
#plt.savefig('./figures/dendrogram.png', dpi=300, 
#            bbox_inches='tight')
plt.show()


# Transformation between Triangular & Square forms of Distance Matrix, use the distance.squareform() function in both cases!!!!

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels,
                        index=labels)

ssd.squareform(X)




### K-means clustering

# scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# elbow plot to find optimal K
from sklearn.cluster import KMeans
distortions = []
for i in range(1, 20):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                n_jobs=-1,
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 20), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# update K-means model with optimal K
km = KMeans(n_clusters=7, 
            init='k-means++', 
            n_init=10, 
            n_jobs=-1,
            max_iter=300, 
            random_state=0)
y_km = km.fit_predict(X)


# get centroids and labels
labels = km.labels_
centroids = km.cluster_centers_

# get silhouette scores for all data
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

# plot silhouette scores by cluster
from matplotlib import cm
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
# plt.savefig('./figures/silhouette.png', dpi=300)
plt.show()





# use plotly in Jupyter Notebook
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *

# initiate notebook for offline plot
init_notebook_mode(connected=True)        

