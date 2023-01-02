# Pandas

- python open-source library
- excel for python
    - excel is not as powerful
- can handle large datasets, multiple files, pulling data from APIs, & is python-capable

```python
import pandas as pd

a_dataframe = pd.read_csv('filepath.csv')
# csv: comma-seperated values
# reads from flat text file

DataFrame.to_sql("data", engine, chunksize=1000)
# writes to SQL database
```

### Series

- one column from a dataframe

```python
s = pd.**Series**({'id1': 1, 'id2': 2, 'id3': 'three'})
# set index of numpy (first column) with a dict

s = pd.Series(data=[1,2,'three'], **index**=['id1','id2','id3'])
# set index with two lists
```

### DataFrame

- dictionary of a series

```python
pd.DataFrame.from_records()
# constructor from tuples, record arrays, list of dicts

pd.DataFrame**.from_dict**()
# from sicts of Series, arrays, or dicts

pd.DataFrame**.read_csv**(data, args)

pd.DataFrame**.read_table**()

pd.DataFrame.read_clipboard()
```

```python
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

df = pd.DataFrame(data=d, dtype=np.int8)

d = {'col1': [0, 1, 2, 3], 'col2': pd.Series([2, 3], index=[2, 3])}

pd.DataFrame(data=d, index=[0, 1, 2, 3])

df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                  columns=['a', 'b', 'c'])
```

```python
df.columns =['Col_1', 'Col_2']
df.index = ['Row_1', 'Row_2', 'Row_3', 'Row_4']

df = df.**rename**(columns = {"Col_1":"Mod_col"})
df = df.rename({"Mod_col":"Col_1","B":"Col_2"}, axis='columns')

df.columns.**values**[1] = 'Student_Age'

columns = ['First Name', 'Age']
df = pd.read_csv('out.csv', **header** = None, **names** = columns)
# automatically takes first column and row for the index and column headers

df = df.rename(index = {...})
```

### Functions & Cleaning

```python
import pandas as pd

listings_df = pd.**read_csv**("data/AB_NYC_2019.csv")

listings_df.**info**() # big ol' list of types, indexes, non-null etc.
listings_df.**describe**() #

listings_df.**shape**   # tuple (rows, columns)
listings_df.**columns** # index([column names])
listings_df.**dtypes**  
# table list: name of column, type of column (float, object, int)

listings_df.**head**(x=5)  # graph of first 5 (or other) rows of DF
listings_df.**tail**(x-5)  # graph of last 5 (or other) rows

# cleaning data

listings_df.**ismull().sum()** 
# for each column, how many rows have a blank value 
# if 0 - clean, if None - bad

# creating a Dataframe object
df = pd.DataFrame(details, columns = ['Name', 'Age', 'University'],
                  index = ['a', 'b', 'c', 'd'])
 
# return a new dataframe by dropping a row
# 'b' & 'c' from dataframe
update_df = df.drop(['b', 'c'])

columns_to_drop = ["id", "host_name", "last review"]
listings_df.**drop**(columns_to_drop, axis="columns", **inplace**=True)
# inplace=True - data is changed in original data frame
# inplace=False - copy of data is given with dropped data, 
#  but original still contains dropped data

listings_df.**fillna**({"reviews_per_month": 0}, inplace=True)
# sets default value as 0 and replaces null values with it
# NaN - null "not a number" in float lists
```

```python
DataFrame.max(axis=_NoDefault.no_default, skipna=True, level=None, 
		numeric_only=None, **kwargs)

DataFrame.min(axis=_NoDefault.no_default, skipna=True, level=None, 
		numeric_only=None, **kwargs)[source]

DataFrame.mean(axis=_NoDefault.no_default, skipna=True, level=None, 
		numeric_only=None, **kwargs)[source]

DataFrame.idxmax(axis=0, skipna=True, numeric_only=False)
# prints the row name
```

```python
df.str.contains('ball')
# boolean index

.isin(["round", "triangle",...]) # very specific

df["area"].unique()
# shows values that are unique within a column

df['area'].str.strip[] #cleans white spaces
```

### Columns and Rows

```python
#filtering

listings_df["name"] # doesn't come back as dataframe, only a series
listings_df[**[**"name", "neighborhood_group", "price"**]**] # gets back DF
listings_df9[0:8][["name", "neighborhood_group", "price"]]
# rows 0 through 7, only columns shown
```

### Slicing

```python
DataFrame.**loc**[]
#Access a group of rows and columns by label(s) or a boolean array.
"""
Allowed inputs are:

    A single label, e.g. 5 or 'a', (note that 5 is interpreted as a label of the index, 
and never as an integer position along the index).

    A list or array of labels, e.g. ['a', 'b', 'c'].

    A slice object with labels, e.g. 'a':'f'.
"""
DF.loc[:, ["area", "volume"]] # if list, just area and volume
DF.loc[:, "area":"volume"] # if index, all columns from area through volume
DF.loc[:5, "area":"volume":3] # every third column

DataFrame.**iloc**[]
# Purely integer-location based indexing for selection by position.
"""
Allowed inputs are:

    An integer, e.g. 5.

    A list or array of integers, e.g. [4, 3, 0].

    A slice object with ints, e.g. 1:7.

    A boolean array.

    A callable function with one argument (the calling Series or DataFrame) and that returns valid output 
for indexing (one of the above). This is useful in method chains, when you don’t have a reference to the 
calling object, but would like to base your selection on some value.

    A tuple of row and column indexes. The tuple elements consist of one of the above inputs, e.g. (0, 1).
"""
DF.iloc[:, 0:3]
# all rows, first 4 columns **(inclusive)**
```

### Sorting

```python
df.sort_index(ascending=False) # descending

df.sort_values(by="population")
```

### Boolean Indexing

```python
listings_df["price"] < 100
## 0     False
## 1     False
## ...
#index, not value

listings_df[listings_df["price"] < 100]
# keeps only rows which have a price less than 100
```

```python
boolean_mask = df['population'] > 1_000_000
# creates series that is true if population is over one million
df[boolean_mask]
# only shows rows that have "True" (pop over 1mil)
```

### Indexing

```python
DataFrame.set_index(keys, drop=True, append=False, inplace=False, 
		verify_integrity=False)
# automatically sets index with an existing column

df['country'] = df['country'].map(str.strip)
df.set_index('country', inplace=True)
# change index to a string
```

### Grouping

- split, apply aggregate function, combine

```python
new = df.groupby(by='region').sum()
# makes the index the region and then sums all the values within each region
```