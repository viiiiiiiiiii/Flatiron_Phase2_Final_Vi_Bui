<img src='images/King County.jpeg'>

**King County, WA, U.S.A.**  

# Housing Guidance for King County, WA, U.S.A

* Student name: Vi Bui
* Student pace: Part-Time
* Scheduled project review date/time: 10/28/21
* Instructor name: Claude Fried
* Blog post URL: https://datasciish.com/

## Overview

**Client:** New WA state home buyers needing consultation on WA real estate market and expectations (price, size, location) 

**Data, Methodology, and Analysis:** King County (WA, U.S.A.) housing data from 2014-2015

**Results & Recommendations:** After analyzing data and building models assessing relationships between price and square feet; price and bedrooms; and price to zip code, we've modeled the expectations for price range depending on square feet of living space, grade, condition, and renovation status 

# Data Exploration, Cleansing, Visualization, and Preparation

**Data Exploration** <br>
Explore King County, WA, U.S.A. data from years 2014-2015

**Data Cleansing** <br>
Check for duplicates (none); drop NaN values and unnecessary columns; continuously clean data as necessary 

**Data Visualization** <br>
Use visualizations to explore the data and determine how to further refine the dataset in order to prepare for modeling 

**Data Preparation** <br>

## Data Exploration and Cleansing
Import data and all packages needed for data exploration and modeling 


```python
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
%matplotlib inline 

import seaborn as sns

import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os 
import warnings
```

Explore: columns, shape, info 


```python
df = pd.read_csv('data/kc_house_data.csv', index_col=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Explore number of entries; which columns have missing data; and data types 

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21597 entries, 7129300520 to 1523300157
    Data columns (total 20 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   date           21597 non-null  object 
     1   price          21597 non-null  float64
     2   bedrooms       21597 non-null  int64  
     3   bathrooms      21597 non-null  float64
     4   sqft_living    21597 non-null  int64  
     5   sqft_lot       21597 non-null  int64  
     6   floors         21597 non-null  float64
     7   waterfront     19221 non-null  float64
     8   view           21534 non-null  float64
     9   condition      21597 non-null  int64  
     10  grade          21597 non-null  int64  
     11  sqft_above     21597 non-null  int64  
     12  sqft_basement  21597 non-null  object 
     13  yr_built       21597 non-null  int64  
     14  yr_renovated   17755 non-null  float64
     15  zipcode        21597 non-null  int64  
     16  lat            21597 non-null  float64
     17  long           21597 non-null  float64
     18  sqft_living15  21597 non-null  int64  
     19  sqft_lot15     21597 non-null  int64  
    dtypes: float64(8), int64(10), object(2)
    memory usage: 3.5+ MB



```python
# Check for duplicates 

df.duplicated(keep='first').sum()
```




    0




```python
# Check for NaN values 

df.isna().sum()

# Columns and number of respective NaN values 
# waterfront       2376
# view               63
# yr_renovated     3842
```




    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2376
    view               63
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3842
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64




```python
# Explore columns 

df.columns
```




    Index(['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
           'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
           'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
           'sqft_living15', 'sqft_lot15'],
          dtype='object')



### Understand Column Names and Descriptions for King County's Data Set
* **id** - unique identified for a house
* **dateDate** - house was sold
* **pricePrice** -  is prediction target
* **bedroomsNumber** -  of Bedrooms/House
* **bathroomsNumber** -  of bathrooms/bedrooms
* **sqft_livingsquare** -  footage of the home
* **sqft_lotsquare** -  footage of the lot
* **floorsTotal** -  floors (levels) in house
* **waterfront** - House which has a view to a waterfront
* **view** - Has been viewed
* **condition** - How good the condition is ( Overall )
* **grade** - overall grade given to the housing unit, based on King County grading system
* **sqft_above** - square footage of house apart from basement
* **sqft_basement** - square footage of the basement
* **yr_built** - Built Year
* **yr_renovated** - Year when house was renovated
* **zipcode** - zip
* **lat** - Latitude coordinate
* **long** - Longitude coordinate
* **sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors


```python
# Calculate the percentage of NaN 

df['waterfront'].value_counts()

# Ask colleagues how to do this "smarter": (19075/(19075+146))
# (2376+19075)/(2376+19075+146)
```




    0.0    19075
    1.0      146
    Name: waterfront, dtype: int64



**Observations after exploring waterfront data:** 
- 99.2% of houses (146 out of 19,221) do not have a waterfront view
- With 2376 entries with NaN values, imputing the NaN values to 0 makes no material difference 
- Clean data: impute waterfront NaN values to 0 (represents no waterfront view) 
- Resulting data: 99.3% of houses (21,451 out of 21,597) do not have a waterfront view 


```python
# Impute waterfront NaN values to 0 

df['waterfront'] = df['waterfront'].fillna(0)
df['waterfront'].value_counts()
```




    0.0    21451
    1.0      146
    Name: waterfront, dtype: int64




```python
# Double check for NaN values left 

df['waterfront'].isna().sum()
```




    0




```python
# Continue exploring other data that needs to be cleansed 

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21597 entries, 7129300520 to 1523300157
    Data columns (total 20 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   date           21597 non-null  object 
     1   price          21597 non-null  float64
     2   bedrooms       21597 non-null  int64  
     3   bathrooms      21597 non-null  float64
     4   sqft_living    21597 non-null  int64  
     5   sqft_lot       21597 non-null  int64  
     6   floors         21597 non-null  float64
     7   waterfront     21597 non-null  float64
     8   view           21534 non-null  float64
     9   condition      21597 non-null  int64  
     10  grade          21597 non-null  int64  
     11  sqft_above     21597 non-null  int64  
     12  sqft_basement  21597 non-null  object 
     13  yr_built       21597 non-null  int64  
     14  yr_renovated   17755 non-null  float64
     15  zipcode        21597 non-null  int64  
     16  lat            21597 non-null  float64
     17  long           21597 non-null  float64
     18  sqft_living15  21597 non-null  int64  
     19  sqft_lot15     21597 non-null  int64  
    dtypes: float64(8), int64(10), object(2)
    memory usage: 3.5+ MB



```python
df['yr_renovated'].value_counts()
```




    0.0       17011
    2014.0       73
    2003.0       31
    2013.0       31
    2007.0       30
              ...  
    1946.0        1
    1959.0        1
    1971.0        1
    1951.0        1
    1954.0        1
    Name: yr_renovated, Length: 70, dtype: int64



**'yr_renovated' data needs to be cleansed. Observations about 'yr_renovated':** 

- 'yr_renovated' has 3842 NaN values 
- About the data: if house has been renovated, the year is entered. If not, 0 has been entered 
- 95.8% of current data set (17,011 of 17,755 houses) have not been renovated
- Imputing the 3842 NaN values to 0 (not renovated) does not make a substantial difference  
- Resulting data: 96.6% of new data set (20,853 of 21,597 houses) have not been renovated  


```python
df['yr_renovated'] = df['yr_renovated'].fillna(0)
df['yr_renovated'].value_counts()
```




    0.0       20853
    2014.0       73
    2003.0       31
    2013.0       31
    2007.0       30
              ...  
    1946.0        1
    1959.0        1
    1971.0        1
    1951.0        1
    1954.0        1
    Name: yr_renovated, Length: 70, dtype: int64




```python
# ask colleagues how you would get the sum of the value counts 
# df['yr_renovated'].value_counts('0')
20853/21597
```




    0.9655507709404084




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



**'sqft_basement' data needs to be cleansed. Observations about 'sqft_basement':** 

- Ran into an error with 'sqft_basement' data
- Found there are 454 entries with '?' symbols in the data
- 60.7% of current data set (12,826 of 21,143 houses) have 0 as entered for sqft_basement
- Imputing the 454 '?' entries to 0 does not make a substantial difference  
- Resulting data: 61.5% of new data set (13,280 of 21,597 houses) have 0 sqft_basement


```python
# Check how many entries for 'sqft_basement' are '?'
# Ask colleagues how to view the entire data set for sqft_basement

df['sqft_basement'].value_counts()
```




    0.0       12826
    ?           454
    600.0       217
    500.0       209
    700.0       208
              ...  
    2240.0        1
    2850.0        1
    1135.0        1
    3000.0        1
    508.0         1
    Name: sqft_basement, Length: 304, dtype: int64




```python
# Impute 454 '?' entries to 0 values 
# Transform data type from object to float 

df['sqft_basement'] = df['sqft_basement'].apply(lambda x: 0 if x == '?' else x).astype(float)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21597 entries, 7129300520 to 1523300157
    Data columns (total 20 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   date           21597 non-null  object 
     1   price          21597 non-null  float64
     2   bedrooms       21597 non-null  int64  
     3   bathrooms      21597 non-null  float64
     4   sqft_living    21597 non-null  int64  
     5   sqft_lot       21597 non-null  int64  
     6   floors         21597 non-null  float64
     7   waterfront     21597 non-null  float64
     8   view           21534 non-null  float64
     9   condition      21597 non-null  int64  
     10  grade          21597 non-null  int64  
     11  sqft_above     21597 non-null  int64  
     12  sqft_basement  21597 non-null  float64
     13  yr_built       21597 non-null  int64  
     14  yr_renovated   21597 non-null  float64
     15  zipcode        21597 non-null  int64  
     16  lat            21597 non-null  float64
     17  long           21597 non-null  float64
     18  sqft_living15  21597 non-null  int64  
     19  sqft_lot15     21597 non-null  int64  
    dtypes: float64(9), int64(10), object(1)
    memory usage: 3.5+ MB


**Continue cleaning data/transform data types:** 
- Transform data types
- Most importantly, convert zipcode from integer to string


```python
# yr_renovated from float to integer (preference) 
# zipcode from integer to string 

df['yr_renovated'] = (df['yr_renovated'].astype(int))
df['zipcode'] = (df['zipcode'].astype(str))
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21597 entries, 7129300520 to 1523300157
    Data columns (total 20 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   date           21597 non-null  object 
     1   price          21597 non-null  float64
     2   bedrooms       21597 non-null  int64  
     3   bathrooms      21597 non-null  float64
     4   sqft_living    21597 non-null  int64  
     5   sqft_lot       21597 non-null  int64  
     6   floors         21597 non-null  float64
     7   waterfront     21597 non-null  float64
     8   view           21534 non-null  float64
     9   condition      21597 non-null  int64  
     10  grade          21597 non-null  int64  
     11  sqft_above     21597 non-null  int64  
     12  sqft_basement  21597 non-null  float64
     13  yr_built       21597 non-null  int64  
     14  yr_renovated   21597 non-null  int64  
     15  zipcode        21597 non-null  object 
     16  lat            21597 non-null  float64
     17  long           21597 non-null  float64
     18  sqft_living15  21597 non-null  int64  
     19  sqft_lot15     21597 non-null  int64  
    dtypes: float64(8), int64(10), object(2)
    memory usage: 3.5+ MB


## Create New Features 

1. Year Sold (from date column) 
2. Renovated (make renovated a binary value: renovated = 1; not renovated = 0)
3. Basement Present (make basement a binary value: renovated = 1; not renovated = 0)
4. Actual Age of Property (year sold - year built) 
5. Bathrooms Per Bedroom (bathrooms/bedrooms)
6. Square Feet Living to Square Foot Lot (sqft_living/sqft_lot)


```python
# Create new features 

df['yr_sold'] = (df['date'].str[-4:].astype(int))
df['renovated'] = np.where(df['yr_renovated']!=0, 1,0)
df['basement_present'] = np.where(df['sqft_basement']!=0, 1,0)
df['actual_age_of_property'] = df['yr_sold']-df['yr_built']
df['bathrooms_per_bedroom'] = df['bathrooms']/df['bedrooms']
df['sqft_living_to_sqft_lot'] = df['sqft_living']/df['sqft_lot'] 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>...</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>2014</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.333333</td>
      <td>0.208850</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>...</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>2014</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.750000</td>
      <td>0.354874</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>...</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>2015</td>
      <td>0</td>
      <td>0</td>
      <td>82</td>
      <td>0.500000</td>
      <td>0.077000</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>...</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>2014</td>
      <td>0</td>
      <td>1</td>
      <td>49</td>
      <td>0.750000</td>
      <td>0.392000</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>...</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>2015</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.666667</td>
      <td>0.207921</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# Check: data types
# Check: all value counts match 

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21597 entries, 7129300520 to 1523300157
    Data columns (total 26 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   date                     21597 non-null  object 
     1   price                    21597 non-null  float64
     2   bedrooms                 21597 non-null  int64  
     3   bathrooms                21597 non-null  float64
     4   sqft_living              21597 non-null  int64  
     5   sqft_lot                 21597 non-null  int64  
     6   floors                   21597 non-null  float64
     7   waterfront               21597 non-null  float64
     8   view                     21534 non-null  float64
     9   condition                21597 non-null  int64  
     10  grade                    21597 non-null  int64  
     11  sqft_above               21597 non-null  int64  
     12  sqft_basement            21597 non-null  float64
     13  yr_built                 21597 non-null  int64  
     14  yr_renovated             21597 non-null  int64  
     15  zipcode                  21597 non-null  object 
     16  lat                      21597 non-null  float64
     17  long                     21597 non-null  float64
     18  sqft_living15            21597 non-null  int64  
     19  sqft_lot15               21597 non-null  int64  
     20  yr_sold                  21597 non-null  int64  
     21  renovated                21597 non-null  int64  
     22  basement_present         21597 non-null  int64  
     23  actual_age_of_property   21597 non-null  int64  
     24  bathrooms_per_bedroom    21597 non-null  float64
     25  sqft_living_to_sqft_lot  21597 non-null  float64
    dtypes: float64(10), int64(14), object(2)
    memory usage: 4.4+ MB



```python
df.columns
```




    Index(['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
           'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
           'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
           'sqft_living15', 'sqft_lot15', 'yr_sold', 'renovated',
           'basement_present', 'actual_age_of_property', 'bathrooms_per_bedroom',
           'sqft_living_to_sqft_lot'],
          dtype='object')




```python
# Explore correlation for numerical values with .corr()

df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>1.000000</td>
      <td>0.308787</td>
      <td>0.525906</td>
      <td>0.701917</td>
      <td>0.089876</td>
      <td>0.256804</td>
      <td>0.264306</td>
      <td>0.395734</td>
      <td>0.036056</td>
      <td>0.667951</td>
      <td>...</td>
      <td>0.306692</td>
      <td>0.022036</td>
      <td>0.585241</td>
      <td>0.082845</td>
      <td>0.003727</td>
      <td>0.117543</td>
      <td>0.178264</td>
      <td>-0.053890</td>
      <td>0.281227</td>
      <td>0.123063</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>0.308787</td>
      <td>1.000000</td>
      <td>0.514508</td>
      <td>0.578212</td>
      <td>0.032471</td>
      <td>0.177944</td>
      <td>-0.002127</td>
      <td>0.078523</td>
      <td>0.026496</td>
      <td>0.356563</td>
      <td>...</td>
      <td>-0.009951</td>
      <td>0.132054</td>
      <td>0.393406</td>
      <td>0.030690</td>
      <td>-0.009949</td>
      <td>0.017635</td>
      <td>0.158412</td>
      <td>-0.155817</td>
      <td>-0.236129</td>
      <td>0.026798</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>0.525906</td>
      <td>0.514508</td>
      <td>1.000000</td>
      <td>0.755758</td>
      <td>0.088373</td>
      <td>0.502582</td>
      <td>0.063629</td>
      <td>0.186451</td>
      <td>-0.126479</td>
      <td>0.665838</td>
      <td>...</td>
      <td>0.024280</td>
      <td>0.224903</td>
      <td>0.569884</td>
      <td>0.088303</td>
      <td>-0.026577</td>
      <td>0.046742</td>
      <td>0.159863</td>
      <td>-0.507561</td>
      <td>0.652668</td>
      <td>0.287015</td>
    </tr>
    <tr>
      <th>sqft_living</th>
      <td>0.701917</td>
      <td>0.578212</td>
      <td>0.755758</td>
      <td>1.000000</td>
      <td>0.173453</td>
      <td>0.353953</td>
      <td>0.104637</td>
      <td>0.282532</td>
      <td>-0.059445</td>
      <td>0.762779</td>
      <td>...</td>
      <td>0.052155</td>
      <td>0.241214</td>
      <td>0.756402</td>
      <td>0.184342</td>
      <td>-0.029014</td>
      <td>0.050829</td>
      <td>0.201198</td>
      <td>-0.318592</td>
      <td>0.310690</td>
      <td>0.076988</td>
    </tr>
    <tr>
      <th>sqft_lot</th>
      <td>0.089876</td>
      <td>0.032471</td>
      <td>0.088373</td>
      <td>0.173453</td>
      <td>1.000000</td>
      <td>-0.004814</td>
      <td>0.021459</td>
      <td>0.075298</td>
      <td>-0.008830</td>
      <td>0.114731</td>
      <td>...</td>
      <td>-0.085514</td>
      <td>0.230227</td>
      <td>0.144763</td>
      <td>0.718204</td>
      <td>0.005628</td>
      <td>0.005091</td>
      <td>-0.034889</td>
      <td>-0.052853</td>
      <td>0.063306</td>
      <td>-0.252601</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>0.256804</td>
      <td>0.177944</td>
      <td>0.502582</td>
      <td>0.353953</td>
      <td>-0.004814</td>
      <td>1.000000</td>
      <td>0.020797</td>
      <td>0.028436</td>
      <td>-0.264075</td>
      <td>0.458794</td>
      <td>...</td>
      <td>0.049239</td>
      <td>0.125943</td>
      <td>0.280102</td>
      <td>-0.010722</td>
      <td>-0.022352</td>
      <td>0.003713</td>
      <td>-0.252465</td>
      <td>-0.489514</td>
      <td>0.421169</td>
      <td>0.556700</td>
    </tr>
    <tr>
      <th>waterfront</th>
      <td>0.264306</td>
      <td>-0.002127</td>
      <td>0.063629</td>
      <td>0.104637</td>
      <td>0.021459</td>
      <td>0.020797</td>
      <td>1.000000</td>
      <td>0.382000</td>
      <td>0.016648</td>
      <td>0.082818</td>
      <td>...</td>
      <td>-0.012157</td>
      <td>-0.037628</td>
      <td>0.083823</td>
      <td>0.030658</td>
      <td>-0.005018</td>
      <td>0.074267</td>
      <td>0.039220</td>
      <td>0.024406</td>
      <td>0.073760</td>
      <td>-0.029806</td>
    </tr>
    <tr>
      <th>view</th>
      <td>0.395734</td>
      <td>0.078523</td>
      <td>0.186451</td>
      <td>0.282532</td>
      <td>0.075298</td>
      <td>0.028436</td>
      <td>0.382000</td>
      <td>1.000000</td>
      <td>0.045735</td>
      <td>0.249727</td>
      <td>...</td>
      <td>0.006141</td>
      <td>-0.077894</td>
      <td>0.279561</td>
      <td>0.073332</td>
      <td>0.001504</td>
      <td>0.090465</td>
      <td>0.177478</td>
      <td>0.054584</td>
      <td>0.131381</td>
      <td>-0.002310</td>
    </tr>
    <tr>
      <th>condition</th>
      <td>0.036056</td>
      <td>0.026496</td>
      <td>-0.126479</td>
      <td>-0.059445</td>
      <td>-0.008830</td>
      <td>-0.264075</td>
      <td>0.016648</td>
      <td>0.045735</td>
      <td>1.000000</td>
      <td>-0.146896</td>
      <td>...</td>
      <td>-0.015102</td>
      <td>-0.105877</td>
      <td>-0.093072</td>
      <td>-0.003126</td>
      <td>-0.045898</td>
      <td>-0.055383</td>
      <td>0.130542</td>
      <td>0.360836</td>
      <td>-0.158662</td>
      <td>-0.156501</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>0.667951</td>
      <td>0.356563</td>
      <td>0.665838</td>
      <td>0.762779</td>
      <td>0.114731</td>
      <td>0.458794</td>
      <td>0.082818</td>
      <td>0.249727</td>
      <td>-0.146896</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.113575</td>
      <td>0.200341</td>
      <td>0.713867</td>
      <td>0.120981</td>
      <td>-0.030635</td>
      <td>0.015259</td>
      <td>0.050701</td>
      <td>-0.448322</td>
      <td>0.409125</td>
      <td>0.191398</td>
    </tr>
    <tr>
      <th>sqft_above</th>
      <td>0.605368</td>
      <td>0.479386</td>
      <td>0.686668</td>
      <td>0.876448</td>
      <td>0.184139</td>
      <td>0.523989</td>
      <td>0.071778</td>
      <td>0.166299</td>
      <td>-0.158904</td>
      <td>0.756073</td>
      <td>...</td>
      <td>-0.001199</td>
      <td>0.344842</td>
      <td>0.731767</td>
      <td>0.195077</td>
      <td>-0.023782</td>
      <td>0.020566</td>
      <td>-0.207268</td>
      <td>-0.424386</td>
      <td>0.309985</td>
      <td>0.051612</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>0.321108</td>
      <td>0.297229</td>
      <td>0.278485</td>
      <td>0.428660</td>
      <td>0.015031</td>
      <td>-0.241866</td>
      <td>0.083050</td>
      <td>0.271689</td>
      <td>0.168482</td>
      <td>0.165843</td>
      <td>...</td>
      <td>0.109853</td>
      <td>-0.142369</td>
      <td>0.199288</td>
      <td>0.015885</td>
      <td>-0.014997</td>
      <td>0.064675</td>
      <td>0.820893</td>
      <td>0.129837</td>
      <td>0.063303</td>
      <td>0.062006</td>
    </tr>
    <tr>
      <th>yr_built</th>
      <td>0.053953</td>
      <td>0.155670</td>
      <td>0.507173</td>
      <td>0.318152</td>
      <td>0.052946</td>
      <td>0.489193</td>
      <td>-0.024487</td>
      <td>-0.054564</td>
      <td>-0.361592</td>
      <td>0.447865</td>
      <td>...</td>
      <td>-0.148370</td>
      <td>0.409993</td>
      <td>0.326377</td>
      <td>0.070777</td>
      <td>0.003574</td>
      <td>-0.202837</td>
      <td>-0.163992</td>
      <td>-0.999873</td>
      <td>0.427585</td>
      <td>0.279053</td>
    </tr>
    <tr>
      <th>yr_renovated</th>
      <td>0.117855</td>
      <td>0.017900</td>
      <td>0.047177</td>
      <td>0.051060</td>
      <td>0.004979</td>
      <td>0.003793</td>
      <td>0.073939</td>
      <td>0.090324</td>
      <td>-0.055808</td>
      <td>0.015623</td>
      <td>...</td>
      <td>0.027970</td>
      <td>-0.064543</td>
      <td>0.000683</td>
      <td>0.004286</td>
      <td>-0.019713</td>
      <td>0.999968</td>
      <td>0.044838</td>
      <td>0.202227</td>
      <td>0.039398</td>
      <td>-0.001567</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>0.306692</td>
      <td>-0.009951</td>
      <td>0.024280</td>
      <td>0.052155</td>
      <td>-0.085514</td>
      <td>0.049239</td>
      <td>-0.012157</td>
      <td>0.006141</td>
      <td>-0.015102</td>
      <td>0.113575</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.135371</td>
      <td>0.048679</td>
      <td>-0.086139</td>
      <td>-0.029003</td>
      <td>0.027908</td>
      <td>0.136602</td>
      <td>0.147898</td>
      <td>0.045327</td>
      <td>0.164559</td>
    </tr>
    <tr>
      <th>long</th>
      <td>0.022036</td>
      <td>0.132054</td>
      <td>0.224903</td>
      <td>0.241214</td>
      <td>0.230227</td>
      <td>0.125943</td>
      <td>-0.037628</td>
      <td>-0.077894</td>
      <td>-0.105877</td>
      <td>0.200341</td>
      <td>...</td>
      <td>-0.135371</td>
      <td>1.000000</td>
      <td>0.335626</td>
      <td>0.255586</td>
      <td>0.000296</td>
      <td>-0.064511</td>
      <td>-0.233366</td>
      <td>-0.409959</td>
      <td>0.122091</td>
      <td>-0.203992</td>
    </tr>
    <tr>
      <th>sqft_living15</th>
      <td>0.585241</td>
      <td>0.393406</td>
      <td>0.569884</td>
      <td>0.756402</td>
      <td>0.144763</td>
      <td>0.280102</td>
      <td>0.083823</td>
      <td>0.279561</td>
      <td>-0.093072</td>
      <td>0.713867</td>
      <td>...</td>
      <td>0.048679</td>
      <td>0.335626</td>
      <td>1.000000</td>
      <td>0.183515</td>
      <td>-0.021549</td>
      <td>0.000622</td>
      <td>0.044577</td>
      <td>-0.326697</td>
      <td>0.263859</td>
      <td>-0.042097</td>
    </tr>
    <tr>
      <th>sqft_lot15</th>
      <td>0.082845</td>
      <td>0.030690</td>
      <td>0.088303</td>
      <td>0.184342</td>
      <td>0.718204</td>
      <td>-0.010722</td>
      <td>0.030658</td>
      <td>0.073332</td>
      <td>-0.003126</td>
      <td>0.120981</td>
      <td>...</td>
      <td>-0.086139</td>
      <td>0.255586</td>
      <td>0.183515</td>
      <td>1.000000</td>
      <td>0.000162</td>
      <td>0.004380</td>
      <td>-0.041747</td>
      <td>-0.070770</td>
      <td>0.061022</td>
      <td>-0.277338</td>
    </tr>
    <tr>
      <th>yr_sold</th>
      <td>0.003727</td>
      <td>-0.009949</td>
      <td>-0.026577</td>
      <td>-0.029014</td>
      <td>0.005628</td>
      <td>-0.022352</td>
      <td>-0.005018</td>
      <td>0.001504</td>
      <td>-0.045898</td>
      <td>-0.030635</td>
      <td>...</td>
      <td>-0.029003</td>
      <td>0.000296</td>
      <td>-0.021549</td>
      <td>0.000162</td>
      <td>1.000000</td>
      <td>-0.019699</td>
      <td>-0.006323</td>
      <td>0.012344</td>
      <td>-0.023098</td>
      <td>-0.008569</td>
    </tr>
    <tr>
      <th>renovated</th>
      <td>0.117543</td>
      <td>0.017635</td>
      <td>0.046742</td>
      <td>0.050829</td>
      <td>0.005091</td>
      <td>0.003713</td>
      <td>0.074267</td>
      <td>0.090465</td>
      <td>-0.055383</td>
      <td>0.015259</td>
      <td>...</td>
      <td>0.027908</td>
      <td>-0.064511</td>
      <td>0.000622</td>
      <td>0.004380</td>
      <td>-0.019699</td>
      <td>1.000000</td>
      <td>0.044600</td>
      <td>0.202510</td>
      <td>0.039192</td>
      <td>-0.001831</td>
    </tr>
    <tr>
      <th>basement_present</th>
      <td>0.178264</td>
      <td>0.158412</td>
      <td>0.159863</td>
      <td>0.201198</td>
      <td>-0.034889</td>
      <td>-0.252465</td>
      <td>0.039220</td>
      <td>0.177478</td>
      <td>0.130542</td>
      <td>0.050701</td>
      <td>...</td>
      <td>0.136602</td>
      <td>-0.233366</td>
      <td>0.044577</td>
      <td>-0.041747</td>
      <td>-0.006323</td>
      <td>0.044600</td>
      <td>1.000000</td>
      <td>0.163880</td>
      <td>0.066211</td>
      <td>0.152095</td>
    </tr>
    <tr>
      <th>actual_age_of_property</th>
      <td>-0.053890</td>
      <td>-0.155817</td>
      <td>-0.507561</td>
      <td>-0.318592</td>
      <td>-0.052853</td>
      <td>-0.489514</td>
      <td>0.024406</td>
      <td>0.054584</td>
      <td>0.360836</td>
      <td>-0.448322</td>
      <td>...</td>
      <td>0.147898</td>
      <td>-0.409959</td>
      <td>-0.326697</td>
      <td>-0.070770</td>
      <td>0.012344</td>
      <td>0.202510</td>
      <td>0.163880</td>
      <td>1.000000</td>
      <td>-0.427923</td>
      <td>-0.279170</td>
    </tr>
    <tr>
      <th>bathrooms_per_bedroom</th>
      <td>0.281227</td>
      <td>-0.236129</td>
      <td>0.652668</td>
      <td>0.310690</td>
      <td>0.063306</td>
      <td>0.421169</td>
      <td>0.073760</td>
      <td>0.131381</td>
      <td>-0.158662</td>
      <td>0.409125</td>
      <td>...</td>
      <td>0.045327</td>
      <td>0.122091</td>
      <td>0.263859</td>
      <td>0.061022</td>
      <td>-0.023098</td>
      <td>0.039192</td>
      <td>0.066211</td>
      <td>-0.427923</td>
      <td>1.000000</td>
      <td>0.343570</td>
    </tr>
    <tr>
      <th>sqft_living_to_sqft_lot</th>
      <td>0.123063</td>
      <td>0.026798</td>
      <td>0.287015</td>
      <td>0.076988</td>
      <td>-0.252601</td>
      <td>0.556700</td>
      <td>-0.029806</td>
      <td>-0.002310</td>
      <td>-0.156501</td>
      <td>0.191398</td>
      <td>...</td>
      <td>0.164559</td>
      <td>-0.203992</td>
      <td>-0.042097</td>
      <td>-0.277338</td>
      <td>-0.008569</td>
      <td>-0.001831</td>
      <td>0.152095</td>
      <td>-0.279170</td>
      <td>0.343570</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>24 rows × 24 columns</p>
</div>



- Explore descriptive statistics with .describe()
- Summarizes central tendency (mean), dispersion and shape of a dataset’s distribution, excluding NaN values


```python
# Explore descriptive statistics with .describe()
# Summarizes central tendency (mean), dispersion and shape of a dataset’s distribution, excluding NaN values

df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21534.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>...</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.00000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.402966e+05</td>
      <td>3.373200</td>
      <td>2.115826</td>
      <td>2080.321850</td>
      <td>1.509941e+04</td>
      <td>1.494096</td>
      <td>0.006760</td>
      <td>0.233863</td>
      <td>3.409825</td>
      <td>7.657915</td>
      <td>...</td>
      <td>47.560093</td>
      <td>-122.213982</td>
      <td>1986.620318</td>
      <td>12758.283512</td>
      <td>2014.322962</td>
      <td>0.034449</td>
      <td>0.38510</td>
      <td>43.323286</td>
      <td>0.640969</td>
      <td>0.323755</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.673681e+05</td>
      <td>0.926299</td>
      <td>0.768984</td>
      <td>918.106125</td>
      <td>4.141264e+04</td>
      <td>0.539683</td>
      <td>0.081944</td>
      <td>0.765686</td>
      <td>0.650546</td>
      <td>1.173200</td>
      <td>...</td>
      <td>0.138552</td>
      <td>0.140724</td>
      <td>685.230472</td>
      <td>27274.441950</td>
      <td>0.467619</td>
      <td>0.182384</td>
      <td>0.48663</td>
      <td>29.377285</td>
      <td>0.211651</td>
      <td>0.268460</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>-1.000000</td>
      <td>0.053030</td>
      <td>0.000610</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.471100</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>18.000000</td>
      <td>0.500000</td>
      <td>0.156663</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.571800</td>
      <td>-122.231000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>40.000000</td>
      <td>0.625000</td>
      <td>0.247666</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068500e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
      <td>2015.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>63.000000</td>
      <td>0.750000</td>
      <td>0.407609</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>...</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>115.000000</td>
      <td>2.500000</td>
      <td>4.653846</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 24 columns</p>
</div>




```python
# Explore distribution (value_counts) of bedroom data 

df['bedrooms'].value_counts()
```




    3     9824
    4     6882
    2     2760
    5     1601
    6      272
    1      196
    7       38
    8       13
    9        6
    10       3
    11       1
    33       1
    Name: bedrooms, dtype: int64



## Data Visualization 


```python
plt.figure(figsize=(12,8))
sns.displot(df['price'],bins=1000)
plt.title('Price')
plt.show();
```


    <Figure size 864x576 with 0 Axes>



    
![png](output_40_1.png)
    



```python
fig, ax = plt.subplots(figsize=(12,8))
sns.boxplot(x='price', data=df, ax=ax)
```




    <AxesSubplot:xlabel='price'>




    
![png](output_41_1.png)
    



```python
plt.figure(figsize=(12,8))
sns.displot(df['price'],bins=25)
plt.title('Price')
plt.show();
```


    <Figure size 864x576 with 0 Axes>



    
![png](output_42_1.png)
    



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21534.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>...</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.00000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.402966e+05</td>
      <td>3.373200</td>
      <td>2.115826</td>
      <td>2080.321850</td>
      <td>1.509941e+04</td>
      <td>1.494096</td>
      <td>0.006760</td>
      <td>0.233863</td>
      <td>3.409825</td>
      <td>7.657915</td>
      <td>...</td>
      <td>47.560093</td>
      <td>-122.213982</td>
      <td>1986.620318</td>
      <td>12758.283512</td>
      <td>2014.322962</td>
      <td>0.034449</td>
      <td>0.38510</td>
      <td>43.323286</td>
      <td>0.640969</td>
      <td>0.323755</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.673681e+05</td>
      <td>0.926299</td>
      <td>0.768984</td>
      <td>918.106125</td>
      <td>4.141264e+04</td>
      <td>0.539683</td>
      <td>0.081944</td>
      <td>0.765686</td>
      <td>0.650546</td>
      <td>1.173200</td>
      <td>...</td>
      <td>0.138552</td>
      <td>0.140724</td>
      <td>685.230472</td>
      <td>27274.441950</td>
      <td>0.467619</td>
      <td>0.182384</td>
      <td>0.48663</td>
      <td>29.377285</td>
      <td>0.211651</td>
      <td>0.268460</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>-1.000000</td>
      <td>0.053030</td>
      <td>0.000610</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.471100</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>18.000000</td>
      <td>0.500000</td>
      <td>0.156663</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.571800</td>
      <td>-122.231000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>40.000000</td>
      <td>0.625000</td>
      <td>0.247666</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068500e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
      <td>2015.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>63.000000</td>
      <td>0.750000</td>
      <td>0.407609</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>...</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>115.000000</td>
      <td>2.500000</td>
      <td>4.653846</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 24 columns</p>
</div>



**Narrow price range to \\$175,000-\\$650,000**


```python
# Ask colleagues how to find where the "majority" of prices fall 

df = df[df['price'].between(175_000,650_000)]
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15993 entries, 7129300520 to 1523300157
    Data columns (total 26 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   date                     15993 non-null  object 
     1   price                    15993 non-null  float64
     2   bedrooms                 15993 non-null  int64  
     3   bathrooms                15993 non-null  float64
     4   sqft_living              15993 non-null  int64  
     5   sqft_lot                 15993 non-null  int64  
     6   floors                   15993 non-null  float64
     7   waterfront               15993 non-null  float64
     8   view                     15949 non-null  float64
     9   condition                15993 non-null  int64  
     10  grade                    15993 non-null  int64  
     11  sqft_above               15993 non-null  int64  
     12  sqft_basement            15993 non-null  float64
     13  yr_built                 15993 non-null  int64  
     14  yr_renovated             15993 non-null  int64  
     15  zipcode                  15993 non-null  object 
     16  lat                      15993 non-null  float64
     17  long                     15993 non-null  float64
     18  sqft_living15            15993 non-null  int64  
     19  sqft_lot15               15993 non-null  int64  
     20  yr_sold                  15993 non-null  int64  
     21  renovated                15993 non-null  int64  
     22  basement_present         15993 non-null  int64  
     23  actual_age_of_property   15993 non-null  int64  
     24  bathrooms_per_bedroom    15993 non-null  float64
     25  sqft_living_to_sqft_lot  15993 non-null  float64
    dtypes: float64(10), int64(14), object(2)
    memory usage: 3.3+ MB



```python
# Percentage of data that will be used 

15_993/21_597
```




    0.7405195165995277




```python
plt.figure(figsize=(12,8))
sns.displot(df['price'],bins=25)
plt.title('Price')
plt.show();
```


    <Figure size 864x576 with 0 Axes>



    
![png](output_48_1.png)
    



```python
# Explore the data - specifically bedrooms, bathrooms, and sqft_living

df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>1.599300e+04</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15949.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>...</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>400987.591009</td>
      <td>3.244982</td>
      <td>1.953667</td>
      <td>1800.124742</td>
      <td>1.317614e+04</td>
      <td>1.431533</td>
      <td>0.001438</td>
      <td>0.109411</td>
      <td>3.397174</td>
      <td>7.301132</td>
      <td>...</td>
      <td>47.543802</td>
      <td>-122.217163</td>
      <td>1792.705246</td>
      <td>11677.541987</td>
      <td>2014.323517</td>
      <td>0.024136</td>
      <td>0.359720</td>
      <td>43.468142</td>
      <td>0.619331</td>
      <td>0.312434</td>
    </tr>
    <tr>
      <th>std</th>
      <td>123893.841894</td>
      <td>0.886582</td>
      <td>0.660056</td>
      <td>631.746037</td>
      <td>3.310214e+04</td>
      <td>0.537112</td>
      <td>0.037897</td>
      <td>0.500945</td>
      <td>0.634151</td>
      <td>0.831989</td>
      <td>...</td>
      <td>0.148391</td>
      <td>0.141616</td>
      <td>496.582637</td>
      <td>24240.185920</td>
      <td>0.467833</td>
      <td>0.153475</td>
      <td>0.479933</td>
      <td>28.270545</td>
      <td>0.206016</td>
      <td>0.271903</td>
    </tr>
    <tr>
      <th>min</th>
      <td>175000.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.720000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.053030</td>
      <td>0.000610</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299950.000000</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>1330.000000</td>
      <td>5.000000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.431000</td>
      <td>-122.332000</td>
      <td>1430.000000</td>
      <td>5040.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>0.500000</td>
      <td>0.152857</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>392000.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1720.000000</td>
      <td>7.439000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.545900</td>
      <td>-122.247000</td>
      <td>1710.000000</td>
      <td>7500.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>41.000000</td>
      <td>0.583333</td>
      <td>0.232076</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>499990.000000</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2180.000000</td>
      <td>9.968000e+03</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>47.682700</td>
      <td>-122.133000</td>
      <td>2090.000000</td>
      <td>9601.000000</td>
      <td>2015.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>62.000000</td>
      <td>0.750000</td>
      <td>0.370000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>650000.000000</td>
      <td>33.000000</td>
      <td>7.500000</td>
      <td>5461.000000</td>
      <td>1.164794e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>...</td>
      <td>47.777600</td>
      <td>-121.319000</td>
      <td>4362.000000</td>
      <td>438213.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>115.000000</td>
      <td>2.500000</td>
      <td>2.291399</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 24 columns</p>
</div>




```python
# Explore correlation after narrowing data set 

df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>1.000000</td>
      <td>0.176433</td>
      <td>0.315515</td>
      <td>0.417273</td>
      <td>0.071915</td>
      <td>0.200133</td>
      <td>0.022750</td>
      <td>0.126625</td>
      <td>-0.003818</td>
      <td>0.451436</td>
      <td>...</td>
      <td>0.479098</td>
      <td>0.054300</td>
      <td>0.382913</td>
      <td>0.067769</td>
      <td>0.007087</td>
      <td>0.027333</td>
      <td>0.174533</td>
      <td>-0.040202</td>
      <td>0.184907</td>
      <td>0.188304</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>0.176433</td>
      <td>1.000000</td>
      <td>0.456624</td>
      <td>0.589010</td>
      <td>0.020603</td>
      <td>0.102590</td>
      <td>-0.038404</td>
      <td>0.008418</td>
      <td>0.020562</td>
      <td>0.256539</td>
      <td>...</td>
      <td>-0.107569</td>
      <td>0.132547</td>
      <td>0.342631</td>
      <td>0.015451</td>
      <td>-0.009127</td>
      <td>-0.008531</td>
      <td>0.138967</td>
      <td>-0.162862</td>
      <td>-0.306894</td>
      <td>-0.024121</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>0.315515</td>
      <td>0.456624</td>
      <td>1.000000</td>
      <td>0.666310</td>
      <td>0.023810</td>
      <td>0.492458</td>
      <td>-0.029209</td>
      <td>0.038959</td>
      <td>-0.161222</td>
      <td>0.557766</td>
      <td>...</td>
      <td>-0.112134</td>
      <td>0.232980</td>
      <td>0.478078</td>
      <td>0.024263</td>
      <td>-0.027191</td>
      <td>-0.013034</td>
      <td>0.121556</td>
      <td>-0.594414</td>
      <td>0.638663</td>
      <td>0.328013</td>
    </tr>
    <tr>
      <th>sqft_living</th>
      <td>0.417273</td>
      <td>0.589010</td>
      <td>0.666310</td>
      <td>1.000000</td>
      <td>0.132353</td>
      <td>0.268110</td>
      <td>-0.011641</td>
      <td>0.109977</td>
      <td>-0.076913</td>
      <td>0.586257</td>
      <td>...</td>
      <td>-0.143029</td>
      <td>0.256092</td>
      <td>0.682108</td>
      <td>0.143713</td>
      <td>-0.024446</td>
      <td>-0.001096</td>
      <td>0.200849</td>
      <td>-0.353860</td>
      <td>0.181322</td>
      <td>0.050878</td>
    </tr>
    <tr>
      <th>sqft_lot</th>
      <td>0.071915</td>
      <td>0.020603</td>
      <td>0.023810</td>
      <td>0.132353</td>
      <td>1.000000</td>
      <td>-0.052962</td>
      <td>0.021022</td>
      <td>0.105113</td>
      <td>0.015127</td>
      <td>0.034926</td>
      <td>...</td>
      <td>-0.107369</td>
      <td>0.217447</td>
      <td>0.152715</td>
      <td>0.712409</td>
      <td>-0.006834</td>
      <td>0.018080</td>
      <td>-0.020799</td>
      <td>-0.005201</td>
      <td>0.005793</td>
      <td>-0.257394</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>0.200133</td>
      <td>0.102590</td>
      <td>0.492458</td>
      <td>0.268110</td>
      <td>-0.052962</td>
      <td>1.000000</td>
      <td>-0.015131</td>
      <td>-0.030150</td>
      <td>-0.291827</td>
      <td>0.424865</td>
      <td>...</td>
      <td>-0.017537</td>
      <td>0.105360</td>
      <td>0.204379</td>
      <td>-0.057095</td>
      <td>-0.018975</td>
      <td>-0.021294</td>
      <td>-0.292454</td>
      <td>-0.547963</td>
      <td>0.435363</td>
      <td>0.610488</td>
    </tr>
    <tr>
      <th>waterfront</th>
      <td>0.022750</td>
      <td>-0.038404</td>
      <td>-0.029209</td>
      <td>-0.011641</td>
      <td>0.021022</td>
      <td>-0.015131</td>
      <td>1.000000</td>
      <td>0.265469</td>
      <td>0.020465</td>
      <td>-0.021669</td>
      <td>...</td>
      <td>-0.036257</td>
      <td>-0.055569</td>
      <td>0.000292</td>
      <td>0.046424</td>
      <td>-0.005082</td>
      <td>0.047788</td>
      <td>0.009374</td>
      <td>0.036551</td>
      <td>0.006884</td>
      <td>-0.031985</td>
    </tr>
    <tr>
      <th>view</th>
      <td>0.126625</td>
      <td>0.008418</td>
      <td>0.038959</td>
      <td>0.109977</td>
      <td>0.105113</td>
      <td>-0.030150</td>
      <td>0.265469</td>
      <td>1.000000</td>
      <td>0.021241</td>
      <td>0.075946</td>
      <td>...</td>
      <td>-0.073858</td>
      <td>-0.053688</td>
      <td>0.138863</td>
      <td>0.106203</td>
      <td>0.010493</td>
      <td>0.016915</td>
      <td>0.093483</td>
      <td>0.060348</td>
      <td>0.036357</td>
      <td>-0.040835</td>
    </tr>
    <tr>
      <th>condition</th>
      <td>-0.003818</td>
      <td>0.020562</td>
      <td>-0.161222</td>
      <td>-0.076913</td>
      <td>0.015127</td>
      <td>-0.291827</td>
      <td>0.020465</td>
      <td>0.021241</td>
      <td>1.000000</td>
      <td>-0.198612</td>
      <td>...</td>
      <td>-0.035918</td>
      <td>-0.063308</td>
      <td>-0.134264</td>
      <td>0.024193</td>
      <td>-0.047841</td>
      <td>-0.044530</td>
      <td>0.086716</td>
      <td>0.332479</td>
      <td>-0.180666</td>
      <td>-0.204763</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>0.451436</td>
      <td>0.256539</td>
      <td>0.557766</td>
      <td>0.586257</td>
      <td>0.034926</td>
      <td>0.424865</td>
      <td>-0.021669</td>
      <td>0.075946</td>
      <td>-0.198612</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.035509</td>
      <td>0.172615</td>
      <td>0.559089</td>
      <td>0.041749</td>
      <td>-0.030212</td>
      <td>-0.047129</td>
      <td>0.015753</td>
      <td>-0.536056</td>
      <td>0.352979</td>
      <td>0.261955</td>
    </tr>
    <tr>
      <th>sqft_above</th>
      <td>0.318006</td>
      <td>0.449544</td>
      <td>0.589683</td>
      <td>0.811530</td>
      <td>0.127287</td>
      <td>0.493237</td>
      <td>-0.017667</td>
      <td>0.034585</td>
      <td>-0.176216</td>
      <td>0.592673</td>
      <td>...</td>
      <td>-0.206645</td>
      <td>0.356559</td>
      <td>0.652220</td>
      <td>0.138676</td>
      <td>-0.017015</td>
      <td>-0.007929</td>
      <td>-0.311896</td>
      <td>-0.463000</td>
      <td>0.202572</td>
      <td>0.050999</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>0.196807</td>
      <td>0.276462</td>
      <td>0.188410</td>
      <td>0.397274</td>
      <td>0.020780</td>
      <td>-0.319435</td>
      <td>0.008740</td>
      <td>0.127365</td>
      <td>0.142490</td>
      <td>0.051224</td>
      <td>...</td>
      <td>0.084389</td>
      <td>-0.129310</td>
      <td>0.118294</td>
      <td>0.020077</td>
      <td>-0.013618</td>
      <td>0.010215</td>
      <td>0.840211</td>
      <td>0.131106</td>
      <td>-0.013363</td>
      <td>0.003994</td>
    </tr>
    <tr>
      <th>yr_built</th>
      <td>0.040321</td>
      <td>0.162717</td>
      <td>0.593987</td>
      <td>0.353469</td>
      <td>0.005088</td>
      <td>0.547670</td>
      <td>-0.036636</td>
      <td>-0.060177</td>
      <td>-0.333284</td>
      <td>0.535577</td>
      <td>...</td>
      <td>-0.193605</td>
      <td>0.365274</td>
      <td>0.360170</td>
      <td>0.022660</td>
      <td>0.006001</td>
      <td>-0.179857</td>
      <td>-0.130128</td>
      <td>-0.999863</td>
      <td>0.465860</td>
      <td>0.379680</td>
    </tr>
    <tr>
      <th>yr_renovated</th>
      <td>0.027449</td>
      <td>-0.008372</td>
      <td>-0.012700</td>
      <td>-0.001082</td>
      <td>0.018037</td>
      <td>-0.021295</td>
      <td>0.047760</td>
      <td>0.016721</td>
      <td>-0.044898</td>
      <td>-0.046896</td>
      <td>...</td>
      <td>0.008549</td>
      <td>-0.032615</td>
      <td>-0.043865</td>
      <td>0.016889</td>
      <td>-0.017308</td>
      <td>0.999961</td>
      <td>0.001082</td>
      <td>0.179240</td>
      <td>0.002075</td>
      <td>-0.025775</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>0.479098</td>
      <td>-0.107569</td>
      <td>-0.112134</td>
      <td>-0.143029</td>
      <td>-0.107369</td>
      <td>-0.017537</td>
      <td>-0.036257</td>
      <td>-0.073858</td>
      <td>-0.035918</td>
      <td>-0.035509</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.160966</td>
      <td>-0.117312</td>
      <td>-0.108887</td>
      <td>-0.036913</td>
      <td>0.008618</td>
      <td>0.128388</td>
      <td>0.192987</td>
      <td>-0.005270</td>
      <td>0.148259</td>
    </tr>
    <tr>
      <th>long</th>
      <td>0.054300</td>
      <td>0.132547</td>
      <td>0.232980</td>
      <td>0.256092</td>
      <td>0.217447</td>
      <td>0.105360</td>
      <td>-0.055569</td>
      <td>-0.053688</td>
      <td>-0.063308</td>
      <td>0.172615</td>
      <td>...</td>
      <td>-0.160966</td>
      <td>1.000000</td>
      <td>0.334266</td>
      <td>0.235019</td>
      <td>0.001415</td>
      <td>-0.032706</td>
      <td>-0.201602</td>
      <td>-0.365236</td>
      <td>0.109482</td>
      <td>-0.188128</td>
    </tr>
    <tr>
      <th>sqft_living15</th>
      <td>0.382913</td>
      <td>0.342631</td>
      <td>0.478078</td>
      <td>0.682108</td>
      <td>0.152715</td>
      <td>0.204379</td>
      <td>0.000292</td>
      <td>0.138863</td>
      <td>-0.134264</td>
      <td>0.559089</td>
      <td>...</td>
      <td>-0.117312</td>
      <td>0.334266</td>
      <td>1.000000</td>
      <td>0.177382</td>
      <td>-0.007956</td>
      <td>-0.043839</td>
      <td>0.012714</td>
      <td>-0.360288</td>
      <td>0.178566</td>
      <td>-0.069581</td>
    </tr>
    <tr>
      <th>sqft_lot15</th>
      <td>0.067769</td>
      <td>0.015451</td>
      <td>0.024263</td>
      <td>0.143713</td>
      <td>0.712409</td>
      <td>-0.057095</td>
      <td>0.046424</td>
      <td>0.106203</td>
      <td>0.024193</td>
      <td>0.041749</td>
      <td>...</td>
      <td>-0.108887</td>
      <td>0.235019</td>
      <td>0.177382</td>
      <td>1.000000</td>
      <td>-0.007131</td>
      <td>0.016936</td>
      <td>-0.026097</td>
      <td>-0.022777</td>
      <td>0.009874</td>
      <td>-0.272304</td>
    </tr>
    <tr>
      <th>yr_sold</th>
      <td>0.007087</td>
      <td>-0.009127</td>
      <td>-0.027191</td>
      <td>-0.024446</td>
      <td>-0.006834</td>
      <td>-0.018975</td>
      <td>-0.005082</td>
      <td>0.010493</td>
      <td>-0.047841</td>
      <td>-0.030212</td>
      <td>...</td>
      <td>-0.036913</td>
      <td>0.001415</td>
      <td>-0.007956</td>
      <td>-0.007131</td>
      <td>1.000000</td>
      <td>-0.017311</td>
      <td>-0.010915</td>
      <td>0.010547</td>
      <td>-0.023659</td>
      <td>-0.013797</td>
    </tr>
    <tr>
      <th>renovated</th>
      <td>0.027333</td>
      <td>-0.008531</td>
      <td>-0.013034</td>
      <td>-0.001096</td>
      <td>0.018080</td>
      <td>-0.021294</td>
      <td>0.047788</td>
      <td>0.016915</td>
      <td>-0.044530</td>
      <td>-0.047129</td>
      <td>...</td>
      <td>0.008618</td>
      <td>-0.032706</td>
      <td>-0.043839</td>
      <td>0.016936</td>
      <td>-0.017311</td>
      <td>1.000000</td>
      <td>0.000975</td>
      <td>0.179564</td>
      <td>0.001910</td>
      <td>-0.025858</td>
    </tr>
    <tr>
      <th>basement_present</th>
      <td>0.174533</td>
      <td>0.138967</td>
      <td>0.121556</td>
      <td>0.200849</td>
      <td>-0.020799</td>
      <td>-0.292454</td>
      <td>0.009374</td>
      <td>0.093483</td>
      <td>0.086716</td>
      <td>0.015753</td>
      <td>...</td>
      <td>0.128388</td>
      <td>-0.201602</td>
      <td>0.012714</td>
      <td>-0.026097</td>
      <td>-0.010915</td>
      <td>0.000975</td>
      <td>1.000000</td>
      <td>0.129942</td>
      <td>0.038491</td>
      <td>0.116806</td>
    </tr>
    <tr>
      <th>actual_age_of_property</th>
      <td>-0.040202</td>
      <td>-0.162862</td>
      <td>-0.594414</td>
      <td>-0.353860</td>
      <td>-0.005201</td>
      <td>-0.547963</td>
      <td>0.036551</td>
      <td>0.060348</td>
      <td>0.332479</td>
      <td>-0.536056</td>
      <td>...</td>
      <td>0.192987</td>
      <td>-0.365236</td>
      <td>-0.360288</td>
      <td>-0.022777</td>
      <td>0.010547</td>
      <td>0.179564</td>
      <td>0.129942</td>
      <td>1.000000</td>
      <td>-0.466234</td>
      <td>-0.379894</td>
    </tr>
    <tr>
      <th>bathrooms_per_bedroom</th>
      <td>0.184907</td>
      <td>-0.306894</td>
      <td>0.638663</td>
      <td>0.181322</td>
      <td>0.005793</td>
      <td>0.435363</td>
      <td>0.006884</td>
      <td>0.036357</td>
      <td>-0.180666</td>
      <td>0.352979</td>
      <td>...</td>
      <td>-0.005270</td>
      <td>0.109482</td>
      <td>0.178566</td>
      <td>0.009874</td>
      <td>-0.023659</td>
      <td>0.001910</td>
      <td>0.038491</td>
      <td>-0.466234</td>
      <td>1.000000</td>
      <td>0.401996</td>
    </tr>
    <tr>
      <th>sqft_living_to_sqft_lot</th>
      <td>0.188304</td>
      <td>-0.024121</td>
      <td>0.328013</td>
      <td>0.050878</td>
      <td>-0.257394</td>
      <td>0.610488</td>
      <td>-0.031985</td>
      <td>-0.040835</td>
      <td>-0.204763</td>
      <td>0.261955</td>
      <td>...</td>
      <td>0.148259</td>
      <td>-0.188128</td>
      <td>-0.069581</td>
      <td>-0.272304</td>
      <td>-0.013797</td>
      <td>-0.025858</td>
      <td>0.116806</td>
      <td>-0.379894</td>
      <td>0.401996</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>24 rows × 24 columns</p>
</div>




```python
df['bedrooms'].value_counts().plot(kind='bar')
sns.despine;
```


    
![png](output_51_0.png)
    



```python
df['bathrooms'].value_counts().plot(kind='bar')
sns.despine;
```


    
![png](output_52_0.png)
    



```python
df['sqft_living'].value_counts().plot(kind='bar')
sns.despine;
```


    
![png](output_53_0.png)
    



```python
sns.scatterplot(data=df, x='bedrooms', y='price')
```




    <AxesSubplot:xlabel='bedrooms', ylabel='price'>




    
![png](output_54_1.png)
    



```python
sns.scatterplot(data=df, x='bathrooms', y='price')
```




    <AxesSubplot:xlabel='bathrooms', ylabel='price'>




    
![png](output_55_1.png)
    



```python
sns.scatterplot(data=df, x='sqft_living', y='price')
```




    <AxesSubplot:xlabel='sqft_living', ylabel='price'>




    
![png](output_56_1.png)
    


**After seeing outliers in the data, refine data set to:** <br>
**1. Bedrooms to 6 or less** <br>
**2. sqft_living to 4000 or less**


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>1.599300e+04</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15949.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>...</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
      <td>15993.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>400987.591009</td>
      <td>3.244982</td>
      <td>1.953667</td>
      <td>1800.124742</td>
      <td>1.317614e+04</td>
      <td>1.431533</td>
      <td>0.001438</td>
      <td>0.109411</td>
      <td>3.397174</td>
      <td>7.301132</td>
      <td>...</td>
      <td>47.543802</td>
      <td>-122.217163</td>
      <td>1792.705246</td>
      <td>11677.541987</td>
      <td>2014.323517</td>
      <td>0.024136</td>
      <td>0.359720</td>
      <td>43.468142</td>
      <td>0.619331</td>
      <td>0.312434</td>
    </tr>
    <tr>
      <th>std</th>
      <td>123893.841894</td>
      <td>0.886582</td>
      <td>0.660056</td>
      <td>631.746037</td>
      <td>3.310214e+04</td>
      <td>0.537112</td>
      <td>0.037897</td>
      <td>0.500945</td>
      <td>0.634151</td>
      <td>0.831989</td>
      <td>...</td>
      <td>0.148391</td>
      <td>0.141616</td>
      <td>496.582637</td>
      <td>24240.185920</td>
      <td>0.467833</td>
      <td>0.153475</td>
      <td>0.479933</td>
      <td>28.270545</td>
      <td>0.206016</td>
      <td>0.271903</td>
    </tr>
    <tr>
      <th>min</th>
      <td>175000.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.720000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.053030</td>
      <td>0.000610</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299950.000000</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>1330.000000</td>
      <td>5.000000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.431000</td>
      <td>-122.332000</td>
      <td>1430.000000</td>
      <td>5040.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>0.500000</td>
      <td>0.152857</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>392000.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1720.000000</td>
      <td>7.439000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.545900</td>
      <td>-122.247000</td>
      <td>1710.000000</td>
      <td>7500.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>41.000000</td>
      <td>0.583333</td>
      <td>0.232076</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>499990.000000</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2180.000000</td>
      <td>9.968000e+03</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>47.682700</td>
      <td>-122.133000</td>
      <td>2090.000000</td>
      <td>9601.000000</td>
      <td>2015.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>62.000000</td>
      <td>0.750000</td>
      <td>0.370000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>650000.000000</td>
      <td>33.000000</td>
      <td>7.500000</td>
      <td>5461.000000</td>
      <td>1.164794e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>...</td>
      <td>47.777600</td>
      <td>-121.319000</td>
      <td>4362.000000</td>
      <td>438213.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>115.000000</td>
      <td>2.500000</td>
      <td>2.291399</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 24 columns</p>
</div>




```python
df = df[df['bedrooms'] <= 6]
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>1.596500e+04</td>
      <td>15965.00000</td>
      <td>15965.000000</td>
      <td>15921.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>...</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
      <td>15965.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>400835.351519</td>
      <td>3.235766</td>
      <td>1.951425</td>
      <td>1797.901973</td>
      <td>1.318139e+04</td>
      <td>1.43135</td>
      <td>0.001441</td>
      <td>0.109415</td>
      <td>3.397244</td>
      <td>7.301159</td>
      <td>...</td>
      <td>47.543681</td>
      <td>-122.217090</td>
      <td>1792.784654</td>
      <td>11683.813028</td>
      <td>2014.323771</td>
      <td>0.024053</td>
      <td>0.359349</td>
      <td>43.454056</td>
      <td>0.619678</td>
      <td>0.312278</td>
    </tr>
    <tr>
      <th>std</th>
      <td>123867.266044</td>
      <td>0.835528</td>
      <td>0.656015</td>
      <td>629.416330</td>
      <td>3.312958e+04</td>
      <td>0.53716</td>
      <td>0.037930</td>
      <td>0.501092</td>
      <td>0.634289</td>
      <td>0.832428</td>
      <td>...</td>
      <td>0.148420</td>
      <td>0.141673</td>
      <td>496.817412</td>
      <td>24259.422399</td>
      <td>0.467928</td>
      <td>0.153217</td>
      <td>0.479825</td>
      <td>28.274746</td>
      <td>0.205890</td>
      <td>0.271962</td>
    </tr>
    <tr>
      <th>min</th>
      <td>175000.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.720000e+02</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.125000</td>
      <td>0.000610</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299900.000000</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>1330.000000</td>
      <td>5.000000e+03</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.430300</td>
      <td>-122.332000</td>
      <td>1430.000000</td>
      <td>5040.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>0.500000</td>
      <td>0.152703</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>392000.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1720.000000</td>
      <td>7.434000e+03</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.545800</td>
      <td>-122.247000</td>
      <td>1710.000000</td>
      <td>7500.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>41.000000</td>
      <td>0.583333</td>
      <td>0.231818</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>499950.000000</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2180.000000</td>
      <td>9.966000e+03</td>
      <td>2.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>47.682700</td>
      <td>-122.133000</td>
      <td>2091.000000</td>
      <td>9603.000000</td>
      <td>2015.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>62.000000</td>
      <td>0.750000</td>
      <td>0.369718</td>
    </tr>
    <tr>
      <th>max</th>
      <td>650000.000000</td>
      <td>6.000000</td>
      <td>5.250000</td>
      <td>5461.000000</td>
      <td>1.164794e+06</td>
      <td>3.50000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>...</td>
      <td>47.777600</td>
      <td>-121.319000</td>
      <td>4362.000000</td>
      <td>438213.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>115.000000</td>
      <td>2.500000</td>
      <td>2.291399</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 24 columns</p>
</div>




```python
df = df[df['sqft_living'] <= 4000]
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>1.591600e+04</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15872.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>...</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
      <td>15916.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>400358.968145</td>
      <td>3.232093</td>
      <td>1.947427</td>
      <td>1789.922971</td>
      <td>1.310122e+04</td>
      <td>1.430227</td>
      <td>0.001445</td>
      <td>0.107737</td>
      <td>3.397525</td>
      <td>7.295614</td>
      <td>...</td>
      <td>47.543950</td>
      <td>-122.217231</td>
      <td>1789.790211</td>
      <td>11616.665117</td>
      <td>2014.323699</td>
      <td>0.024001</td>
      <td>0.358193</td>
      <td>43.512189</td>
      <td>0.619281</td>
      <td>0.312295</td>
    </tr>
    <tr>
      <th>std</th>
      <td>123707.148968</td>
      <td>0.833128</td>
      <td>0.652116</td>
      <td>613.395670</td>
      <td>3.301546e+04</td>
      <td>0.536934</td>
      <td>0.037988</td>
      <td>0.496587</td>
      <td>0.634433</td>
      <td>0.825913</td>
      <td>...</td>
      <td>0.148286</td>
      <td>0.141678</td>
      <td>493.349512</td>
      <td>24075.034143</td>
      <td>0.467902</td>
      <td>0.153057</td>
      <td>0.479484</td>
      <td>28.280040</td>
      <td>0.205886</td>
      <td>0.271939</td>
    </tr>
    <tr>
      <th>min</th>
      <td>175000.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.720000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.125000</td>
      <td>0.000610</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299725.000000</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>1330.000000</td>
      <td>5.000000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.431400</td>
      <td>-122.332000</td>
      <td>1430.000000</td>
      <td>5040.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>0.500000</td>
      <td>0.152916</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>390000.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1720.000000</td>
      <td>7.420000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>47.546150</td>
      <td>-122.248000</td>
      <td>1710.000000</td>
      <td>7500.000000</td>
      <td>2014.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>41.000000</td>
      <td>0.583333</td>
      <td>0.231807</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>499900.000000</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2180.000000</td>
      <td>9.936000e+03</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>47.682700</td>
      <td>-122.133000</td>
      <td>2090.000000</td>
      <td>9600.000000</td>
      <td>2015.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>62.000000</td>
      <td>0.750000</td>
      <td>0.369323</td>
    </tr>
    <tr>
      <th>max</th>
      <td>650000.000000</td>
      <td>6.000000</td>
      <td>5.250000</td>
      <td>4000.000000</td>
      <td>1.164794e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>...</td>
      <td>47.777600</td>
      <td>-121.319000</td>
      <td>4050.000000</td>
      <td>438213.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>115.000000</td>
      <td>2.500000</td>
      <td>2.291399</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 24 columns</p>
</div>




```python
df['bedrooms'].value_counts().plot(kind='bar')
sns.despine;
```


    
![png](output_61_0.png)
    



```python
df['bathrooms'].value_counts().plot(kind='bar')
sns.despine;
```


    
![png](output_62_0.png)
    



```python
df['sqft_living'].value_counts().plot(kind='bar')
sns.despine;
```


    
![png](output_63_0.png)
    



```python
df['zipcode'].value_counts().plot(kind='bar')
sns.despine;
```


    
![png](output_64_0.png)
    



```python
#pd.set_option("display.max_rows", None, "display.max_columns", None)
```


```python
df['zipcode'].value_counts(ascending=True)
```




    98040     19
    98004     23
    98109     35
    98102     40
    98005     47
            ... 
    98023    466
    98034    469
    98133    475
    98042    516
    98038    559
    Name: zipcode, Length: 69, dtype: int64




```python
# For Future Work: explore correlation between these zip codes and price 
# least_houses_zip = df[df['zipcode' == '98004', '98109', '98112', '98102', '98119']]
# most_houses_zips = 98023, 98034, 98133, 98042, 98038
```


```python
# For future work on zipcodes

plt.figure(figsize=(12,12))
sns.jointplot(x=df['lat'], y=df['long'], size=12)
plt.xlabel('Latitude', fontsize=11)
plt.ylabel('Longitude', fontsize=11)
plt.show()
sns.despine;
```

    /Users/v/opt/anaconda3/envs/learn-env/lib/python3.8/site-packages/seaborn/axisgrid.py:2015: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)



    <Figure size 864x864 with 0 Axes>



    
![png](output_68_2.png)
    



```python
# Boxplot for price 

fig, ax = plt.subplots(figsize=(12,8))
sns.boxplot(x='price', data=df, ax=ax)
```




    <AxesSubplot:xlabel='price'>




```python
# Boxplot for bedrooms 

fig, ax = plt.subplots(figsize=(12,8))
sns.boxplot(x='bedrooms', data=df, ax=ax)
```




    <AxesSubplot:xlabel='bedrooms'>

    



```python
# Scatterplot for sqft_living and price

sns.scatterplot(data=df, x='sqft_living', y='price')
```




    <AxesSubplot:xlabel='sqft_living', ylabel='price'>




```python
# Scatterplot for bedrooms and price

sns.scatterplot(data=df, x='bedrooms', y='price')
```




    <AxesSubplot:xlabel='bedrooms', ylabel='price'>




    
![png](output_72_1.png)
    



```python
# Scatterplot for bathrooms and price

sns.scatterplot(data=df, x='bathrooms', y='price')
```




    <AxesSubplot:xlabel='bathrooms', ylabel='price'>




    
![png](output_73_1.png)
    


## Data Preparation 

Start dropping columns that will not be used <br>
1. 'view'
2. 'date'


```python
# Will not use 'view' (# of times the house has been viewed) for analysis 

if 'view' in df.columns:
    df.drop('view', axis=1, inplace=True)
```


```python
# Drop date 

if 'date' in df.columns:
    df.drop('date', axis=1, inplace=True)
```

### Create Target and Explore Data with More Visualizations

**TARGET is price**


```python
# Target is price 
# X values is everything else 

TARGET = 'price'
X_VALS = [c for c in df.columns if c != TARGET]
TARGET in X_VALS
```




    False




```python
for col in df.columns:
    plt.figure(figsize=(12,8))
    sns.displot(df[col],bins=20)
    plt.title(col)
    plt.show();
```
    



```python
for col in df.columns:
    plt.scatter(df[col], df[TARGET])
    plt.title(col)
    plt.show()
```


   



```python
for col in df.columns:
    plt.hist(df[col])
    plt.title(col)
    plt.show()
```



```python
for col in df.select_dtypes('number').columns:
    plt.boxplot(df[col], vert=False)
    plt.title(col)
    plt.show()
```

    



```python
corr = df.corr().abs()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio 
# GnBu is your color preference 
sns.heatmap(corr, mask=mask, cmap="GnBu", vmin=0, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .75})
```




    <AxesSubplot:>




    
![png](output_83_1.png)
    



```python
# Create a correlation heatmap grid with data 

plt.figure(figsize=(14, 14))
corr_matrix = df.corr().abs().round(2)
sns.heatmap(data=corr_matrix,cmap="GnBu",annot=True)
```




    <AxesSubplot:>




    
![png](output_84_1.png)
    



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>yr_sold</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>actual_age_of_property</th>
      <th>bathrooms_per_bedroom</th>
      <th>sqft_living_to_sqft_lot</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>...</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>2014</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
      <td>0.333333</td>
      <td>0.208850</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>...</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>2014</td>
      <td>1</td>
      <td>1</td>
      <td>63</td>
      <td>0.750000</td>
      <td>0.354874</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>...</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>2015</td>
      <td>0</td>
      <td>0</td>
      <td>82</td>
      <td>0.500000</td>
      <td>0.077000</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>...</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>2014</td>
      <td>0</td>
      <td>1</td>
      <td>49</td>
      <td>0.750000</td>
      <td>0.392000</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>...</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>2015</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.666667</td>
      <td>0.207921</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



### Clean Data Before Modeling 
- drop sqft_basement, sqft_basement15, sqft_lot, sqft_lot15, yr_renovated
- drop longitude, latitude, 


```python
if 'sqft_basement' in df.columns:
    df.drop('sqft_basement', axis=1, inplace=True)
```


```python
if 'sqft_basement15' in df.columns:
    df.drop('sqft_basement15', axis=1, inplace=True)
```


```python
if 'sqft_lot' in df.columns:
    df.drop('sqft_lot', axis=1, inplace=True)
```


```python
if 'sqft_lot15' in df.columns:
    df.drop('sqft_lot15', axis=1, inplace=True)
```


```python
if 'sqft_living15' in df.columns:
    df.drop('sqft_living15', axis=1, inplace=True)
```


```python
if 'yr_renovated' in df.columns:
    df.drop('yr_renovated', axis=1, inplace=True)
```


```python
if 'lat' in df.columns:
    df.drop('lat', axis=1, inplace=True)
```


```python
if 'long' in df.columns:
    df.drop('long', axis=1, inplace=True)
```

**After checking for overfitting in models, drop additional features**



```python
# after checking for overfitting in models, drop additional features

#'const': 49.808021804391046,
#'bedrooms': 8.124041148684404,
#'bathrooms': 13.373648045281776,
#'sqft_living': 10.9560606782623,
#'floors': 3.316672951320918,
#'waterfront': 1.078179446917037,
#'condition': 1.2848095397229615,
#'grade': 2.273801501092483,
#'sqft_above': 11.76560610464051,
#'yr_built': inf,
#'yr_sold': inf,
#'renovated': 1.0815968469358994,
#'basement_present': 3.8066176363082005,
#'actual_age_of_property': inf,
#'bathrooms_per_bedroom': 10.77082956774539,
#'sqft_living_to_sqft_lot': 3.1142842893817773,

if 'sqft_above' in df.columns:
    df.drop('sqft_above', axis=1, inplace=True)
```


```python
if 'bathrooms_per_bedroom' in df.columns:
    df.drop('bathrooms_per_bedroom', axis=1, inplace=True)
```


```python
if 'sqft_living_to_sqft_lot' in df.columns:
    df.drop('sqft_living_to_sqft_lot', axis=1, inplace=True)
```


```python
if 'yr_built' in df.columns:
    df.drop('yr_built', axis=1, inplace=True)
```


```python
if 'yr_sold' in df.columns:
    df.drop('yr_sold', axis=1, inplace=True)
```


```python
if 'actual_age_of_property' in df.columns:
    df.drop('actual_age_of_property', axis=1, inplace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>zipcode</th>
      <th>renovated</th>
      <th>basement_present</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98178</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98125</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>98028</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>98136</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>98074</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# See if correlation heatmap changes with clean data 

corr = df.corr().abs()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap="GnBu", vmin=0, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .75})
```




    <AxesSubplot:>




    
![png](output_103_1.png)
    



```python
# Create a correlation heatmap grid with data 

plt.figure(figsize=(14, 14))
corr_matrix = df.corr().abs().round(2)
sns.heatmap(data=corr_matrix,cmap="GnBu",annot=True)
```




    <AxesSubplot:>




    
![png](output_104_1.png)
    



```python
# Check data one more time 

for col in df.columns:
    plt.scatter(df[col], df[TARGET])
    plt.title(col)
    plt.show()
```


    

```python
for col in df.columns:
    plt.hist(df[col])
    plt.title(col)
    plt.show()
```


    


# Start Building Model 
- Create dependent (y) and independent (x) variables 
- Create Train and Test data subsets

## Create TARGET and Independent Variables for Model


```python
# Dependent variable(y) is price as previously defined as TARGET
# Independent variables(X) are all variables that are not price  

y = df[TARGET]
X = df.drop(columns=[TARGET])
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>zipcode</th>
      <th>renovated</th>
      <th>basement_present</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7129300520</th>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98178</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6414100192</th>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98125</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5631500400</th>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>98028</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2487200875</th>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>98136</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1954400510</th>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>98074</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>263000018</th>
      <td>3</td>
      <td>2.50</td>
      <td>1530</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>98103</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6600060120</th>
      <td>4</td>
      <td>2.50</td>
      <td>2310</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>98146</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1523300141</th>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98144</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>291310100</th>
      <td>3</td>
      <td>2.50</td>
      <td>1600</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>98027</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1523300157</th>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>98144</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>15916 rows × 10 columns</p>
</div>




```python
y.describe()
```




    count     15916.000000
    mean     400358.968145
    std      123707.148968
    min      175000.000000
    25%      299725.000000
    50%      390000.000000
    75%      499900.000000
    max      650000.000000
    Name: price, dtype: float64



## Create Train and Test Data Subsets


```python
# Create Train and Test data subsets using train_test_split
# Check shape of each data set 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=100)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((11937, 10), (3979, 10), (11937,), (3979,))




```python
# Check percentage of data that is Train data 
# Train data is 75% of data; Test data is 25% of data 

11_937/(11_937+3979)
```




    0.75




```python
# Reset index on Train and Test data 

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
```

Create Number and Category Column Variables 


```python
# Create variable for "Number" columns (integers, floats)
# Create variable for "Category" columns (objects, strings)
# Check Category Columns

NUMBER_COLS = X_train.select_dtypes('number').columns
CATEGORY_COLS = X_train.select_dtypes('object').columns
CATEGORY_COLS
```




    Index(['zipcode'], dtype='object')



## One Hot Encode Category Columns (zipcode)


```python
# ONE HOT ENCODE 
# zipcode is the only category column 
# One Hot Encode zipcodes so you can use the data in model 

ohe = OneHotEncoder(drop='first', sparse=False)
X_train_ohe = ohe.fit_transform(X_train[CATEGORY_COLS])
X_test_ohe = ohe.transform(X_test[CATEGORY_COLS])

X_train_ohe = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names(CATEGORY_COLS))
X_test_ohe = pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names(CATEGORY_COLS))

X_train_ohe.columns = [c.lower() for c in X_train_ohe]
X_test_ohe.columns = [c.lower() for c in X_test_ohe]
```


```python
# Check one hot encoding of zipcodes

X_train_ohe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>...</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 68 columns</p>
</div>




```python
# Concatenate Number Columns with One Hot Encoded Columns 

X_train_raw = pd.concat([X_train[NUMBER_COLS], 
                        X_train_ohe], 
                        axis=1)
X_test_raw = pd.concat([X_test[NUMBER_COLS], 
                        X_test_ohe], 
                        axis=1)
```

## Scale the Data


```python
# Scale the Data using StandardScaler()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[NUMBER_COLS])
X_test_scaled = scaler.transform(X_test[NUMBER_COLS])

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train[NUMBER_COLS].columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test[NUMBER_COLS].columns)
```


```python
# Check the shape of the data

X_train_scaled.shape, X_test_scaled.shape
```




    ((11937, 9), (3979, 9))




```python
# Concatenate Scaled data with One Hot Encoded data

X_train_scaled = pd.concat([X_train_scaled, 
                            X_train_ohe], 
                            axis=1)
X_test_scaled = pd.concat([X_test_scaled, 
                           X_test_ohe], 
                           axis=1)
```


```python
# Check X_train_scaled data

X_train_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>renovated</th>
      <th>basement_present</th>
      <th>zipcode_98002</th>
      <th>...</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.484449</td>
      <td>-1.457646</td>
      <td>-1.647388</td>
      <td>-0.796457</td>
      <td>-0.034267</td>
      <td>-0.633146</td>
      <td>-1.566432</td>
      <td>-0.153275</td>
      <td>-0.752412</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.914667</td>
      <td>0.465379</td>
      <td>1.044130</td>
      <td>1.066624</td>
      <td>-0.034267</td>
      <td>-0.633146</td>
      <td>0.855183</td>
      <td>-0.153275</td>
      <td>-0.752412</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.914667</td>
      <td>0.849984</td>
      <td>0.489515</td>
      <td>-0.796457</td>
      <td>-0.034267</td>
      <td>2.512689</td>
      <td>0.855183</td>
      <td>-0.153275</td>
      <td>1.329058</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.914667</td>
      <td>-1.457646</td>
      <td>-1.272207</td>
      <td>-0.796457</td>
      <td>-0.034267</td>
      <td>0.939771</td>
      <td>-1.566432</td>
      <td>-0.153275</td>
      <td>-0.752412</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.484449</td>
      <td>-1.457646</td>
      <td>-0.652342</td>
      <td>-0.796457</td>
      <td>-0.034267</td>
      <td>-0.633146</td>
      <td>-0.355625</td>
      <td>-0.153275</td>
      <td>1.329058</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11932</th>
      <td>2.114224</td>
      <td>1.234589</td>
      <td>2.185987</td>
      <td>-0.796457</td>
      <td>-0.034267</td>
      <td>0.939771</td>
      <td>0.855183</td>
      <td>-0.153275</td>
      <td>1.329058</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11933</th>
      <td>2.114224</td>
      <td>0.849984</td>
      <td>2.626417</td>
      <td>-0.796457</td>
      <td>-0.034267</td>
      <td>0.939771</td>
      <td>0.855183</td>
      <td>-0.153275</td>
      <td>1.329058</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11934</th>
      <td>-0.284891</td>
      <td>0.465379</td>
      <td>-0.211911</td>
      <td>-0.796457</td>
      <td>-0.034267</td>
      <td>0.939771</td>
      <td>0.855183</td>
      <td>-0.153275</td>
      <td>-0.752412</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11935</th>
      <td>-0.284891</td>
      <td>0.465379</td>
      <td>-0.521844</td>
      <td>1.066624</td>
      <td>-0.034267</td>
      <td>-0.633146</td>
      <td>0.855183</td>
      <td>-0.153275</td>
      <td>1.329058</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11936</th>
      <td>-0.284891</td>
      <td>-0.303831</td>
      <td>-0.472907</td>
      <td>-0.796457</td>
      <td>-0.034267</td>
      <td>-0.633146</td>
      <td>-0.355625</td>
      <td>-0.153275</td>
      <td>-0.752412</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>11937 rows × 77 columns</p>
</div>



## Add an Intercept


```python
# Add an intercept with sm.add_constant()
# The b in y = mx + b

X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)
```


```python
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)
```

# MODELS

## Model 1: Everything 
- Used for exploration 


```python
model1 = sm.OLS(y_train, X_train_scaled).fit()
model1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.742</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.740</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   442.0</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 07 Nov 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:32:59</td>     <th>  Log-Likelihood:    </th> <td>-1.4886e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 11937</td>      <th>  AIC:               </th>  <td>2.979e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 11859</td>      <th>  BIC:               </th>  <td>2.985e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    77</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>            <td> 2.735e+05</td> <td> 4044.287</td> <td>   67.617</td> <td> 0.000</td> <td> 2.66e+05</td> <td> 2.81e+05</td>
</tr>
<tr>
  <th>bedrooms</th>         <td>-3777.9235</td> <td>  766.377</td> <td>   -4.930</td> <td> 0.000</td> <td>-5280.148</td> <td>-2275.698</td>
</tr>
<tr>
  <th>bathrooms</th>        <td> 4006.3214</td> <td>  919.168</td> <td>    4.359</td> <td> 0.000</td> <td> 2204.602</td> <td> 5808.041</td>
</tr>
<tr>
  <th>sqft_living</th>      <td> 5.751e+04</td> <td> 1008.240</td> <td>   57.044</td> <td> 0.000</td> <td> 5.55e+04</td> <td> 5.95e+04</td>
</tr>
<tr>
  <th>floors</th>           <td>-7602.7410</td> <td>  836.628</td> <td>   -9.087</td> <td> 0.000</td> <td>-9242.669</td> <td>-5962.812</td>
</tr>
<tr>
  <th>waterfront</th>       <td> 5088.3292</td> <td>  600.906</td> <td>    8.468</td> <td> 0.000</td> <td> 3910.454</td> <td> 6266.204</td>
</tr>
<tr>
  <th>condition</th>        <td> 9541.0686</td> <td>  629.450</td> <td>   15.158</td> <td> 0.000</td> <td> 8307.244</td> <td> 1.08e+04</td>
</tr>
<tr>
  <th>grade</th>            <td> 2.607e+04</td> <td>  818.004</td> <td>   31.876</td> <td> 0.000</td> <td> 2.45e+04</td> <td> 2.77e+04</td>
</tr>
<tr>
  <th>renovated</th>        <td> 1469.4406</td> <td>  585.502</td> <td>    2.510</td> <td> 0.012</td> <td>  321.761</td> <td> 2617.121</td>
</tr>
<tr>
  <th>basement_present</th> <td>-5546.0095</td> <td>  721.806</td> <td>   -7.684</td> <td> 0.000</td> <td>-6960.868</td> <td>-4131.151</td>
</tr>
<tr>
  <th>zipcode_98002</th>    <td> -1.29e+04</td> <td> 6672.443</td> <td>   -1.933</td> <td> 0.053</td> <td> -2.6e+04</td> <td>  178.832</td>
</tr>
<tr>
  <th>zipcode_98003</th>    <td> 1740.5176</td> <td> 6073.043</td> <td>    0.287</td> <td> 0.774</td> <td>-1.02e+04</td> <td> 1.36e+04</td>
</tr>
<tr>
  <th>zipcode_98004</th>    <td> 3.714e+05</td> <td> 1.63e+04</td> <td>   22.743</td> <td> 0.000</td> <td> 3.39e+05</td> <td> 4.03e+05</td>
</tr>
<tr>
  <th>zipcode_98005</th>    <td>  2.74e+05</td> <td>  1.1e+04</td> <td>   24.807</td> <td> 0.000</td> <td> 2.52e+05</td> <td> 2.96e+05</td>
</tr>
<tr>
  <th>zipcode_98006</th>    <td> 2.052e+05</td> <td> 6840.113</td> <td>   30.006</td> <td> 0.000</td> <td> 1.92e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>zipcode_98007</th>    <td> 2.173e+05</td> <td> 8409.334</td> <td>   25.836</td> <td> 0.000</td> <td> 2.01e+05</td> <td> 2.34e+05</td>
</tr>
<tr>
  <th>zipcode_98008</th>    <td> 2.181e+05</td> <td> 6419.708</td> <td>   33.974</td> <td> 0.000</td> <td> 2.06e+05</td> <td> 2.31e+05</td>
</tr>
<tr>
  <th>zipcode_98010</th>    <td> 8.115e+04</td> <td> 8837.877</td> <td>    9.182</td> <td> 0.000</td> <td> 6.38e+04</td> <td> 9.85e+04</td>
</tr>
<tr>
  <th>zipcode_98011</th>    <td> 1.529e+05</td> <td> 6972.915</td> <td>   21.932</td> <td> 0.000</td> <td> 1.39e+05</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>zipcode_98014</th>    <td> 1.097e+05</td> <td> 8779.574</td> <td>   12.491</td> <td> 0.000</td> <td> 9.25e+04</td> <td> 1.27e+05</td>
</tr>
<tr>
  <th>zipcode_98019</th>    <td> 1.098e+05</td> <td> 6680.290</td> <td>   16.431</td> <td> 0.000</td> <td> 9.67e+04</td> <td> 1.23e+05</td>
</tr>
<tr>
  <th>zipcode_98022</th>    <td> 2.712e+04</td> <td> 6326.221</td> <td>    4.288</td> <td> 0.000</td> <td> 1.47e+04</td> <td> 3.95e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>    <td>-1.575e+04</td> <td> 5246.049</td> <td>   -3.002</td> <td> 0.003</td> <td> -2.6e+04</td> <td>-5466.508</td>
</tr>
<tr>
  <th>zipcode_98024</th>    <td> 1.462e+05</td> <td> 1.03e+04</td> <td>   14.240</td> <td> 0.000</td> <td> 1.26e+05</td> <td> 1.66e+05</td>
</tr>
<tr>
  <th>zipcode_98027</th>    <td> 1.763e+05</td> <td> 5977.162</td> <td>   29.489</td> <td> 0.000</td> <td> 1.65e+05</td> <td> 1.88e+05</td>
</tr>
<tr>
  <th>zipcode_98028</th>    <td> 1.363e+05</td> <td> 6044.350</td> <td>   22.553</td> <td> 0.000</td> <td> 1.24e+05</td> <td> 1.48e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th>    <td> 2.123e+05</td> <td> 6367.167</td> <td>   33.347</td> <td> 0.000</td> <td>    2e+05</td> <td> 2.25e+05</td>
</tr>
<tr>
  <th>zipcode_98030</th>    <td> 7039.3493</td> <td> 6236.424</td> <td>    1.129</td> <td> 0.259</td> <td>-5185.064</td> <td> 1.93e+04</td>
</tr>
<tr>
  <th>zipcode_98031</th>    <td> 8528.6375</td> <td> 5947.097</td> <td>    1.434</td> <td> 0.152</td> <td>-3128.648</td> <td> 2.02e+04</td>
</tr>
<tr>
  <th>zipcode_98032</th>    <td>-1.223e+04</td> <td> 8048.185</td> <td>   -1.519</td> <td> 0.129</td> <td> -2.8e+04</td> <td> 3547.731</td>
</tr>
<tr>
  <th>zipcode_98033</th>    <td> 2.437e+05</td> <td> 6552.817</td> <td>   37.187</td> <td> 0.000</td> <td> 2.31e+05</td> <td> 2.57e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th>    <td> 1.645e+05</td> <td> 5296.904</td> <td>   31.057</td> <td> 0.000</td> <td> 1.54e+05</td> <td> 1.75e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th>    <td>  4.63e+04</td> <td> 5160.252</td> <td>    8.973</td> <td> 0.000</td> <td> 3.62e+04</td> <td> 5.64e+04</td>
</tr>
<tr>
  <th>zipcode_98040</th>    <td>  3.26e+05</td> <td> 1.59e+04</td> <td>   20.537</td> <td> 0.000</td> <td> 2.95e+05</td> <td> 3.57e+05</td>
</tr>
<tr>
  <th>zipcode_98042</th>    <td> 1.271e+04</td> <td> 5169.826</td> <td>    2.458</td> <td> 0.014</td> <td> 2571.503</td> <td> 2.28e+04</td>
</tr>
<tr>
  <th>zipcode_98045</th>    <td> 9.882e+04</td> <td> 6641.193</td> <td>   14.880</td> <td> 0.000</td> <td> 8.58e+04</td> <td> 1.12e+05</td>
</tr>
<tr>
  <th>zipcode_98052</th>    <td> 2.154e+05</td> <td> 5726.400</td> <td>   37.616</td> <td> 0.000</td> <td> 2.04e+05</td> <td> 2.27e+05</td>
</tr>
<tr>
  <th>zipcode_98053</th>    <td> 2.064e+05</td> <td> 6463.692</td> <td>   31.929</td> <td> 0.000</td> <td> 1.94e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>    <td> 3.959e+04</td> <td> 6104.436</td> <td>    6.485</td> <td> 0.000</td> <td> 2.76e+04</td> <td> 5.16e+04</td>
</tr>
<tr>
  <th>zipcode_98056</th>    <td> 8.944e+04</td> <td> 5560.338</td> <td>   16.086</td> <td> 0.000</td> <td> 7.85e+04</td> <td>    1e+05</td>
</tr>
<tr>
  <th>zipcode_98058</th>    <td> 4.089e+04</td> <td> 5385.866</td> <td>    7.593</td> <td> 0.000</td> <td> 3.03e+04</td> <td> 5.15e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>    <td> 9.496e+04</td> <td> 5502.125</td> <td>   17.258</td> <td> 0.000</td> <td> 8.42e+04</td> <td> 1.06e+05</td>
</tr>
<tr>
  <th>zipcode_98065</th>    <td>  1.48e+05</td> <td> 6142.417</td> <td>   24.090</td> <td> 0.000</td> <td> 1.36e+05</td> <td>  1.6e+05</td>
</tr>
<tr>
  <th>zipcode_98070</th>    <td> 1.408e+05</td> <td> 8714.996</td> <td>   16.152</td> <td> 0.000</td> <td> 1.24e+05</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>zipcode_98072</th>    <td> 1.642e+05</td> <td> 6510.426</td> <td>   25.221</td> <td> 0.000</td> <td> 1.51e+05</td> <td> 1.77e+05</td>
</tr>
<tr>
  <th>zipcode_98074</th>    <td> 2.013e+05</td> <td> 6243.559</td> <td>   32.239</td> <td> 0.000</td> <td> 1.89e+05</td> <td> 2.14e+05</td>
</tr>
<tr>
  <th>zipcode_98075</th>    <td> 2.343e+05</td> <td> 8190.912</td> <td>   28.608</td> <td> 0.000</td> <td> 2.18e+05</td> <td>  2.5e+05</td>
</tr>
<tr>
  <th>zipcode_98077</th>    <td> 1.627e+05</td> <td> 8442.365</td> <td>   19.277</td> <td> 0.000</td> <td> 1.46e+05</td> <td> 1.79e+05</td>
</tr>
<tr>
  <th>zipcode_98092</th>    <td> 1469.8656</td> <td> 5764.394</td> <td>    0.255</td> <td> 0.799</td> <td>-9829.292</td> <td> 1.28e+04</td>
</tr>
<tr>
  <th>zipcode_98102</th>    <td> 2.971e+05</td> <td> 1.17e+04</td> <td>   25.400</td> <td> 0.000</td> <td> 2.74e+05</td> <td>  3.2e+05</td>
</tr>
<tr>
  <th>zipcode_98103</th>    <td> 2.441e+05</td> <td> 5508.917</td> <td>   44.306</td> <td> 0.000</td> <td> 2.33e+05</td> <td> 2.55e+05</td>
</tr>
<tr>
  <th>zipcode_98105</th>    <td> 3.051e+05</td> <td> 8436.754</td> <td>   36.162</td> <td> 0.000</td> <td> 2.89e+05</td> <td> 3.22e+05</td>
</tr>
<tr>
  <th>zipcode_98106</th>    <td> 9.325e+04</td> <td> 5853.373</td> <td>   15.931</td> <td> 0.000</td> <td> 8.18e+04</td> <td> 1.05e+05</td>
</tr>
<tr>
  <th>zipcode_98107</th>    <td> 2.819e+05</td> <td> 6487.600</td> <td>   43.455</td> <td> 0.000</td> <td> 2.69e+05</td> <td> 2.95e+05</td>
</tr>
<tr>
  <th>zipcode_98108</th>    <td> 1.028e+05</td> <td> 6800.267</td> <td>   15.124</td> <td> 0.000</td> <td> 8.95e+04</td> <td> 1.16e+05</td>
</tr>
<tr>
  <th>zipcode_98109</th>    <td> 3.119e+05</td> <td> 1.29e+04</td> <td>   24.230</td> <td> 0.000</td> <td> 2.87e+05</td> <td> 3.37e+05</td>
</tr>
<tr>
  <th>zipcode_98112</th>    <td> 3.033e+05</td> <td> 9963.820</td> <td>   30.437</td> <td> 0.000</td> <td> 2.84e+05</td> <td> 3.23e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>    <td> 2.569e+05</td> <td> 5559.862</td> <td>   46.199</td> <td> 0.000</td> <td> 2.46e+05</td> <td> 2.68e+05</td>
</tr>
<tr>
  <th>zipcode_98116</th>    <td> 2.537e+05</td> <td> 6430.978</td> <td>   39.453</td> <td> 0.000</td> <td> 2.41e+05</td> <td> 2.66e+05</td>
</tr>
<tr>
  <th>zipcode_98117</th>    <td> 2.582e+05</td> <td> 5454.138</td> <td>   47.348</td> <td> 0.000</td> <td> 2.48e+05</td> <td> 2.69e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>    <td> 1.346e+05</td> <td> 5496.994</td> <td>   24.481</td> <td> 0.000</td> <td> 1.24e+05</td> <td> 1.45e+05</td>
</tr>
<tr>
  <th>zipcode_98119</th>    <td> 2.984e+05</td> <td> 9594.954</td> <td>   31.095</td> <td> 0.000</td> <td>  2.8e+05</td> <td> 3.17e+05</td>
</tr>
<tr>
  <th>zipcode_98122</th>    <td> 2.415e+05</td> <td> 6810.371</td> <td>   35.462</td> <td> 0.000</td> <td> 2.28e+05</td> <td> 2.55e+05</td>
</tr>
<tr>
  <th>zipcode_98125</th>    <td> 1.686e+05</td> <td> 5663.514</td> <td>   29.765</td> <td> 0.000</td> <td> 1.57e+05</td> <td>  1.8e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>    <td>  1.61e+05</td> <td> 5796.992</td> <td>   27.777</td> <td> 0.000</td> <td>  1.5e+05</td> <td> 1.72e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>    <td> 1.264e+05</td> <td> 5269.619</td> <td>   23.988</td> <td> 0.000</td> <td> 1.16e+05</td> <td> 1.37e+05</td>
</tr>
<tr>
  <th>zipcode_98136</th>    <td> 2.102e+05</td> <td> 6579.194</td> <td>   31.956</td> <td> 0.000</td> <td> 1.97e+05</td> <td> 2.23e+05</td>
</tr>
<tr>
  <th>zipcode_98144</th>    <td> 1.814e+05</td> <td> 6201.612</td> <td>   29.254</td> <td> 0.000</td> <td> 1.69e+05</td> <td> 1.94e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>    <td> 8.473e+04</td> <td> 6252.114</td> <td>   13.552</td> <td> 0.000</td> <td> 7.25e+04</td> <td>  9.7e+04</td>
</tr>
<tr>
  <th>zipcode_98148</th>    <td> 3.585e+04</td> <td> 1.09e+04</td> <td>    3.286</td> <td> 0.001</td> <td> 1.45e+04</td> <td> 5.72e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>    <td> 1.244e+05</td> <td> 5408.652</td> <td>   23.005</td> <td> 0.000</td> <td> 1.14e+05</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>zipcode_98166</th>    <td> 1.012e+05</td> <td> 6410.773</td> <td>   15.789</td> <td> 0.000</td> <td> 8.87e+04</td> <td> 1.14e+05</td>
</tr>
<tr>
  <th>zipcode_98168</th>    <td>   2.8e+04</td> <td> 6444.679</td> <td>    4.345</td> <td> 0.000</td> <td> 1.54e+04</td> <td> 4.06e+04</td>
</tr>
<tr>
  <th>zipcode_98177</th>    <td> 1.859e+05</td> <td> 7048.310</td> <td>   26.374</td> <td> 0.000</td> <td> 1.72e+05</td> <td>    2e+05</td>
</tr>
<tr>
  <th>zipcode_98178</th>    <td>  4.12e+04</td> <td> 6272.872</td> <td>    6.569</td> <td> 0.000</td> <td> 2.89e+04</td> <td> 5.35e+04</td>
</tr>
<tr>
  <th>zipcode_98188</th>    <td> 2.428e+04</td> <td> 7679.720</td> <td>    3.161</td> <td> 0.002</td> <td> 9223.028</td> <td> 3.93e+04</td>
</tr>
<tr>
  <th>zipcode_98198</th>    <td> 3.161e+04</td> <td> 6226.177</td> <td>    5.077</td> <td> 0.000</td> <td> 1.94e+04</td> <td> 4.38e+04</td>
</tr>
<tr>
  <th>zipcode_98199</th>    <td> 2.673e+05</td> <td> 7404.979</td> <td>   36.100</td> <td> 0.000</td> <td> 2.53e+05</td> <td> 2.82e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>561.223</td> <th>  Durbin-Watson:     </th> <td>   2.024</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1486.527</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.243</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.659</td>  <th>  Cond. No.          </th> <td>    100.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# Results sorted by coefficients descending 

results1_as_html = model1.summary().tables[1].as_html()
results1 = pd.read_html(results1_as_html, header=0, index_col=0)[0]
results1.sort_values('coef', ascending=False)#.set_option('display.max_rows', None)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coef</th>
      <th>std err</th>
      <th>t</th>
      <th>P&gt;|t|</th>
      <th>[0.025</th>
      <th>0.975]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>zipcode_98004</th>
      <td>371400.0000</td>
      <td>16300.000</td>
      <td>22.743</td>
      <td>0.000</td>
      <td>339000.000</td>
      <td>403000.000</td>
    </tr>
    <tr>
      <th>zipcode_98040</th>
      <td>326000.0000</td>
      <td>15900.000</td>
      <td>20.537</td>
      <td>0.000</td>
      <td>295000.000</td>
      <td>357000.000</td>
    </tr>
    <tr>
      <th>zipcode_98109</th>
      <td>311900.0000</td>
      <td>12900.000</td>
      <td>24.230</td>
      <td>0.000</td>
      <td>287000.000</td>
      <td>337000.000</td>
    </tr>
    <tr>
      <th>zipcode_98105</th>
      <td>305100.0000</td>
      <td>8436.754</td>
      <td>36.162</td>
      <td>0.000</td>
      <td>289000.000</td>
      <td>322000.000</td>
    </tr>
    <tr>
      <th>zipcode_98112</th>
      <td>303300.0000</td>
      <td>9963.820</td>
      <td>30.437</td>
      <td>0.000</td>
      <td>284000.000</td>
      <td>323000.000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>basement_present</th>
      <td>-5546.0095</td>
      <td>721.806</td>
      <td>-7.684</td>
      <td>0.000</td>
      <td>-6960.868</td>
      <td>-4131.151</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>-7602.7410</td>
      <td>836.628</td>
      <td>-9.087</td>
      <td>0.000</td>
      <td>-9242.669</td>
      <td>-5962.812</td>
    </tr>
    <tr>
      <th>zipcode_98032</th>
      <td>-12230.0000</td>
      <td>8048.185</td>
      <td>-1.519</td>
      <td>0.129</td>
      <td>-28000.000</td>
      <td>3547.731</td>
    </tr>
    <tr>
      <th>zipcode_98002</th>
      <td>-12900.0000</td>
      <td>6672.443</td>
      <td>-1.933</td>
      <td>0.053</td>
      <td>-26000.000</td>
      <td>178.832</td>
    </tr>
    <tr>
      <th>zipcode_98023</th>
      <td>-15750.0000</td>
      <td>5246.049</td>
      <td>-3.002</td>
      <td>0.003</td>
      <td>-26000.000</td>
      <td>-5466.508</td>
    </tr>
  </tbody>
</table>
<p>78 rows × 6 columns</p>
</div>



### Check Linear Model Assumptions

**1. Linearity** 

**2. Residual Normality** <br>
sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True) <br>
Omnibus Value

**3. Homoskedasticity** <br>
Durbin-Watson: range of 1.5 to 2.5 is relatively normal

**4. Multicollinearity** <br>
VIF (variance_inflation_factor())

**Also check p-value** <br>
A p-value less than 0.05 (typically ≤ 0.05) is statistically significant. It indicates strong evidence against the null hypothesis, as there is less than a 5% probability the null is correct (and the results are random).

**Check for overfitting** <br>
- Mean Absolute Error (MAE)
- Mean Sqaured Error (MSE)
- Root Mean Sqaured Error (RMSE)


```python
# Check linearity and residual normality 

sm.graphics.qqplot(model1.resid, dist=stats.norm, line='45', fit=True);
```


    
![png](output_134_0.png)
    



```python
# Check for Multicollinearity

def create_vif_dct(dataframe, const_col_name='const'):
    
    if const_col_name not in dataframe.columns:
        dataframe = sm.add_constant(dataframe)
        
    # Dummy-checking
    df = dataframe.select_dtypes('number')
    if df.shape != dataframe.shape:
        warnings.warn('\n\nThere are non-numerical columns trying to be passed!\nThese have automatically been removed.\n')
    if df.isna().sum().any():
        raise ValueError('There may not be any missing values in the dataframe!')
        
    # Creating VIF Dictionary
    vif_dct = {}

    # Loop through each row and set the variable name to the VIF
    for i in range(len(df.columns)):
        vif = variance_inflation_factor(df.values, i)
        v = df.columns[i]
        vif_dct[v] = vif

    return vif_dct
```


```python
# Check for multicolinearity - anything over 10 is not good! 

create_vif_dct(X_train_scaled)
```




    {'const': 48.804888593432594,
     'bedrooms': 1.7525266115446538,
     'bathrooms': 2.520977861021298,
     'sqft_living': 3.0332438247636637,
     'floors': 2.0885480888237606,
     'waterfront': 1.0774400957148569,
     'condition': 1.1822282938705688,
     'grade': 1.996598740009202,
     'renovated': 1.0229074802205151,
     'basement_present': 1.554607741508546,
     'zipcode_98002': 1.572374400097395,
     'zipcode_98003': 1.7683916273984623,
     'zipcode_98004': 1.065100539678895,
     'zipcode_98005': 1.1552525010993069,
     'zipcode_98006': 1.5495760787475503,
     'zipcode_98007': 1.2999857058283855,
     'zipcode_98008': 1.6562704322361503,
     'zipcode_98010': 1.2621828886369568,
     'zipcode_98011': 1.4914193011733052,
     'zipcode_98014': 1.264640922782956,
     'zipcode_98019': 1.5651862005750914,
     'zipcode_98022': 1.6764572920324579,
     'zipcode_98023': 2.401885345658481,
     'zipcode_98024': 1.1811733768623345,
     'zipcode_98027': 1.8251705776763347,
     'zipcode_98028': 1.7782165922916775,
     'zipcode_98029': 1.688382860504492,
     'zipcode_98030': 1.695277852470107,
     'zipcode_98031': 1.8239180673146302,
     'zipcode_98032': 1.3345289185972118,
     'zipcode_98033': 1.6002524120784773,
     'zipcode_98034': 2.336472752494877,
     'zipcode_98038': 2.529762947645856,
     'zipcode_98040': 1.0695100160422801,
     'zipcode_98042': 2.48919498398572,
     'zipcode_98045': 1.5684405390947247,
     'zipcode_98052': 1.9670133379258323,
     'zipcode_98053': 1.6383968948327174,
     'zipcode_98055': 1.7506671291350397,
     'zipcode_98056': 2.08370547501707,
     'zipcode_98058': 2.230646819307026,
     'zipcode_98059': 2.1268217530668387,
     'zipcode_98065': 1.772519797918147,
     'zipcode_98070': 1.3587003781601552,
     'zipcode_98072': 1.610588276152504,
     'zipcode_98074': 1.7180666400873077,
     'zipcode_98075': 1.316109939412652,
     'zipcode_98077': 1.2926215328810982,
     'zipcode_98092': 1.9374125918788754,
     'zipcode_98102': 1.1593679933535124,
     'zipcode_98103': 2.3265534775693357,
     'zipcode_98105': 1.3084770106186392,
     'zipcode_98106': 1.9318565444070566,
     'zipcode_98107': 1.7119511051672336,
     'zipcode_98108': 1.5428738521577468,
     'zipcode_98109': 1.1155508114641512,
     'zipcode_98112': 1.2110043603791862,
     'zipcode_98115': 2.164337059909012,
     'zipcode_98116': 1.6821988102483125,
     'zipcode_98117': 2.280513896798148,
     'zipcode_98118': 2.1946814064237246,
     'zipcode_98119': 1.2370712597262954,
     'zipcode_98122': 1.5701214411554572,
     'zipcode_98125': 2.0238874056833405,
     'zipcode_98126': 1.9755145425871357,
     'zipcode_98133': 2.4169932736595734,
     'zipcode_98136': 1.6342507660544252,
     'zipcode_98144': 1.7882327681733308,
     'zipcode_98146': 1.713299220605527,
     'zipcode_98148': 1.1564530569895037,
     'zipcode_98155': 2.228779395730326,
     'zipcode_98166': 1.651663330462584,
     'zipcode_98168': 1.6590811305558477,
     'zipcode_98177': 1.487356216897902,
     'zipcode_98178': 1.7056064002303757,
     'zipcode_98188': 1.3748946030663856,
     'zipcode_98198': 1.7179126771024968,
     'zipcode_98199': 1.4534675049413095}




```python
# Predictions on the training set

y_train_pred = model1.predict(X_train_scaled) 
y_test_pred = model1.predict(X_test_scaled)
```


```python
# Regression metrics for Train data
# Mean Absolute Error (MAE)
# Mean Sqaured Error (MSE)
# Root Mean Sqaured Error (RMSE)
# Is this only done once? 

train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

print('Train MAE:', train_mae)
print('Train MSE:', train_mse)
print('Train RMSE:', train_rmse)
```

    Train MAE: 47495.546251964
    Train MSE: 3974372692.5231338
    Train RMSE: 63042.62599640924



```python
# Regression metrics for Test data

test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print('Test MAE:', test_mae)
print('Test MSE:', test_mse)
print('Test RMSE:', test_rmse)
```

    Test MAE: 48683.51726441247
    Test MSE: 4208824782.898014
    Test RMSE: 64875.45593595481


## Model 2: All Features Excluding Zipcode


```python
# Model without zipcodes 

[c for c in X_train_scaled.columns if not c.startswith('zipcode')]
```




    ['const',
     'bedrooms',
     'bathrooms',
     'sqft_living',
     'floors',
     'waterfront',
     'condition',
     'grade',
     'renovated',
     'basement_present']




```python
model2 = sm.OLS(y_train, X_train_scaled[[c for c in X_train_scaled.columns if not c.startswith('zipcode')]]).fit()
model2.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.274</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>310058.2259</td>
</tr>
<tr>
         <td>Date:</td>        <td>2021-11-07 12:33</td>        <td>BIC:</td>         <td>310132.0999</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>11937</td>        <td>Log-Likelihood:</td>   <td>-1.5502e+05</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>9</td>           <td>F-statistic:</td>        <td>502.4</td>   
</tr>
<tr>
     <td>Df Residuals:</td>          <td>11927</td>      <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.275</td>            <td>Scale:</td>        <td>1.1163e+10</td> 
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>Coef.</th>    <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>   <th>[0.025</th>      <th>0.975]</th>   
</tr>
<tr>
  <th>const</th>            <td>399995.9067</td> <td>967.0205</td>  <td>413.6375</td> <td>0.0000</td> <td>398100.3890</td> <td>401891.4243</td>
</tr>
<tr>
  <th>bedrooms</th>         <td>-9259.6531</td>  <td>1252.8124</td>  <td>-7.3911</td> <td>0.0000</td> <td>-11715.3696</td> <td>-6803.9366</td> 
</tr>
<tr>
  <th>bathrooms</th>        <td>-7685.7973</td>  <td>1492.9577</td>  <td>-5.1480</td> <td>0.0000</td> <td>-10612.2376</td> <td>-4759.3570</td> 
</tr>
<tr>
  <th>sqft_living</th>      <td>32467.9265</td>  <td>1589.2035</td>  <td>20.4303</td> <td>0.0000</td> <td>29352.8288</td>  <td>35583.0242</td> 
</tr>
<tr>
  <th>floors</th>           <td>13156.1308</td>  <td>1279.6647</td>  <td>10.2809</td> <td>0.0000</td> <td>10647.7796</td>  <td>15664.4820</td> 
</tr>
<tr>
  <th>waterfront</th>        <td>3319.7723</td>  <td>968.5369</td>   <td>3.4276</td>  <td>0.0006</td>  <td>1421.2822</td>   <td>5218.2624</td> 
</tr>
<tr>
  <th>condition</th>        <td>10388.1898</td>  <td>1020.9067</td>  <td>10.1755</td> <td>0.0000</td>  <td>8387.0464</td>  <td>12389.3333</td> 
</tr>
<tr>
  <th>grade</th>            <td>40496.1086</td>  <td>1306.6716</td>  <td>30.9918</td> <td>0.0000</td> <td>37934.8194</td>  <td>43057.3978</td> 
</tr>
<tr>
  <th>renovated</th>         <td>4306.2156</td>  <td>970.8841</td>   <td>4.4354</td>  <td>0.0000</td>  <td>2403.1245</td>   <td>6209.3066</td> 
</tr>
<tr>
  <th>basement_present</th> <td>20118.1663</td>  <td>1080.4635</td>  <td>18.6199</td> <td>0.0000</td> <td>18000.2819</td>  <td>22236.0508</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>560.278</td>  <td>Durbin-Watson:</td>    <td>2.020</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>331.293</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.266</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>2.381</td>   <td>Condition No.:</td>      <td>3</td>   
</tr>
</table>




```python
# Check linearity and residual normality 

sm.graphics.qqplot(model2.resid, dist=stats.norm, line='45', fit=True);
```


    
![png](output_143_0.png)
    



```python
create_vif_dct(X_train_scaled[[c for c in X_train_scaled.columns if not c.startswith('zipcode')]])
```




    {'const': 1.0,
     'bedrooms': 1.6784204890659655,
     'bathrooms': 2.3835466342659077,
     'sqft_living': 2.7007704622217075,
     'floors': 1.7511405980093482,
     'waterfront': 1.003138717164317,
     'condition': 1.114553090585022,
     'grade': 1.8258350881548409,
     'renovated': 1.0080067669410961,
     'basement_present': 1.2483858731730055}



## MODEL VI - Combining Features Into One Model


```python
model_vi = sm.OLS(y_train, X_train_scaled[['const', 'sqft_living','floors', 'waterfront', 'condition','grade', 
         'zipcode_98023', 'zipcode_98034','zipcode_98133','zipcode_98042', 'zipcode_98038']]).fit()
model_vi.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.329</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.329</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   585.8</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 07 Nov 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:33:02</td>     <th>  Log-Likelihood:    </th> <td>-1.5455e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 11937</td>      <th>  AIC:               </th>  <td>3.091e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 11926</td>      <th>  BIC:               </th>  <td>3.092e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>         <td> 4.107e+05</td> <td> 1011.622</td> <td>  405.947</td> <td> 0.000</td> <td> 4.09e+05</td> <td> 4.13e+05</td>
</tr>
<tr>
  <th>sqft_living</th>   <td> 3.192e+04</td> <td> 1151.958</td> <td>   27.707</td> <td> 0.000</td> <td> 2.97e+04</td> <td> 3.42e+04</td>
</tr>
<tr>
  <th>floors</th>        <td> 3804.4366</td> <td> 1063.923</td> <td>    3.576</td> <td> 0.000</td> <td> 1718.975</td> <td> 5889.898</td>
</tr>
<tr>
  <th>waterfront</th>    <td> 4354.0289</td> <td>  930.851</td> <td>    4.677</td> <td> 0.000</td> <td> 2529.409</td> <td> 6178.649</td>
</tr>
<tr>
  <th>condition</th>     <td> 9896.3054</td> <td>  980.274</td> <td>   10.095</td> <td> 0.000</td> <td> 7974.809</td> <td> 1.18e+04</td>
</tr>
<tr>
  <th>grade</th>         <td> 4.113e+04</td> <td> 1228.119</td> <td>   33.488</td> <td> 0.000</td> <td> 3.87e+04</td> <td> 4.35e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th> <td>-1.487e+05</td> <td> 5477.516</td> <td>  -27.149</td> <td> 0.000</td> <td>-1.59e+05</td> <td>-1.38e+05</td>
</tr>
<tr>
  <th>zipcode_98034</th> <td> 3.045e+04</td> <td> 5589.444</td> <td>    5.447</td> <td> 0.000</td> <td> 1.95e+04</td> <td> 4.14e+04</td>
</tr>
<tr>
  <th>zipcode_98133</th> <td>  -1.6e+04</td> <td> 5469.389</td> <td>   -2.925</td> <td> 0.003</td> <td>-2.67e+04</td> <td>-5277.816</td>
</tr>
<tr>
  <th>zipcode_98042</th> <td>-1.179e+05</td> <td> 5291.954</td> <td>  -22.277</td> <td> 0.000</td> <td>-1.28e+05</td> <td>-1.08e+05</td>
</tr>
<tr>
  <th>zipcode_98038</th> <td> -8.43e+04</td> <td> 5273.910</td> <td>  -15.985</td> <td> 0.000</td> <td>-9.46e+04</td> <td> -7.4e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>253.525</td> <th>  Durbin-Watson:     </th> <td>   2.020</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 194.004</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.225</td>  <th>  Prob(JB):          </th> <td>7.46e-43</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.567</td>  <th>  Cond. No.          </th> <td>    8.78</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
model_vii = sm.OLS(y_train, X_train_scaled[
                                            ['const', 'bedrooms', 'bathrooms', 'sqft_living','floors', 'waterfront', 
                                             'condition','grade','renovated', 'basement_present', 
                                             'zipcode_98038', 'zipcode_98042','zipcode_98133','zipcode_98034', 
                                             'zipcode_98023', 'zipcode_98118', 'zipcode_98058','zipcode_98103',
                                             'zipcode_98155', 'zipcode_98117', 'zipcode_98115', 'zipcode_98056', 
                                             'zipcode_98125', 'zipcode_98092', 'zipcode_98126','zipcode_98052', 
                                             'zipcode_98059', 'zipcode_98106', 'zipcode_98027','zipcode_98028',
                                             'zipcode_98031', 'zipcode_98003', 'zipcode_98055','zipcode_98144', 
                                             'zipcode_98065', 'zipcode_98030', 'zipcode_98198','zipcode_98146',
                                             'zipcode_98074', 'zipcode_98178', 'zipcode_98116','zipcode_98168',
                                             'zipcode_98022', 'zipcode_98029', 'zipcode_98107','zipcode_98166',
                                             'zipcode_98053', 'zipcode_98008', 'zipcode_98072'
                                             ]]).fit()
model_vii.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.596</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.594</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   364.7</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 07 Nov 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:33:02</td>     <th>  Log-Likelihood:    </th> <td>-1.5153e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 11937</td>      <th>  AIC:               </th>  <td>3.032e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 11888</td>      <th>  BIC:               </th>  <td>3.035e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    48</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>            <td> 4.241e+05</td> <td> 1512.474</td> <td>  280.394</td> <td> 0.000</td> <td> 4.21e+05</td> <td> 4.27e+05</td>
</tr>
<tr>
  <th>bedrooms</th>         <td>-5652.4156</td> <td>  949.744</td> <td>   -5.952</td> <td> 0.000</td> <td>-7514.070</td> <td>-3790.761</td>
</tr>
<tr>
  <th>bathrooms</th>        <td>  673.0833</td> <td> 1143.095</td> <td>    0.589</td> <td> 0.556</td> <td>-1567.569</td> <td> 2913.736</td>
</tr>
<tr>
  <th>sqft_living</th>      <td> 4.784e+04</td> <td> 1232.513</td> <td>   38.813</td> <td> 0.000</td> <td> 4.54e+04</td> <td> 5.03e+04</td>
</tr>
<tr>
  <th>floors</th>           <td>-2145.8828</td> <td> 1023.023</td> <td>   -2.098</td> <td> 0.036</td> <td>-4151.175</td> <td> -140.591</td>
</tr>
<tr>
  <th>waterfront</th>       <td> 4766.5831</td> <td>  727.412</td> <td>    6.553</td> <td> 0.000</td> <td> 3340.736</td> <td> 6192.430</td>
</tr>
<tr>
  <th>condition</th>        <td> 9873.3829</td> <td>  778.366</td> <td>   12.685</td> <td> 0.000</td> <td> 8347.658</td> <td> 1.14e+04</td>
</tr>
<tr>
  <th>grade</th>            <td>   3.4e+04</td> <td> 1007.305</td> <td>   33.757</td> <td> 0.000</td> <td>  3.2e+04</td> <td>  3.6e+04</td>
</tr>
<tr>
  <th>renovated</th>        <td> 2936.9211</td> <td>  729.642</td> <td>    4.025</td> <td> 0.000</td> <td> 1506.704</td> <td> 4367.138</td>
</tr>
<tr>
  <th>basement_present</th> <td> 4027.3021</td> <td>  869.001</td> <td>    4.634</td> <td> 0.000</td> <td> 2323.919</td> <td> 5730.685</td>
</tr>
<tr>
  <th>zipcode_98038</th>    <td>-9.724e+04</td> <td> 4338.657</td> <td>  -22.411</td> <td> 0.000</td> <td>-1.06e+05</td> <td>-8.87e+04</td>
</tr>
<tr>
  <th>zipcode_98042</th>    <td> -1.32e+05</td> <td> 4338.533</td> <td>  -30.418</td> <td> 0.000</td> <td> -1.4e+05</td> <td>-1.23e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th>    <td>-2.739e+04</td> <td> 4455.160</td> <td>   -6.148</td> <td> 0.000</td> <td>-3.61e+04</td> <td>-1.87e+04</td>
</tr>
<tr>
  <th>zipcode_98034</th>    <td> 1.609e+04</td> <td> 4546.334</td> <td>    3.538</td> <td> 0.000</td> <td> 7173.768</td> <td>  2.5e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th>    <td>-1.649e+05</td> <td> 4453.855</td> <td>  -37.019</td> <td> 0.000</td> <td>-1.74e+05</td> <td>-1.56e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th>    <td>-1.931e+04</td> <td> 4858.360</td> <td>   -3.975</td> <td> 0.000</td> <td>-2.88e+04</td> <td>-9788.531</td>
</tr>
<tr>
  <th>zipcode_98058</th>    <td>-1.062e+05</td> <td> 4717.466</td> <td>  -22.505</td> <td> 0.000</td> <td>-1.15e+05</td> <td>-9.69e+04</td>
</tr>
<tr>
  <th>zipcode_98103</th>    <td> 8.163e+04</td> <td> 4841.251</td> <td>   16.862</td> <td> 0.000</td> <td> 7.21e+04</td> <td> 9.11e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th>    <td>-2.489e+04</td> <td> 4733.689</td> <td>   -5.259</td> <td> 0.000</td> <td>-3.42e+04</td> <td>-1.56e+04</td>
</tr>
<tr>
  <th>zipcode_98117</th>    <td> 9.952e+04</td> <td> 4753.278</td> <td>   20.937</td> <td> 0.000</td> <td> 9.02e+04</td> <td> 1.09e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th>    <td> 9.841e+04</td> <td> 4945.940</td> <td>   19.898</td> <td> 0.000</td> <td> 8.87e+04</td> <td> 1.08e+05</td>
</tr>
<tr>
  <th>zipcode_98056</th>    <td>-5.582e+04</td> <td> 5024.579</td> <td>  -11.109</td> <td> 0.000</td> <td>-6.57e+04</td> <td> -4.6e+04</td>
</tr>
<tr>
  <th>zipcode_98125</th>    <td> 1.505e+04</td> <td> 5160.293</td> <td>    2.917</td> <td> 0.004</td> <td> 4938.072</td> <td> 2.52e+04</td>
</tr>
<tr>
  <th>zipcode_98092</th>    <td>-1.441e+05</td> <td> 5387.935</td> <td>  -26.737</td> <td> 0.000</td> <td>-1.55e+05</td> <td>-1.33e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th>    <td> 4884.4199</td> <td> 5362.752</td> <td>    0.911</td> <td> 0.362</td> <td>-5627.452</td> <td> 1.54e+04</td>
</tr>
<tr>
  <th>zipcode_98052</th>    <td> 6.416e+04</td> <td> 5288.481</td> <td>   12.133</td> <td> 0.000</td> <td> 5.38e+04</td> <td> 7.45e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th>    <td>-4.759e+04</td> <td> 4948.292</td> <td>   -9.617</td> <td> 0.000</td> <td>-5.73e+04</td> <td>-3.79e+04</td>
</tr>
<tr>
  <th>zipcode_98106</th>    <td>-6.254e+04</td> <td> 5466.575</td> <td>  -11.440</td> <td> 0.000</td> <td>-7.33e+04</td> <td>-5.18e+04</td>
</tr>
<tr>
  <th>zipcode_98027</th>    <td> 2.428e+04</td> <td> 5701.700</td> <td>    4.258</td> <td> 0.000</td> <td> 1.31e+04</td> <td> 3.55e+04</td>
</tr>
<tr>
  <th>zipcode_98028</th>    <td>-1.144e+04</td> <td> 5831.027</td> <td>   -1.962</td> <td> 0.050</td> <td>-2.29e+04</td> <td>  -12.483</td>
</tr>
<tr>
  <th>zipcode_98031</th>    <td>-1.381e+05</td> <td> 5675.108</td> <td>  -24.343</td> <td> 0.000</td> <td>-1.49e+05</td> <td>-1.27e+05</td>
</tr>
<tr>
  <th>zipcode_98003</th>    <td>-1.477e+05</td> <td> 5868.998</td> <td>  -25.158</td> <td> 0.000</td> <td>-1.59e+05</td> <td>-1.36e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th>    <td>-1.076e+05</td> <td> 5924.361</td> <td>  -18.156</td> <td> 0.000</td> <td>-1.19e+05</td> <td> -9.6e+04</td>
</tr>
<tr>
  <th>zipcode_98144</th>    <td> 2.038e+04</td> <td> 6010.006</td> <td>    3.391</td> <td> 0.001</td> <td> 8598.944</td> <td> 3.22e+04</td>
</tr>
<tr>
  <th>zipcode_98065</th>    <td> 7408.9745</td> <td> 6024.633</td> <td>    1.230</td> <td> 0.219</td> <td>-4400.292</td> <td> 1.92e+04</td>
</tr>
<tr>
  <th>zipcode_98030</th>    <td>-1.382e+05</td> <td> 6153.042</td> <td>  -22.459</td> <td> 0.000</td> <td> -1.5e+05</td> <td>-1.26e+05</td>
</tr>
<tr>
  <th>zipcode_98198</th>    <td>-1.162e+05</td> <td> 6105.088</td> <td>  -19.035</td> <td> 0.000</td> <td>-1.28e+05</td> <td>-1.04e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th>    <td>-6.421e+04</td> <td> 6142.551</td> <td>  -10.453</td> <td> 0.000</td> <td>-7.62e+04</td> <td>-5.22e+04</td>
</tr>
<tr>
  <th>zipcode_98074</th>    <td> 4.896e+04</td> <td> 6139.049</td> <td>    7.975</td> <td> 0.000</td> <td> 3.69e+04</td> <td>  6.1e+04</td>
</tr>
<tr>
  <th>zipcode_98178</th>    <td>-1.067e+05</td> <td> 6180.882</td> <td>  -17.258</td> <td> 0.000</td> <td>-1.19e+05</td> <td>-9.46e+04</td>
</tr>
<tr>
  <th>zipcode_98116</th>    <td> 9.309e+04</td> <td> 6374.520</td> <td>   14.604</td> <td> 0.000</td> <td> 8.06e+04</td> <td> 1.06e+05</td>
</tr>
<tr>
  <th>zipcode_98168</th>    <td> -1.21e+05</td> <td> 6444.202</td> <td>  -18.776</td> <td> 0.000</td> <td>-1.34e+05</td> <td>-1.08e+05</td>
</tr>
<tr>
  <th>zipcode_98022</th>    <td> -1.17e+05</td> <td> 6283.692</td> <td>  -18.626</td> <td> 0.000</td> <td>-1.29e+05</td> <td>-1.05e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th>    <td> 5.868e+04</td> <td> 6335.627</td> <td>    9.262</td> <td> 0.000</td> <td> 4.63e+04</td> <td> 7.11e+04</td>
</tr>
<tr>
  <th>zipcode_98107</th>    <td> 1.172e+05</td> <td> 6450.715</td> <td>   18.164</td> <td> 0.000</td> <td> 1.05e+05</td> <td>  1.3e+05</td>
</tr>
<tr>
  <th>zipcode_98166</th>    <td>-4.721e+04</td> <td> 6394.965</td> <td>   -7.382</td> <td> 0.000</td> <td>-5.97e+04</td> <td>-3.47e+04</td>
</tr>
<tr>
  <th>zipcode_98053</th>    <td> 6.066e+04</td> <td> 6506.307</td> <td>    9.324</td> <td> 0.000</td> <td> 4.79e+04</td> <td> 7.34e+04</td>
</tr>
<tr>
  <th>zipcode_98008</th>    <td> 7.098e+04</td> <td> 6420.251</td> <td>   11.056</td> <td> 0.000</td> <td> 5.84e+04</td> <td> 8.36e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th>    <td>  1.54e+04</td> <td> 6564.236</td> <td>    2.345</td> <td> 0.019</td> <td> 2528.185</td> <td> 2.83e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>270.308</td> <th>  Durbin-Watson:     </th> <td>   2.016</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 488.488</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.178</td>  <th>  Prob(JB):          </th> <td>8.44e-107</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.925</td>  <th>  Cond. No.          </th> <td>    25.7</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
model_viii = sm.OLS(y_train, X_train_scaled[
                                            ['const', 'bedrooms', 'bathrooms', 'sqft_living','floors', 'waterfront', 
                                             'condition','grade',
                                             'zipcode_98038', 'zipcode_98042','zipcode_98133','zipcode_98034', 
                                             'zipcode_98023', 'zipcode_98118', 'zipcode_98058','zipcode_98103',
                                             'zipcode_98155', 'zipcode_98117', 'zipcode_98115', 'zipcode_98056', 
                                             'zipcode_98125', 'zipcode_98092', 'zipcode_98126','zipcode_98052', 
                                             'zipcode_98059', 'zipcode_98106', 'zipcode_98027','zipcode_98028',
                                             'zipcode_98031', 'zipcode_98003', 'zipcode_98055','zipcode_98144', 
                                             'zipcode_98065', 'zipcode_98030', 'zipcode_98198','zipcode_98146',
                                             'zipcode_98074', 'zipcode_98178', 'zipcode_98116','zipcode_98168',
                                             'zipcode_98022', 'zipcode_98029', 'zipcode_98107','zipcode_98166',
                                             'zipcode_98053', 'zipcode_98008', 'zipcode_98072'
                                             ]]).fit()
model_viii.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.594</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.593</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   378.6</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 07 Nov 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:33:02</td>     <th>  Log-Likelihood:    </th> <td>-1.5155e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 11937</td>      <th>  AIC:               </th>  <td>3.032e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 11890</td>      <th>  BIC:               </th>  <td>3.035e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    46</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>         <td> 4.243e+05</td> <td> 1514.371</td> <td>  280.169</td> <td> 0.000</td> <td> 4.21e+05</td> <td> 4.27e+05</td>
</tr>
<tr>
  <th>bedrooms</th>      <td>-5846.3693</td> <td>  950.573</td> <td>   -6.150</td> <td> 0.000</td> <td>-7709.647</td> <td>-3983.092</td>
</tr>
<tr>
  <th>bathrooms</th>     <td> 1885.2190</td> <td> 1116.215</td> <td>    1.689</td> <td> 0.091</td> <td> -302.745</td> <td> 4073.183</td>
</tr>
<tr>
  <th>sqft_living</th>   <td> 4.915e+04</td> <td> 1208.369</td> <td>   40.673</td> <td> 0.000</td> <td> 4.68e+04</td> <td> 5.15e+04</td>
</tr>
<tr>
  <th>floors</th>        <td>-4210.8972</td> <td>  931.176</td> <td>   -4.522</td> <td> 0.000</td> <td>-6036.154</td> <td>-2385.640</td>
</tr>
<tr>
  <th>waterfront</th>    <td> 4848.5357</td> <td>  728.354</td> <td>    6.657</td> <td> 0.000</td> <td> 3420.842</td> <td> 6276.229</td>
</tr>
<tr>
  <th>condition</th>     <td> 9661.6760</td> <td>  777.909</td> <td>   12.420</td> <td> 0.000</td> <td> 8136.847</td> <td> 1.12e+04</td>
</tr>
<tr>
  <th>grade</th>         <td> 3.363e+04</td> <td> 1006.737</td> <td>   33.409</td> <td> 0.000</td> <td> 3.17e+04</td> <td> 3.56e+04</td>
</tr>
<tr>
  <th>zipcode_98038</th> <td>-9.973e+04</td> <td> 4317.858</td> <td>  -23.098</td> <td> 0.000</td> <td>-1.08e+05</td> <td>-9.13e+04</td>
</tr>
<tr>
  <th>zipcode_98042</th> <td> -1.34e+05</td> <td> 4323.998</td> <td>  -30.987</td> <td> 0.000</td> <td>-1.42e+05</td> <td>-1.26e+05</td>
</tr>
<tr>
  <th>zipcode_98133</th> <td>-2.647e+04</td> <td> 4458.504</td> <td>   -5.936</td> <td> 0.000</td> <td>-3.52e+04</td> <td>-1.77e+04</td>
</tr>
<tr>
  <th>zipcode_98034</th> <td> 1.513e+04</td> <td> 4550.279</td> <td>    3.325</td> <td> 0.001</td> <td> 6211.093</td> <td>  2.4e+04</td>
</tr>
<tr>
  <th>zipcode_98023</th> <td>-1.658e+05</td> <td> 4457.706</td> <td>  -37.198</td> <td> 0.000</td> <td>-1.75e+05</td> <td>-1.57e+05</td>
</tr>
<tr>
  <th>zipcode_98118</th> <td>-1.777e+04</td> <td> 4852.669</td> <td>   -3.662</td> <td> 0.000</td> <td>-2.73e+04</td> <td>-8256.803</td>
</tr>
<tr>
  <th>zipcode_98058</th> <td>-1.073e+05</td> <td> 4717.716</td> <td>  -22.753</td> <td> 0.000</td> <td>-1.17e+05</td> <td>-9.81e+04</td>
</tr>
<tr>
  <th>zipcode_98103</th> <td> 8.382e+04</td> <td> 4826.380</td> <td>   17.366</td> <td> 0.000</td> <td> 7.44e+04</td> <td> 9.33e+04</td>
</tr>
<tr>
  <th>zipcode_98155</th> <td>-2.506e+04</td> <td> 4740.521</td> <td>   -5.286</td> <td> 0.000</td> <td>-3.44e+04</td> <td>-1.58e+04</td>
</tr>
<tr>
  <th>zipcode_98117</th> <td> 1.018e+05</td> <td> 4736.929</td> <td>   21.492</td> <td> 0.000</td> <td> 9.25e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>zipcode_98115</th> <td> 1.007e+05</td> <td> 4929.541</td> <td>   20.436</td> <td> 0.000</td> <td> 9.11e+04</td> <td>  1.1e+05</td>
</tr>
<tr>
  <th>zipcode_98056</th> <td>-5.739e+04</td> <td> 5024.052</td> <td>  -11.424</td> <td> 0.000</td> <td>-6.72e+04</td> <td>-4.75e+04</td>
</tr>
<tr>
  <th>zipcode_98125</th> <td> 1.575e+04</td> <td> 5165.362</td> <td>    3.048</td> <td> 0.002</td> <td> 5621.331</td> <td> 2.59e+04</td>
</tr>
<tr>
  <th>zipcode_98092</th> <td>-1.464e+05</td> <td> 5375.938</td> <td>  -27.234</td> <td> 0.000</td> <td>-1.57e+05</td> <td>-1.36e+05</td>
</tr>
<tr>
  <th>zipcode_98126</th> <td> 6264.9652</td> <td> 5358.860</td> <td>    1.169</td> <td> 0.242</td> <td>-4239.276</td> <td> 1.68e+04</td>
</tr>
<tr>
  <th>zipcode_98052</th> <td> 6.385e+04</td> <td> 5295.982</td> <td>   12.056</td> <td> 0.000</td> <td> 5.35e+04</td> <td> 7.42e+04</td>
</tr>
<tr>
  <th>zipcode_98059</th> <td>-4.989e+04</td> <td> 4933.774</td> <td>  -10.112</td> <td> 0.000</td> <td>-5.96e+04</td> <td>-4.02e+04</td>
</tr>
<tr>
  <th>zipcode_98106</th> <td>-6.055e+04</td> <td> 5457.551</td> <td>  -11.095</td> <td> 0.000</td> <td>-7.13e+04</td> <td>-4.99e+04</td>
</tr>
<tr>
  <th>zipcode_98027</th> <td> 2.441e+04</td> <td> 5708.051</td> <td>    4.276</td> <td> 0.000</td> <td> 1.32e+04</td> <td> 3.56e+04</td>
</tr>
<tr>
  <th>zipcode_98028</th> <td>-1.237e+04</td> <td> 5837.607</td> <td>   -2.119</td> <td> 0.034</td> <td>-2.38e+04</td> <td> -926.365</td>
</tr>
<tr>
  <th>zipcode_98031</th> <td>-1.398e+05</td> <td> 5676.382</td> <td>  -24.631</td> <td> 0.000</td> <td>-1.51e+05</td> <td>-1.29e+05</td>
</tr>
<tr>
  <th>zipcode_98003</th> <td>-1.488e+05</td> <td> 5874.459</td> <td>  -25.332</td> <td> 0.000</td> <td> -1.6e+05</td> <td>-1.37e+05</td>
</tr>
<tr>
  <th>zipcode_98055</th> <td>-1.087e+05</td> <td> 5930.030</td> <td>  -18.325</td> <td> 0.000</td> <td> -1.2e+05</td> <td> -9.7e+04</td>
</tr>
<tr>
  <th>zipcode_98144</th> <td> 2.343e+04</td> <td> 5980.197</td> <td>    3.918</td> <td> 0.000</td> <td> 1.17e+04</td> <td> 3.52e+04</td>
</tr>
<tr>
  <th>zipcode_98065</th> <td> 4519.2460</td> <td> 6009.425</td> <td>    0.752</td> <td> 0.452</td> <td>-7260.210</td> <td> 1.63e+04</td>
</tr>
<tr>
  <th>zipcode_98030</th> <td>-1.404e+05</td> <td> 6149.610</td> <td>  -22.837</td> <td> 0.000</td> <td>-1.52e+05</td> <td>-1.28e+05</td>
</tr>
<tr>
  <th>zipcode_98198</th> <td>-1.168e+05</td> <td> 6112.938</td> <td>  -19.110</td> <td> 0.000</td> <td>-1.29e+05</td> <td>-1.05e+05</td>
</tr>
<tr>
  <th>zipcode_98146</th> <td>-6.337e+04</td> <td> 6146.651</td> <td>  -10.309</td> <td> 0.000</td> <td>-7.54e+04</td> <td>-5.13e+04</td>
</tr>
<tr>
  <th>zipcode_98074</th> <td> 4.809e+04</td> <td> 6145.960</td> <td>    7.824</td> <td> 0.000</td> <td>  3.6e+04</td> <td> 6.01e+04</td>
</tr>
<tr>
  <th>zipcode_98178</th> <td> -1.06e+05</td> <td> 6188.362</td> <td>  -17.136</td> <td> 0.000</td> <td>-1.18e+05</td> <td>-9.39e+04</td>
</tr>
<tr>
  <th>zipcode_98116</th> <td> 9.647e+04</td> <td> 6356.201</td> <td>   15.177</td> <td> 0.000</td> <td>  8.4e+04</td> <td> 1.09e+05</td>
</tr>
<tr>
  <th>zipcode_98168</th> <td>-1.206e+05</td> <td> 6451.844</td> <td>  -18.687</td> <td> 0.000</td> <td>-1.33e+05</td> <td>-1.08e+05</td>
</tr>
<tr>
  <th>zipcode_98022</th> <td> -1.19e+05</td> <td> 6270.925</td> <td>  -18.980</td> <td> 0.000</td> <td>-1.31e+05</td> <td>-1.07e+05</td>
</tr>
<tr>
  <th>zipcode_98029</th> <td> 5.718e+04</td> <td> 6339.675</td> <td>    9.019</td> <td> 0.000</td> <td> 4.47e+04</td> <td> 6.96e+04</td>
</tr>
<tr>
  <th>zipcode_98107</th> <td> 1.202e+05</td> <td> 6428.753</td> <td>   18.698</td> <td> 0.000</td> <td> 1.08e+05</td> <td> 1.33e+05</td>
</tr>
<tr>
  <th>zipcode_98166</th> <td>-4.647e+04</td> <td> 6401.602</td> <td>   -7.260</td> <td> 0.000</td> <td> -5.9e+04</td> <td>-3.39e+04</td>
</tr>
<tr>
  <th>zipcode_98053</th> <td> 5.755e+04</td> <td> 6488.655</td> <td>    8.870</td> <td> 0.000</td> <td> 4.48e+04</td> <td> 7.03e+04</td>
</tr>
<tr>
  <th>zipcode_98008</th> <td> 6.966e+04</td> <td> 6425.601</td> <td>   10.841</td> <td> 0.000</td> <td> 5.71e+04</td> <td> 8.23e+04</td>
</tr>
<tr>
  <th>zipcode_98072</th> <td>  1.44e+04</td> <td> 6571.852</td> <td>    2.191</td> <td> 0.028</td> <td> 1519.519</td> <td> 2.73e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>274.230</td> <th>  Durbin-Watson:     </th> <td>   2.017</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 495.916</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.181</td>  <th>  Prob(JB):          </th> <td>2.06e-108</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.930</td>  <th>  Cond. No.          </th> <td>    25.7</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
create_vif_dct(X_train_scaled[['const', 'bedrooms', 'bathrooms', 'sqft_living','floors', 'waterfront', 
                                             'condition','grade', 
                                             'zipcode_98038', 'zipcode_98042','zipcode_98133','zipcode_98034', 
                                             'zipcode_98023', 'zipcode_98118', 'zipcode_98058','zipcode_98103',
                                             'zipcode_98155', 'zipcode_98117', 'zipcode_98115', 'zipcode_98056', 
                                             'zipcode_98125', 'zipcode_98092', 'zipcode_98126','zipcode_98052', 
                                             'zipcode_98059', 'zipcode_98106', 'zipcode_98027','zipcode_98028',
                                             'zipcode_98031', 'zipcode_98003', 'zipcode_98055','zipcode_98144', 
                                             'zipcode_98065', 'zipcode_98030', 'zipcode_98198','zipcode_98146',
                                             'zipcode_98074', 'zipcode_98178', 'zipcode_98116','zipcode_98168',
                                             'zipcode_98022', 'zipcode_98029', 'zipcode_98107','zipcode_98166',
                                             'zipcode_98053', 'zipcode_98008', 'zipcode_98072']])
```




    {'const': 4.369725983879623,
     'bedrooms': 1.7217107677678187,
     'bathrooms': 2.3740262731332087,
     'sqft_living': 2.7822034535701605,
     'floors': 1.6521639283198817,
     'waterfront': 1.0108229358922625,
     'condition': 1.1530477044209724,
     'grade': 1.9311768338639281,
     'zipcode_98038': 1.1310582307350479,
     'zipcode_98042': 1.1119573653335004,
     'zipcode_98133': 1.1048559490438092,
     'zipcode_98034': 1.1010391596644689,
     'zipcode_98023': 1.1074410168301358,
     'zipcode_98118': 1.0921759799392237,
     'zipcode_98058': 1.0929327058184868,
     'zipcode_98103': 1.1403385012782687,
     'zipcode_98155': 1.0933301221801928,
     'zipcode_98117': 1.0984603918015572,
     'zipcode_98115': 1.0864756529302824,
     'zipcode_98056': 1.0863067171350993,
     'zipcode_98125': 1.0750438394752115,
     'zipcode_98092': 1.076052729999994,
     'zipcode_98126': 1.0780274053189864,
     'zipcode_98052': 1.0743536168784191,
     'zipcode_98059': 1.0920411077296286,
     'zipcode_98106': 1.0724286270863268,
     'zipcode_98027': 1.062916758805177,
     'zipcode_98028': 1.059169145684753,
     'zipcode_98031': 1.0610813451510335,
     'zipcode_98003': 1.0566026776285689,
     'zipcode_98055': 1.0549609390062185,
     'zipcode_98144': 1.061832605422847,
     'zipcode_98065': 1.083399255936633,
     'zipcode_98030': 1.0526281865251654,
     'zipcode_98198': 1.0574706129067932,
     'zipcode_98146': 1.0574667844366896,
     'zipcode_98074': 1.0630781153416873,
     'zipcode_98178': 1.0600041905229836,
     'zipcode_98116': 1.049370290096881,
     'zipcode_98168': 1.0618004887674302,
     'zipcode_98022': 1.051906702404663,
     'zipcode_98029': 1.0688645161908064,
     'zipcode_98107': 1.0734626251617383,
     'zipcode_98166': 1.051691153998909,
     'zipcode_98053': 1.0543317642868377,
     'zipcode_98008': 1.059591481845864,
     'zipcode_98072': 1.047976247703205}


```python
# Check linearity and residual normality 

sm.graphics.qqplot(model_viii.resid, dist=stats.norm, line='45', fit=True);
```
    

# FINAL MODEL

** Final Model includes: 

1. Bedrooms 
2. Bathrooms 
3. Square Feet Living
4. Floors
5. Waterfront 
6. Condition
7. Grade
8. Renovation Status
9. Basement Present
10. All Zipcodes

Excluding zipcodes, **Square Feet Living, Grade, and Condition are the strongest determinants of price** 

<img src='images/Final Model.png'>

```python
model_final = sm.OLS(y_train, X_train_scaled).fit()
model_final.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.740</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>297876.8343</td>
</tr>
<tr>
         <td>Date:</td>        <td>2021-11-07 12:33</td>        <td>BIC:</td>         <td>298453.0513</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>11937</td>        <td>Log-Likelihood:</td>   <td>-1.4886e+05</td>
</tr>
<tr>
       <td>Df Model:</td>             <td>77</td>           <td>F-statistic:</td>        <td>442.0</td>   
</tr>
<tr>
     <td>Df Residuals:</td>          <td>11859</td>      <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.742</td>            <td>Scale:</td>        <td>4.0005e+09</td> 
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>Coef.</th>     <th>Std.Err.</th>     <th>t</th>     <th>P>|t|</th>   <th>[0.025</th>      <th>0.975]</th>   
</tr>
<tr>
  <th>const</th>            <td>273461.5916</td>  <td>4044.2866</td> <td>67.6168</td> <td>0.0000</td> <td>265534.1265</td> <td>281389.0568</td>
</tr>
<tr>
  <th>bedrooms</th>         <td>-3777.9235</td>   <td>766.3772</td>  <td>-4.9296</td> <td>0.0000</td> <td>-5280.1485</td>  <td>-2275.6984</td> 
</tr>
<tr>
  <th>bathrooms</th>         <td>4006.3214</td>   <td>919.1677</td>  <td>4.3586</td>  <td>0.0000</td>  <td>2204.6018</td>   <td>5808.0409</td> 
</tr>
<tr>
  <th>sqft_living</th>      <td>57514.2433</td>   <td>1008.2400</td> <td>57.0442</td> <td>0.0000</td> <td>55537.9275</td>  <td>59490.5591</td> 
</tr>
<tr>
  <th>floors</th>           <td>-7602.7410</td>   <td>836.6282</td>  <td>-9.0874</td> <td>0.0000</td> <td>-9242.6695</td>  <td>-5962.8125</td> 
</tr>
<tr>
  <th>waterfront</th>        <td>5088.3292</td>   <td>600.9064</td>  <td>8.4678</td>  <td>0.0000</td>  <td>3910.4541</td>   <td>6266.2043</td> 
</tr>
<tr>
  <th>condition</th>         <td>9541.0686</td>   <td>629.4496</td>  <td>15.1578</td> <td>0.0000</td>  <td>8307.2442</td>  <td>10774.8930</td> 
</tr>
<tr>
  <th>grade</th>            <td>26074.4898</td>   <td>818.0044</td>  <td>31.8757</td> <td>0.0000</td> <td>24471.0669</td>  <td>27677.9126</td> 
</tr>
<tr>
  <th>renovated</th>         <td>1469.4406</td>   <td>585.5021</td>  <td>2.5097</td>  <td>0.0121</td>  <td>321.7605</td>    <td>2617.1207</td> 
</tr>
<tr>
  <th>basement_present</th> <td>-5546.0095</td>   <td>721.8063</td>  <td>-7.6835</td> <td>0.0000</td> <td>-6960.8682</td>  <td>-4131.1507</td> 
</tr>
<tr>
  <th>zipcode_98002</th>    <td>-12900.2509</td>  <td>6672.4427</td> <td>-1.9334</td> <td>0.0532</td> <td>-25979.3332</td>  <td>178.8315</td>  
</tr>
<tr>
  <th>zipcode_98003</th>     <td>1740.5176</td>   <td>6073.0427</td> <td>0.2866</td>  <td>0.7744</td> <td>-10163.6424</td> <td>13644.6776</td> 
</tr>
<tr>
  <th>zipcode_98004</th>    <td>371396.6894</td> <td>16329.9337</td> <td>22.7433</td> <td>0.0000</td> <td>339387.3405</td> <td>403406.0382</td>
</tr>
<tr>
  <th>zipcode_98005</th>    <td>274013.6629</td> <td>11045.7926</td> <td>24.8071</td> <td>0.0000</td> <td>252362.0975</td> <td>295665.2283</td>
</tr>
<tr>
  <th>zipcode_98006</th>    <td>205246.6476</td>  <td>6840.1134</td> <td>30.0063</td> <td>0.0000</td> <td>191838.9032</td> <td>218654.3920</td>
</tr>
<tr>
  <th>zipcode_98007</th>    <td>217263.7113</td>  <td>8409.3343</td> <td>25.8360</td> <td>0.0000</td> <td>200780.0365</td> <td>233747.3860</td>
</tr>
<tr>
  <th>zipcode_98008</th>    <td>218106.1136</td>  <td>6419.7078</td> <td>33.9745</td> <td>0.0000</td> <td>205522.4332</td> <td>230689.7940</td>
</tr>
<tr>
  <th>zipcode_98010</th>    <td>81149.0139</td>   <td>8837.8775</td> <td>9.1820</td>  <td>0.0000</td> <td>63825.3243</td>  <td>98472.7036</td> 
</tr>
<tr>
  <th>zipcode_98011</th>    <td>152927.1362</td>  <td>6972.9153</td> <td>21.9316</td> <td>0.0000</td> <td>139259.0784</td> <td>166595.1941</td>
</tr>
<tr>
  <th>zipcode_98014</th>    <td>109662.0470</td>  <td>8779.5741</td> <td>12.4906</td> <td>0.0000</td> <td>92452.6414</td>  <td>126871.4525</td>
</tr>
<tr>
  <th>zipcode_98019</th>    <td>109764.3159</td>  <td>6680.2900</td> <td>16.4311</td> <td>0.0000</td> <td>96669.8517</td>  <td>122858.7801</td>
</tr>
<tr>
  <th>zipcode_98022</th>    <td>27124.5154</td>   <td>6326.2211</td> <td>4.2876</td>  <td>0.0000</td> <td>14724.0843</td>  <td>39524.9465</td> 
</tr>
<tr>
  <th>zipcode_98023</th>    <td>-15749.6244</td>  <td>5246.0488</td> <td>-3.0022</td> <td>0.0027</td> <td>-26032.7406</td> <td>-5466.5082</td> 
</tr>
<tr>
  <th>zipcode_98024</th>    <td>146195.0964</td> <td>10266.6456</td> <td>14.2398</td> <td>0.0000</td> <td>126070.7867</td> <td>166319.4060</td>
</tr>
<tr>
  <th>zipcode_98027</th>    <td>176260.9931</td>  <td>5977.1620</td> <td>29.4891</td> <td>0.0000</td> <td>164544.7751</td> <td>187977.2110</td>
</tr>
<tr>
  <th>zipcode_98028</th>    <td>136320.8952</td>  <td>6044.3505</td> <td>22.5534</td> <td>0.0000</td> <td>124472.9768</td> <td>148168.8137</td>
</tr>
<tr>
  <th>zipcode_98029</th>    <td>212325.4807</td>  <td>6367.1670</td> <td>33.3469</td> <td>0.0000</td> <td>199844.7890</td> <td>224806.1725</td>
</tr>
<tr>
  <th>zipcode_98030</th>     <td>7039.3493</td>   <td>6236.4236</td> <td>1.1287</td>  <td>0.2590</td> <td>-5185.0639</td>  <td>19263.7626</td> 
</tr>
<tr>
  <th>zipcode_98031</th>     <td>8528.6375</td>   <td>5947.0968</td> <td>1.4341</td>  <td>0.1516</td> <td>-3128.6478</td>  <td>20185.9228</td> 
</tr>
<tr>
  <th>zipcode_98032</th>    <td>-12228.0316</td>  <td>8048.1848</td> <td>-1.5194</td> <td>0.1287</td> <td>-28003.7941</td>  <td>3547.7309</td> 
</tr>
<tr>
  <th>zipcode_98033</th>    <td>243677.5076</td>  <td>6552.8165</td> <td>37.1867</td> <td>0.0000</td> <td>230832.9123</td> <td>256522.1030</td>
</tr>
<tr>
  <th>zipcode_98034</th>    <td>164507.8973</td>  <td>5296.9041</td> <td>31.0574</td> <td>0.0000</td> <td>154125.0964</td> <td>174890.6982</td>
</tr>
<tr>
  <th>zipcode_98038</th>    <td>46302.0157</td>   <td>5160.2523</td> <td>8.9728</td>  <td>0.0000</td> <td>36187.0748</td>  <td>56416.9567</td> 
</tr>
<tr>
  <th>zipcode_98040</th>    <td>326034.5377</td> <td>15875.7881</td> <td>20.5366</td> <td>0.0000</td> <td>294915.3887</td> <td>357153.6868</td>
</tr>
<tr>
  <th>zipcode_98042</th>    <td>12705.2104</td>   <td>5169.8264</td> <td>2.4576</td>  <td>0.0140</td>  <td>2571.5026</td>  <td>22838.9181</td> 
</tr>
<tr>
  <th>zipcode_98045</th>    <td>98823.0902</td>   <td>6641.1928</td> <td>14.8803</td> <td>0.0000</td> <td>85805.2629</td>  <td>111840.9175</td>
</tr>
<tr>
  <th>zipcode_98052</th>    <td>215402.6310</td>  <td>5726.3997</td> <td>37.6157</td> <td>0.0000</td> <td>204177.9482</td> <td>226627.3139</td>
</tr>
<tr>
  <th>zipcode_98053</th>    <td>206380.7334</td>  <td>6463.6923</td> <td>31.9292</td> <td>0.0000</td> <td>193710.8361</td> <td>219050.6307</td>
</tr>
<tr>
  <th>zipcode_98055</th>    <td>39586.2889</td>   <td>6104.4362</td> <td>6.4848</td>  <td>0.0000</td> <td>27620.5925</td>  <td>51551.9853</td> 
</tr>
<tr>
  <th>zipcode_98056</th>    <td>89444.2979</td>   <td>5560.3379</td> <td>16.0861</td> <td>0.0000</td> <td>78545.1235</td>  <td>100343.4723</td>
</tr>
<tr>
  <th>zipcode_98058</th>    <td>40893.6048</td>   <td>5385.8663</td> <td>7.5928</td>  <td>0.0000</td> <td>30336.4233</td>  <td>51450.7863</td> 
</tr>
<tr>
  <th>zipcode_98059</th>    <td>94956.6570</td>   <td>5502.1248</td> <td>17.2582</td> <td>0.0000</td> <td>84171.5898</td>  <td>105741.7241</td>
</tr>
<tr>
  <th>zipcode_98065</th>    <td>147969.0730</td>  <td>6142.4173</td> <td>24.0897</td> <td>0.0000</td> <td>135928.9274</td> <td>160009.2186</td>
</tr>
<tr>
  <th>zipcode_98070</th>    <td>140768.2153</td>  <td>8714.9958</td> <td>16.1524</td> <td>0.0000</td> <td>123685.3938</td> <td>157851.0367</td>
</tr>
<tr>
  <th>zipcode_98072</th>    <td>164199.6052</td>  <td>6510.4262</td> <td>25.2210</td> <td>0.0000</td> <td>151438.1018</td> <td>176961.1086</td>
</tr>
<tr>
  <th>zipcode_98074</th>    <td>201284.7308</td>  <td>6243.5589</td> <td>32.2388</td> <td>0.0000</td> <td>189046.3312</td> <td>213523.1305</td>
</tr>
<tr>
  <th>zipcode_98075</th>    <td>234327.9686</td>  <td>8190.9122</td> <td>28.6083</td> <td>0.0000</td> <td>218272.4370</td> <td>250383.5001</td>
</tr>
<tr>
  <th>zipcode_98077</th>    <td>162743.2665</td>  <td>8442.3655</td> <td>19.2770</td> <td>0.0000</td> <td>146194.8452</td> <td>179291.6877</td>
</tr>
<tr>
  <th>zipcode_98092</th>     <td>1469.8656</td>   <td>5764.3941</td> <td>0.2550</td>  <td>0.7987</td> <td>-9829.2924</td>  <td>12769.0236</td> 
</tr>
<tr>
  <th>zipcode_98102</th>    <td>297088.5562</td> <td>11696.2982</td> <td>25.4002</td> <td>0.0000</td> <td>274161.8930</td> <td>320015.2194</td>
</tr>
<tr>
  <th>zipcode_98103</th>    <td>244078.3724</td>  <td>5508.9174</td> <td>44.3061</td> <td>0.0000</td> <td>233279.9906</td> <td>254876.7543</td>
</tr>
<tr>
  <th>zipcode_98105</th>    <td>305088.9218</td>  <td>8436.7538</td> <td>36.1619</td> <td>0.0000</td> <td>288551.5003</td> <td>321626.3433</td>
</tr>
<tr>
  <th>zipcode_98106</th>    <td>93248.0402</td>   <td>5853.3725</td> <td>15.9307</td> <td>0.0000</td> <td>81774.4699</td>  <td>104721.6106</td>
</tr>
<tr>
  <th>zipcode_98107</th>    <td>281918.8926</td>  <td>6487.5995</td> <td>43.4550</td> <td>0.0000</td> <td>269202.1333</td> <td>294635.6520</td>
</tr>
<tr>
  <th>zipcode_98108</th>    <td>102847.1230</td>  <td>6800.2672</td> <td>15.1240</td> <td>0.0000</td> <td>89517.4839</td>  <td>116176.7622</td>
</tr>
<tr>
  <th>zipcode_98109</th>    <td>311859.4786</td> <td>12871.0024</td> <td>24.2296</td> <td>0.0000</td> <td>286630.2026</td> <td>337088.7547</td>
</tr>
<tr>
  <th>zipcode_98112</th>    <td>303265.9515</td>  <td>9963.8199</td> <td>30.4367</td> <td>0.0000</td> <td>283735.2300</td> <td>322796.6731</td>
</tr>
<tr>
  <th>zipcode_98115</th>    <td>256862.6724</td>  <td>5559.8617</td> <td>46.1995</td> <td>0.0000</td> <td>245964.4313</td> <td>267760.9134</td>
</tr>
<tr>
  <th>zipcode_98116</th>    <td>253719.7551</td>  <td>6430.9779</td> <td>39.4527</td> <td>0.0000</td> <td>241113.9835</td> <td>266325.5268</td>
</tr>
<tr>
  <th>zipcode_98117</th>    <td>258242.1319</td>  <td>5454.1378</td> <td>47.3479</td> <td>0.0000</td> <td>247551.1272</td> <td>268933.1366</td>
</tr>
<tr>
  <th>zipcode_98118</th>    <td>134570.4171</td>  <td>5496.9936</td> <td>24.4807</td> <td>0.0000</td> <td>123795.4079</td> <td>145345.4263</td>
</tr>
<tr>
  <th>zipcode_98119</th>    <td>298357.7687</td>  <td>9594.9537</td> <td>31.0953</td> <td>0.0000</td> <td>279550.0854</td> <td>317165.4520</td>
</tr>
<tr>
  <th>zipcode_98122</th>    <td>241509.4396</td>  <td>6810.3713</td> <td>35.4620</td> <td>0.0000</td> <td>228159.9947</td> <td>254858.8845</td>
</tr>
<tr>
  <th>zipcode_98125</th>    <td>168574.7553</td>  <td>5663.5136</td> <td>29.7650</td> <td>0.0000</td> <td>157473.3396</td> <td>179676.1710</td>
</tr>
<tr>
  <th>zipcode_98126</th>    <td>161024.4470</td>  <td>5796.9920</td> <td>27.7772</td> <td>0.0000</td> <td>149661.3916</td> <td>172387.5023</td>
</tr>
<tr>
  <th>zipcode_98133</th>    <td>126406.2882</td>  <td>5269.6185</td> <td>23.9877</td> <td>0.0000</td> <td>116076.9714</td> <td>136735.6050</td>
</tr>
<tr>
  <th>zipcode_98136</th>    <td>210242.1271</td>  <td>6579.1944</td> <td>31.9556</td> <td>0.0000</td> <td>197345.8268</td> <td>223138.4275</td>
</tr>
<tr>
  <th>zipcode_98144</th>    <td>181420.9715</td>  <td>6201.6122</td> <td>29.2538</td> <td>0.0000</td> <td>169264.7942</td> <td>193577.1488</td>
</tr>
<tr>
  <th>zipcode_98146</th>    <td>84728.6154</td>   <td>6252.1137</td> <td>13.5520</td> <td>0.0000</td> <td>72473.4470</td>  <td>96983.7839</td> 
</tr>
<tr>
  <th>zipcode_98148</th>    <td>35848.0297</td>  <td>10909.3826</td> <td>3.2860</td>  <td>0.0010</td> <td>14463.8502</td>  <td>57232.2092</td> 
</tr>
<tr>
  <th>zipcode_98155</th>    <td>124424.9608</td>  <td>5408.6518</td> <td>23.0048</td> <td>0.0000</td> <td>113823.1159</td> <td>135026.8057</td>
</tr>
<tr>
  <th>zipcode_98166</th>    <td>101217.5922</td>  <td>6410.7730</td> <td>15.7887</td> <td>0.0000</td> <td>88651.4254</td>  <td>113783.7589</td>
</tr>
<tr>
  <th>zipcode_98168</th>    <td>27999.3152</td>   <td>6444.6792</td> <td>4.3446</td>  <td>0.0000</td> <td>15366.6869</td>  <td>40631.9436</td> 
</tr>
<tr>
  <th>zipcode_98177</th>    <td>185891.7024</td>  <td>7048.3104</td> <td>26.3739</td> <td>0.0000</td> <td>172075.8577</td> <td>199707.5471</td>
</tr>
<tr>
  <th>zipcode_98178</th>    <td>41204.4668</td>   <td>6272.8717</td> <td>6.5687</td>  <td>0.0000</td> <td>28908.6093</td>  <td>53500.3244</td> 
</tr>
<tr>
  <th>zipcode_98188</th>    <td>24276.5388</td>   <td>7679.7197</td> <td>3.1611</td>  <td>0.0016</td>  <td>9223.0284</td>  <td>39330.0491</td> 
</tr>
<tr>
  <th>zipcode_98198</th>    <td>31612.4188</td>   <td>6226.1773</td> <td>5.0773</td>  <td>0.0000</td> <td>19408.0899</td>  <td>43816.7477</td> 
</tr>
<tr>
  <th>zipcode_98199</th>    <td>267319.0664</td>  <td>7404.9794</td> <td>36.0999</td> <td>0.0000</td> <td>252804.0919</td> <td>281834.0408</td>
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>561.223</td>  <td>Durbin-Watson:</td>     <td>2.024</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>1486.527</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.243</td>      <td>Prob(JB):</td>       <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>4.659</td>   <td>Condition No.:</td>      <td>100</td>  
</tr>
</table>




```python
# Results sorted by coefficients descending 

results_as_html = model_final.summary().tables[1].as_html()
results = pd.read_html(results1_as_html, header=0, index_col=0)[0]
results.sort_values('coef', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coef</th>
      <th>std err</th>
      <th>t</th>
      <th>P&gt;|t|</th>
      <th>[0.025</th>
      <th>0.975]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>zipcode_98004</th>
      <td>371400.0000</td>
      <td>16300.000</td>
      <td>22.743</td>
      <td>0.000</td>
      <td>339000.000</td>
      <td>403000.000</td>
    </tr>
    <tr>
      <th>zipcode_98040</th>
      <td>326000.0000</td>
      <td>15900.000</td>
      <td>20.537</td>
      <td>0.000</td>
      <td>295000.000</td>
      <td>357000.000</td>
    </tr>
    <tr>
      <th>zipcode_98109</th>
      <td>311900.0000</td>
      <td>12900.000</td>
      <td>24.230</td>
      <td>0.000</td>
      <td>287000.000</td>
      <td>337000.000</td>
    </tr>
    <tr>
      <th>zipcode_98105</th>
      <td>305100.0000</td>
      <td>8436.754</td>
      <td>36.162</td>
      <td>0.000</td>
      <td>289000.000</td>
      <td>322000.000</td>
    </tr>
    <tr>
      <th>zipcode_98112</th>
      <td>303300.0000</td>
      <td>9963.820</td>
      <td>30.437</td>
      <td>0.000</td>
      <td>284000.000</td>
      <td>323000.000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>basement_present</th>
      <td>-5546.0095</td>
      <td>721.806</td>
      <td>-7.684</td>
      <td>0.000</td>
      <td>-6960.868</td>
      <td>-4131.151</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>-7602.7410</td>
      <td>836.628</td>
      <td>-9.087</td>
      <td>0.000</td>
      <td>-9242.669</td>
      <td>-5962.812</td>
    </tr>
    <tr>
      <th>zipcode_98032</th>
      <td>-12230.0000</td>
      <td>8048.185</td>
      <td>-1.519</td>
      <td>0.129</td>
      <td>-28000.000</td>
      <td>3547.731</td>
    </tr>
    <tr>
      <th>zipcode_98002</th>
      <td>-12900.0000</td>
      <td>6672.443</td>
      <td>-1.933</td>
      <td>0.053</td>
      <td>-26000.000</td>
      <td>178.832</td>
    </tr>
    <tr>
      <th>zipcode_98023</th>
      <td>-15750.0000</td>
      <td>5246.049</td>
      <td>-3.002</td>
      <td>0.003</td>
      <td>-26000.000</td>
      <td>-5466.508</td>
    </tr>
  </tbody>
</table>
<p>78 rows × 6 columns</p>
</div>



### Check Linear Model Assumptions

**1. Linearity** <br>
- sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)

**2. Residual Normality** <br>
- sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True) <br>
- Omnibus Value

**3. Homoskedasticity** <br>
- Durbin-Watson: range of 1.5 to 2.5 is relatively normal

**4. Multicollinearity** <br>
- VIF (variance_inflation_factor()) <br>
- Anything above 10 needs to be removed 

**Also check p-value** <br>
- A p-value less than 0.05 (typically ≤ 0.05) is statistically significant. It indicates strong evidence against the null hypothesis, as there is less than a 5% probability the null is correct (and the results are random).

**Check for overfitting** <br>
- Test vs. Train Data, compare: 
1. Mean Absolute Error (MAE)
2. Mean Sqaured Error (MSE)
3. Root Mean Sqaured Error (RMSE)

## Check for Linearity and Residual Normality using Q-Q Plot 
- There are some tails but overall residuals appear normal 


```python
# Check Linearity and Residual Normality 

sm.graphics.qqplot(model_final.resid, dist=stats.norm, line='45', fit=True);
```


    
<img src='images/Residual Normality.png'>
    


## Check for Multicolinearity using VIF (Variance Inflation Factor) 
- Removed all variance inflation factors above 10 
- All remaining are below 10 


```python
# Check for multicolinearity - anything over 10 is not good! 

create_vif_dct(X_train_scaled)
```




    {'const': 48.804888593432594,
     'bedrooms': 1.7525266115446538,
     'bathrooms': 2.520977861021298,
     'sqft_living': 3.0332438247636637,
     'floors': 2.0885480888237606,
     'waterfront': 1.0774400957148569,
     'condition': 1.1822282938705688,
     'grade': 1.996598740009202,
     'renovated': 1.0229074802205151,
     'basement_present': 1.554607741508546,
     'zipcode_98002': 1.572374400097395,
     'zipcode_98003': 1.7683916273984623,
     'zipcode_98004': 1.065100539678895,
     'zipcode_98005': 1.1552525010993069,
     'zipcode_98006': 1.5495760787475503,
     'zipcode_98007': 1.2999857058283855,
     'zipcode_98008': 1.6562704322361503,
     'zipcode_98010': 1.2621828886369568,
     'zipcode_98011': 1.4914193011733052,
     'zipcode_98014': 1.264640922782956,
     'zipcode_98019': 1.5651862005750914,
     'zipcode_98022': 1.6764572920324579,
     'zipcode_98023': 2.401885345658481,
     'zipcode_98024': 1.1811733768623345,
     'zipcode_98027': 1.8251705776763347,
     'zipcode_98028': 1.7782165922916775,
     'zipcode_98029': 1.688382860504492,
     'zipcode_98030': 1.695277852470107,
     'zipcode_98031': 1.8239180673146302,
     'zipcode_98032': 1.3345289185972118,
     'zipcode_98033': 1.6002524120784773,
     'zipcode_98034': 2.336472752494877,
     'zipcode_98038': 2.529762947645856,
     'zipcode_98040': 1.0695100160422801,
     'zipcode_98042': 2.48919498398572,
     'zipcode_98045': 1.5684405390947247,
     'zipcode_98052': 1.9670133379258323,
     'zipcode_98053': 1.6383968948327174,
     'zipcode_98055': 1.7506671291350397,
     'zipcode_98056': 2.08370547501707,
     'zipcode_98058': 2.230646819307026,
     'zipcode_98059': 2.1268217530668387,
     'zipcode_98065': 1.772519797918147,
     'zipcode_98070': 1.3587003781601552,
     'zipcode_98072': 1.610588276152504,
     'zipcode_98074': 1.7180666400873077,
     'zipcode_98075': 1.316109939412652,
     'zipcode_98077': 1.2926215328810982,
     'zipcode_98092': 1.9374125918788754,
     'zipcode_98102': 1.1593679933535124,
     'zipcode_98103': 2.3265534775693357,
     'zipcode_98105': 1.3084770106186392,
     'zipcode_98106': 1.9318565444070566,
     'zipcode_98107': 1.7119511051672336,
     'zipcode_98108': 1.5428738521577468,
     'zipcode_98109': 1.1155508114641512,
     'zipcode_98112': 1.2110043603791862,
     'zipcode_98115': 2.164337059909012,
     'zipcode_98116': 1.6821988102483125,
     'zipcode_98117': 2.280513896798148,
     'zipcode_98118': 2.1946814064237246,
     'zipcode_98119': 1.2370712597262954,
     'zipcode_98122': 1.5701214411554572,
     'zipcode_98125': 2.0238874056833405,
     'zipcode_98126': 1.9755145425871357,
     'zipcode_98133': 2.4169932736595734,
     'zipcode_98136': 1.6342507660544252,
     'zipcode_98144': 1.7882327681733308,
     'zipcode_98146': 1.713299220605527,
     'zipcode_98148': 1.1564530569895037,
     'zipcode_98155': 2.228779395730326,
     'zipcode_98166': 1.651663330462584,
     'zipcode_98168': 1.6590811305558477,
     'zipcode_98177': 1.487356216897902,
     'zipcode_98178': 1.7056064002303757,
     'zipcode_98188': 1.3748946030663856,
     'zipcode_98198': 1.7179126771024968,
     'zipcode_98199': 1.4534675049413095}



## Check for Homoskedasticity

- Durbin-Watson: range of 1.5 to 2.5 is relatively normal
- Model's Durbin-Watson is 2.024

## Check for Over-fitting
- Check expected vs. predicted errors of Train Test sets 
- Train and Test data are within range of each other 
- The average expected error (mean absolute error) of the Train data is \\$47,496 while the aveage expected error of the Test data is \\$48,684


```python
# Predictions on the training set

y_train_pred = model1.predict(X_train_scaled) 
y_test_pred = model1.predict(X_test_scaled)

# Regression metrics for Train data

train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

# Regression metrics for Test data

test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print('Train MAE:', train_mae)
print('Test MAE:', test_mae)

print('Train MSE:', train_mse)
print('Test MSE:', test_mse)

print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)
```

    Train MAE: 47495.546251964
    Test MAE: 48683.51726441247
    Train MSE: 3974372692.5231338
    Test MSE: 4208824782.898014
    Train RMSE: 63042.62599640924
    Test RMSE: 64875.45593595481


## Look at Unscaled/Raw Data for Price Expectations for Each Feature 


```python
X_train_raw = sm.add_constant(X_train_raw)
X_test_raw = sm.add_constant(X_test_raw)

model_unscaled = sm.OLS(y_train, X_train_raw).fit()
model_unscaled.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.740</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>297876.8343</td>
</tr>
<tr>
         <td>Date:</td>        <td>2021-11-07 12:33</td>        <td>BIC:</td>         <td>298453.0513</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>11937</td>        <td>Log-Likelihood:</td>   <td>-1.4886e+05</td>
</tr>
<tr>
       <td>Df Model:</td>             <td>77</td>           <td>F-statistic:</td>        <td>442.0</td>   
</tr>
<tr>
     <td>Df Residuals:</td>          <td>11859</td>      <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.742</td>            <td>Scale:</td>        <td>4.0005e+09</td> 
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>Coef.</th>     <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>    <th>[0.025</th>       <th>0.975]</th>   
</tr>
<tr>
  <th>const</th>            <td>-149137.4013</td>  <td>8569.2272</td> <td>-17.4038</td> <td>0.0000</td> <td>-165934.4924</td> <td>-132340.3102</td>
</tr>
<tr>
  <th>bedrooms</th>          <td>-4531.8375</td>   <td>919.3137</td>   <td>-4.9296</td> <td>0.0000</td>  <td>-6333.8432</td>   <td>-2729.8318</td> 
</tr>
<tr>
  <th>bathrooms</th>          <td>6163.4048</td>   <td>1414.0660</td>  <td>4.3586</td>  <td>0.0000</td>   <td>3391.6034</td>    <td>8935.2061</td> 
</tr>
<tr>
  <th>sqft_living</th>         <td>93.8186</td>     <td>1.6447</td>    <td>57.0442</td> <td>0.0000</td>    <td>90.5948</td>      <td>97.0424</td>  
</tr>
<tr>
  <th>floors</th>            <td>-14164.5223</td>  <td>1558.7061</td>  <td>-9.0874</td> <td>0.0000</td>  <td>-17219.8420</td>  <td>-11109.2026</td>
</tr>
<tr>
  <th>waterfront</th>        <td>148666.6664</td> <td>17556.7947</td>  <td>8.4678</td>  <td>0.0000</td>  <td>114252.4686</td>  <td>183080.8641</td>
</tr>
<tr>
  <th>condition</th>         <td>15007.3150</td>   <td>990.0723</td>   <td>15.1578</td> <td>0.0000</td>  <td>13066.6108</td>   <td>16948.0191</td> 
</tr>
<tr>
  <th>grade</th>             <td>31571.1934</td>   <td>990.4461</td>   <td>31.8757</td> <td>0.0000</td>  <td>29629.7566</td>   <td>33512.6302</td> 
</tr>
<tr>
  <th>renovated</th>          <td>9812.2096</td>   <td>3909.6980</td>  <td>2.5097</td>  <td>0.0121</td>   <td>2148.5602</td>   <td>17475.8591</td> 
</tr>
<tr>
  <th>basement_present</th>  <td>-11543.8567</td>  <td>1502.4188</td>  <td>-7.6835</td> <td>0.0000</td>  <td>-14488.8439</td>  <td>-8598.8695</td> 
</tr>
<tr>
  <th>zipcode_98002</th>     <td>-12900.2509</td>  <td>6672.4427</td>  <td>-1.9334</td> <td>0.0532</td>  <td>-25979.3332</td>   <td>178.8315</td>  
</tr>
<tr>
  <th>zipcode_98003</th>      <td>1740.5176</td>   <td>6073.0427</td>  <td>0.2866</td>  <td>0.7744</td>  <td>-10163.6424</td>  <td>13644.6776</td> 
</tr>
<tr>
  <th>zipcode_98004</th>     <td>371396.6894</td> <td>16329.9337</td>  <td>22.7433</td> <td>0.0000</td>  <td>339387.3405</td>  <td>403406.0382</td>
</tr>
<tr>
  <th>zipcode_98005</th>     <td>274013.6629</td> <td>11045.7926</td>  <td>24.8071</td> <td>0.0000</td>  <td>252362.0975</td>  <td>295665.2283</td>
</tr>
<tr>
  <th>zipcode_98006</th>     <td>205246.6476</td>  <td>6840.1134</td>  <td>30.0063</td> <td>0.0000</td>  <td>191838.9032</td>  <td>218654.3920</td>
</tr>
<tr>
  <th>zipcode_98007</th>     <td>217263.7113</td>  <td>8409.3343</td>  <td>25.8360</td> <td>0.0000</td>  <td>200780.0365</td>  <td>233747.3860</td>
</tr>
<tr>
  <th>zipcode_98008</th>     <td>218106.1136</td>  <td>6419.7078</td>  <td>33.9745</td> <td>0.0000</td>  <td>205522.4332</td>  <td>230689.7940</td>
</tr>
<tr>
  <th>zipcode_98010</th>     <td>81149.0139</td>   <td>8837.8775</td>  <td>9.1820</td>  <td>0.0000</td>  <td>63825.3243</td>   <td>98472.7036</td> 
</tr>
<tr>
  <th>zipcode_98011</th>     <td>152927.1362</td>  <td>6972.9153</td>  <td>21.9316</td> <td>0.0000</td>  <td>139259.0784</td>  <td>166595.1941</td>
</tr>
<tr>
  <th>zipcode_98014</th>     <td>109662.0470</td>  <td>8779.5741</td>  <td>12.4906</td> <td>0.0000</td>  <td>92452.6414</td>   <td>126871.4525</td>
</tr>
<tr>
  <th>zipcode_98019</th>     <td>109764.3159</td>  <td>6680.2900</td>  <td>16.4311</td> <td>0.0000</td>  <td>96669.8517</td>   <td>122858.7801</td>
</tr>
<tr>
  <th>zipcode_98022</th>     <td>27124.5154</td>   <td>6326.2211</td>  <td>4.2876</td>  <td>0.0000</td>  <td>14724.0843</td>   <td>39524.9465</td> 
</tr>
<tr>
  <th>zipcode_98023</th>     <td>-15749.6244</td>  <td>5246.0488</td>  <td>-3.0022</td> <td>0.0027</td>  <td>-26032.7406</td>  <td>-5466.5082</td> 
</tr>
<tr>
  <th>zipcode_98024</th>     <td>146195.0964</td> <td>10266.6456</td>  <td>14.2398</td> <td>0.0000</td>  <td>126070.7867</td>  <td>166319.4060</td>
</tr>
<tr>
  <th>zipcode_98027</th>     <td>176260.9931</td>  <td>5977.1620</td>  <td>29.4891</td> <td>0.0000</td>  <td>164544.7751</td>  <td>187977.2110</td>
</tr>
<tr>
  <th>zipcode_98028</th>     <td>136320.8952</td>  <td>6044.3505</td>  <td>22.5534</td> <td>0.0000</td>  <td>124472.9768</td>  <td>148168.8137</td>
</tr>
<tr>
  <th>zipcode_98029</th>     <td>212325.4807</td>  <td>6367.1670</td>  <td>33.3469</td> <td>0.0000</td>  <td>199844.7890</td>  <td>224806.1725</td>
</tr>
<tr>
  <th>zipcode_98030</th>      <td>7039.3493</td>   <td>6236.4236</td>  <td>1.1287</td>  <td>0.2590</td>  <td>-5185.0639</td>   <td>19263.7626</td> 
</tr>
<tr>
  <th>zipcode_98031</th>      <td>8528.6375</td>   <td>5947.0968</td>  <td>1.4341</td>  <td>0.1516</td>  <td>-3128.6478</td>   <td>20185.9228</td> 
</tr>
<tr>
  <th>zipcode_98032</th>     <td>-12228.0316</td>  <td>8048.1848</td>  <td>-1.5194</td> <td>0.1287</td>  <td>-28003.7941</td>   <td>3547.7309</td> 
</tr>
<tr>
  <th>zipcode_98033</th>     <td>243677.5076</td>  <td>6552.8165</td>  <td>37.1867</td> <td>0.0000</td>  <td>230832.9123</td>  <td>256522.1030</td>
</tr>
<tr>
  <th>zipcode_98034</th>     <td>164507.8973</td>  <td>5296.9041</td>  <td>31.0574</td> <td>0.0000</td>  <td>154125.0964</td>  <td>174890.6982</td>
</tr>
<tr>
  <th>zipcode_98038</th>     <td>46302.0157</td>   <td>5160.2523</td>  <td>8.9728</td>  <td>0.0000</td>  <td>36187.0748</td>   <td>56416.9567</td> 
</tr>
<tr>
  <th>zipcode_98040</th>     <td>326034.5377</td> <td>15875.7881</td>  <td>20.5366</td> <td>0.0000</td>  <td>294915.3887</td>  <td>357153.6868</td>
</tr>
<tr>
  <th>zipcode_98042</th>     <td>12705.2104</td>   <td>5169.8264</td>  <td>2.4576</td>  <td>0.0140</td>   <td>2571.5026</td>   <td>22838.9181</td> 
</tr>
<tr>
  <th>zipcode_98045</th>     <td>98823.0902</td>   <td>6641.1928</td>  <td>14.8803</td> <td>0.0000</td>  <td>85805.2629</td>   <td>111840.9175</td>
</tr>
<tr>
  <th>zipcode_98052</th>     <td>215402.6310</td>  <td>5726.3997</td>  <td>37.6157</td> <td>0.0000</td>  <td>204177.9482</td>  <td>226627.3139</td>
</tr>
<tr>
  <th>zipcode_98053</th>     <td>206380.7334</td>  <td>6463.6923</td>  <td>31.9292</td> <td>0.0000</td>  <td>193710.8361</td>  <td>219050.6307</td>
</tr>
<tr>
  <th>zipcode_98055</th>     <td>39586.2889</td>   <td>6104.4362</td>  <td>6.4848</td>  <td>0.0000</td>  <td>27620.5925</td>   <td>51551.9853</td> 
</tr>
<tr>
  <th>zipcode_98056</th>     <td>89444.2979</td>   <td>5560.3379</td>  <td>16.0861</td> <td>0.0000</td>  <td>78545.1235</td>   <td>100343.4723</td>
</tr>
<tr>
  <th>zipcode_98058</th>     <td>40893.6048</td>   <td>5385.8663</td>  <td>7.5928</td>  <td>0.0000</td>  <td>30336.4233</td>   <td>51450.7863</td> 
</tr>
<tr>
  <th>zipcode_98059</th>     <td>94956.6570</td>   <td>5502.1248</td>  <td>17.2582</td> <td>0.0000</td>  <td>84171.5898</td>   <td>105741.7241</td>
</tr>
<tr>
  <th>zipcode_98065</th>     <td>147969.0730</td>  <td>6142.4173</td>  <td>24.0897</td> <td>0.0000</td>  <td>135928.9274</td>  <td>160009.2186</td>
</tr>
<tr>
  <th>zipcode_98070</th>     <td>140768.2153</td>  <td>8714.9958</td>  <td>16.1524</td> <td>0.0000</td>  <td>123685.3938</td>  <td>157851.0367</td>
</tr>
<tr>
  <th>zipcode_98072</th>     <td>164199.6052</td>  <td>6510.4262</td>  <td>25.2210</td> <td>0.0000</td>  <td>151438.1018</td>  <td>176961.1086</td>
</tr>
<tr>
  <th>zipcode_98074</th>     <td>201284.7308</td>  <td>6243.5589</td>  <td>32.2388</td> <td>0.0000</td>  <td>189046.3312</td>  <td>213523.1305</td>
</tr>
<tr>
  <th>zipcode_98075</th>     <td>234327.9686</td>  <td>8190.9122</td>  <td>28.6083</td> <td>0.0000</td>  <td>218272.4370</td>  <td>250383.5001</td>
</tr>
<tr>
  <th>zipcode_98077</th>     <td>162743.2665</td>  <td>8442.3655</td>  <td>19.2770</td> <td>0.0000</td>  <td>146194.8452</td>  <td>179291.6877</td>
</tr>
<tr>
  <th>zipcode_98092</th>      <td>1469.8656</td>   <td>5764.3941</td>  <td>0.2550</td>  <td>0.7987</td>  <td>-9829.2924</td>   <td>12769.0236</td> 
</tr>
<tr>
  <th>zipcode_98102</th>     <td>297088.5562</td> <td>11696.2982</td>  <td>25.4002</td> <td>0.0000</td>  <td>274161.8930</td>  <td>320015.2194</td>
</tr>
<tr>
  <th>zipcode_98103</th>     <td>244078.3724</td>  <td>5508.9174</td>  <td>44.3061</td> <td>0.0000</td>  <td>233279.9906</td>  <td>254876.7543</td>
</tr>
<tr>
  <th>zipcode_98105</th>     <td>305088.9218</td>  <td>8436.7538</td>  <td>36.1619</td> <td>0.0000</td>  <td>288551.5003</td>  <td>321626.3433</td>
</tr>
<tr>
  <th>zipcode_98106</th>     <td>93248.0402</td>   <td>5853.3725</td>  <td>15.9307</td> <td>0.0000</td>  <td>81774.4699</td>   <td>104721.6106</td>
</tr>
<tr>
  <th>zipcode_98107</th>     <td>281918.8926</td>  <td>6487.5995</td>  <td>43.4550</td> <td>0.0000</td>  <td>269202.1333</td>  <td>294635.6520</td>
</tr>
<tr>
  <th>zipcode_98108</th>     <td>102847.1230</td>  <td>6800.2672</td>  <td>15.1240</td> <td>0.0000</td>  <td>89517.4839</td>   <td>116176.7622</td>
</tr>
<tr>
  <th>zipcode_98109</th>     <td>311859.4786</td> <td>12871.0024</td>  <td>24.2296</td> <td>0.0000</td>  <td>286630.2026</td>  <td>337088.7547</td>
</tr>
<tr>
  <th>zipcode_98112</th>     <td>303265.9515</td>  <td>9963.8199</td>  <td>30.4367</td> <td>0.0000</td>  <td>283735.2300</td>  <td>322796.6731</td>
</tr>
<tr>
  <th>zipcode_98115</th>     <td>256862.6724</td>  <td>5559.8617</td>  <td>46.1995</td> <td>0.0000</td>  <td>245964.4313</td>  <td>267760.9134</td>
</tr>
<tr>
  <th>zipcode_98116</th>     <td>253719.7551</td>  <td>6430.9779</td>  <td>39.4527</td> <td>0.0000</td>  <td>241113.9835</td>  <td>266325.5268</td>
</tr>
<tr>
  <th>zipcode_98117</th>     <td>258242.1319</td>  <td>5454.1378</td>  <td>47.3479</td> <td>0.0000</td>  <td>247551.1272</td>  <td>268933.1366</td>
</tr>
<tr>
  <th>zipcode_98118</th>     <td>134570.4171</td>  <td>5496.9936</td>  <td>24.4807</td> <td>0.0000</td>  <td>123795.4079</td>  <td>145345.4263</td>
</tr>
<tr>
  <th>zipcode_98119</th>     <td>298357.7687</td>  <td>9594.9537</td>  <td>31.0953</td> <td>0.0000</td>  <td>279550.0854</td>  <td>317165.4520</td>
</tr>
<tr>
  <th>zipcode_98122</th>     <td>241509.4396</td>  <td>6810.3713</td>  <td>35.4620</td> <td>0.0000</td>  <td>228159.9947</td>  <td>254858.8845</td>
</tr>
<tr>
  <th>zipcode_98125</th>     <td>168574.7553</td>  <td>5663.5136</td>  <td>29.7650</td> <td>0.0000</td>  <td>157473.3396</td>  <td>179676.1710</td>
</tr>
<tr>
  <th>zipcode_98126</th>     <td>161024.4470</td>  <td>5796.9920</td>  <td>27.7772</td> <td>0.0000</td>  <td>149661.3916</td>  <td>172387.5023</td>
</tr>
<tr>
  <th>zipcode_98133</th>     <td>126406.2882</td>  <td>5269.6185</td>  <td>23.9877</td> <td>0.0000</td>  <td>116076.9714</td>  <td>136735.6050</td>
</tr>
<tr>
  <th>zipcode_98136</th>     <td>210242.1271</td>  <td>6579.1944</td>  <td>31.9556</td> <td>0.0000</td>  <td>197345.8268</td>  <td>223138.4275</td>
</tr>
<tr>
  <th>zipcode_98144</th>     <td>181420.9715</td>  <td>6201.6122</td>  <td>29.2538</td> <td>0.0000</td>  <td>169264.7942</td>  <td>193577.1488</td>
</tr>
<tr>
  <th>zipcode_98146</th>     <td>84728.6154</td>   <td>6252.1137</td>  <td>13.5520</td> <td>0.0000</td>  <td>72473.4470</td>   <td>96983.7839</td> 
</tr>
<tr>
  <th>zipcode_98148</th>     <td>35848.0297</td>  <td>10909.3826</td>  <td>3.2860</td>  <td>0.0010</td>  <td>14463.8502</td>   <td>57232.2092</td> 
</tr>
<tr>
  <th>zipcode_98155</th>     <td>124424.9608</td>  <td>5408.6518</td>  <td>23.0048</td> <td>0.0000</td>  <td>113823.1159</td>  <td>135026.8057</td>
</tr>
<tr>
  <th>zipcode_98166</th>     <td>101217.5922</td>  <td>6410.7730</td>  <td>15.7887</td> <td>0.0000</td>  <td>88651.4254</td>   <td>113783.7589</td>
</tr>
<tr>
  <th>zipcode_98168</th>     <td>27999.3152</td>   <td>6444.6792</td>  <td>4.3446</td>  <td>0.0000</td>  <td>15366.6869</td>   <td>40631.9436</td> 
</tr>
<tr>
  <th>zipcode_98177</th>     <td>185891.7024</td>  <td>7048.3104</td>  <td>26.3739</td> <td>0.0000</td>  <td>172075.8577</td>  <td>199707.5471</td>
</tr>
<tr>
  <th>zipcode_98178</th>     <td>41204.4668</td>   <td>6272.8717</td>  <td>6.5687</td>  <td>0.0000</td>  <td>28908.6093</td>   <td>53500.3244</td> 
</tr>
<tr>
  <th>zipcode_98188</th>     <td>24276.5388</td>   <td>7679.7197</td>  <td>3.1611</td>  <td>0.0016</td>   <td>9223.0284</td>   <td>39330.0491</td> 
</tr>
<tr>
  <th>zipcode_98198</th>     <td>31612.4188</td>   <td>6226.1773</td>  <td>5.0773</td>  <td>0.0000</td>  <td>19408.0899</td>   <td>43816.7477</td> 
</tr>
<tr>
  <th>zipcode_98199</th>     <td>267319.0664</td>  <td>7404.9794</td>  <td>36.0999</td> <td>0.0000</td>  <td>252804.0919</td>  <td>281834.0408</td>
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>561.223</td>  <td>Durbin-Watson:</td>     <td>2.024</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>1486.527</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.243</td>      <td>Prob(JB):</td>       <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>4.659</td>   <td>Condition No.:</td>    <td>111890</td> 
</tr>
</table>



# Evaluation and Conclusions

After building models to evaluate the relationship between price and several factors, we can offer guidance to new home buyers in WA State about the expectation of price relative to square feet of living, waterfront views, condition, and grade. 

**** Important note: the results are best suited for home buyers seeking homes with a maximum of 6 bedrooms, 4000 square feet, and a budget ranging from \\$175,000 to \\$650,000

**Conclusions** 
- **The most important factors in our model, besides zipcode, are: Square Feet Living, Grade, and Condition**
- The **average price for a home in King County, WA is approximately \\$400,358**
- **Every additional square feet of space costs approximately \\$94. Note: other models showed this cost could be up to $200 per square feet in the densest zipcodes** 
- The grade of a home (1-13) is a strong determinant of price. **Every grade increase costs approximately \\$31,571** 
- The condition (1-5) is also a strong determinant; **Every condition level increase costs approximately \\$15,007** 
- **If the home has been renovated, the price is expected to be approximately \\$9812 more**

<img src='images/Condition Grade and Residential Building Grades.png'>

# Future Work 

**Future work:** 
* Refine existing models and expand dataset for different types of home buyers 
* Explore relationship of price to zip code 
* Build models for Suburbs (Medina, WA) vs. City (Seattle, WA)
* Build more comprehensive models considering other factors such as location, renovations, waterfront view 
