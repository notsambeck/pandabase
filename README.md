![pandabase_logo](https://github.com/notsambeck/notsambeck.github.io/blob/master/media/pandabase_logo.jpg?raw=true)

##### pip install pandabase

[![Build Status](https://travis-ci.org/notsambeck/pandabase.svg?branch=master)](https://travis-ci.org/notsambeck/pandabase)
[![Coverage Status](https://coveralls.io/repos/github/notsambeck/pandabase/badge.svg?branch=master)](https://coveralls.io/github/notsambeck/pandabase?branch=master)

### DEPRECATION WARNING:

### pandabase is OK, but you should probably look at [Pangres](https://github.com/ThibTrip/pangres) instead

Pangres is a similar tool, but better written. It is generally faster. It doesn't have as many database management features, but they would be pretty easy to add...

### Description

pandabase links pandas DataFrames to SQL databases, supporting read, append, upsert, and basic database management operations. 

By default, pandabase uses DataFrame.index as the primary key. Using an explicit primary key makes rational database schemas the obvious choice, and makes it easy to maintain clean data even when it must be updated frequently. 

Designed specifically for time-series datasets that need to be stored to disk permanently, but are updated over time and used primarily in-memory for computation. All supported types are nullable, ideal for ML applications.

Tested under:
* Python >= 3.6
* Pandas >= 0.24, including 1.0
* SQLAlchemy >= 1.3 
* SQLite
* Postgres
    * requires psycopg2 and postgres >= 8

### Features
* pandabase.to_sql(df, ...) replaces df.to_sql(...)
* pandabase.read_sql(...)   replaces pd.read_sql(...)
* primary key support:
    * by default, uses df.index as table PK (must have name != None)
    * filter results with lowest/highest kwargs: lowest <= results.pk <= highest 
    * (new in 0.4): support for multi-indexes
    * optionally, generate integer index (with parameter auto_index=True)
* multiple insert modes: how='create_only', 'upsert', or 'append'
* datatypes (all nullable): 
    * boolean
    * int
    * float
    * datetime (UTC only)
    * string

### Bonus Features
* moderately smart insertion handles new records that 'almost correspond' with database schema automatically
* to_sql can automatically add new columns to database as needed with kwarg: add_new_columns=True
* supports arbitrary schemas in Postgres with kwarg: schema=name
* comprehensive test suite (pytest)
* companda(df1, df2) test tool: rich comparisons of DataFrames

### Design Considerations
* Minimal dependencies: Pandas (>= 0.24) & SQLAlchemy (>= 1.3) are the only requirements
* Database is the source of truth: will coerce incoming DataFrames to fit existing schema
  * but also is reasonably smart about how new tables are created from DataFrames
* Not horrendously slow (?)

### License
MIT license

### Thanks
Code partially stolen from:
[Dataset](https://github.com/pudo/dataset) (nice, more general-purpose SQL interaction library) and 
[pandas.sql](https://github.com/pandas-dev/pandas/blob/master/pandas/io/sql.py)

See also:
[Pangres](https://github.com/ThibTrip/pangres) like Pandabase, but faster.

### Installation
From your virtual environment of choice (including Conda):

```bash
~/$ pip install pandabase
```

For latest version:

```bash
~/$ git clone https://github.com/notsambeck/pandabase
~/$ cd pandabase
~/pandabase/$ pip install -r requirements.txt
~/pandabase/$ pip install .
```

### Usage
```python
# Python >= 3.6
>>> import pandas as pd
>>> import numpy as np
>>> import pandabase
>>> my_data = pd.DataFrame(index=range(7, 12), 
                           columns=['some_number'],
                           data=np.random.random((5,1)))
>>> my_data.index.name = 'my_index_name'        # index must be named to use as PK
>>> pandabase.to_sql(my_data, table_name='my_table', con='sqlite:///new_sqlite_db.sqlite', how='create_only')
Table('my_table', ...
>>> exit()
```

Your data is now persistently stored in a SQLite database, using my_data.index as primary key. To append or update data, replace 'create_only' with 'append' or 'upsert'. To store records without an explicit index, use 'autoindex=True'.

```bash
~/pandabase$ ls
new_sqlite_db.sqlite
```

```python
>>> import pandabase
>>> df = pandabase.read_sql('my_table', con='sqlite:///new_sqlite_db.sqlite'))
>>> df
    some_number 
7   0.722416 
8   0.076045 
9   0.213118 
10  0.453716 
11  0.406995
```

### Usage notes & recommendations:

#### Engines vs. strings
All methods accept either a string or sqlalchemy.Engine for argument 'con' (i.e. database connection).
Using a string works, but the connection may not be returned to the connection pool at transaction end.
Eventually, this may exhaust the connection pool.
For applications, always pass an engine object to pandabase.to_sql and pandabase.read_sql. Example:

```python
>>> import pandabase
>>> engine = pandabase.engine_builder('postgresql+psycopg2://postgres:postgres@localhost:5432/testdb')
>>> pandabase.to_sql(df=df, con=engine, table_name='table0')   # to use default schema=None => 'public'
>>> pandabase.to_sql(df=df, con=engine, table_name='table0', schema='my_schema')   #  access my_schema.table
```


#### Keyword arguments for pandabase.read_sql:

* [lowest, highest]: minimum/maximum values for PK that will be retrieved.
    * Can be used independently of each other.
    * For multi-index tables, use a tuple of values in order.
    * e.g. `pandabase.to_sql(con=con, table_name='multi_index_table', highest=(max_value_for_pk0, max_value_for_pk1, ), lowest=(min_value_for_pk0, min_value_for_pk1, )`
* schema: string, schema for Postgres
    * e.g. for `pandabase.to_sql(con=con, table_name='bare_table', schema='my_schema')   # myschema.bare_table`
* add_new_columns: bool, default False. if True, add columns to database as necessary to match incoming DataFrame
    * e.g. for `pandabase.to_sql(con=con, table_name='table0', add_new_columns=True)`
* how: ['create_only', 'append', or 'upsert']


Usage note up to 0.4.5 release.
selecting an empty subset of data will raise an error if type(lowest) != type(data), 
even if the types are comparable, e.g. float vs. int.

(New behavior will return a DataFrame with length zero for  

### Using Extra Features
Companda - rich comparisons of DataFrames. call companda on two DataFrames, get a Companda object back (that evaluates to True/False).

```python
>>> from pandabse.companda import companda
>>> df = pandabase.read_sql('my_table', con='sqlite:///new_sqlite_db.sqlite'))

>>> companda(df, df.copy())
Companda(True, message='DataFrames are equal')
>>> bool(companda(df, df.copy()))
True

>>> df2 = df.copy()
>>> df2.iloc[1, 2] = -1000

>>> companda(df, df2)
Companda(False, message='Columns and indices are equal, but unequal values in columns [col_a]...')
>>> bool(companda(df, df2))
False
```

### Table utility functions:

Under basic use cases, Pandabase can handle database administration tasks. All support schema=name kwarg in Postgres.

* drop_db_table(table_name, con):
    * Drop table [table_name] from con - be careful with this!
* get_db_table_names(con):
    * Get a list of table names from database.
* get_table_column_names(con, table_name):
    * Get a list of column names from database, table.
* describe_database(con):
    * Get a description of database content: {table_names: {table_info_dicts}}.
