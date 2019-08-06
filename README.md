# pandabase
[![Build Status](https://travis-ci.org/notsambeck/pandabase.svg?branch=master)](https://travis-ci.org/notsambeck/pandabase)

pandabase converts pandas DataFrames to &amp; from SQLite

Tested under Python 3.6 and 3.7, with new versions of Pandas and SQLAlchemy

### Features
* primary keys (any named index is assumed to be the PK)
* allows upsert
* also includes companda, a tool for comparing DataFrames
* replaces pd.DataFrame.to_sql and pd.read_sql for some use cases
* is simpler than pandas.sql module
* is not guaranteed to work with all database backends (but could be extended...)
* automated tests in pytest

designed for time series datasets that need to be updated over time and stored to disk,
but are used in-memory for computation.

### Design Considerations
* Minimal dependencies: SQLAlchemy and Pandas are the only requirements
* Database is the source of truth: will coerce incoming DataFrames to fit existing schema
* Not horrendously slow

### License
MIT license

### Thanks
Code partially stolen from Dataset and pandas.sql

### Installation

From your inside your virtual environment of choice:

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
>>> import pandabase
>>> my_data = pd.DataFrame(index=range(7, 12), 
                           columns=['some_number'],
                           data=pd.np.random.random((5,1)))
>>> my_data.index.name = 'made_up_name'        # index must be named to use as PK
>>> pandabase.to_sql(my_data, table_name='my_table', con='sqlite:///new_sqlite_db.sqlite', how='create_only')
Table('my_table', ...
>>> exit()
```

That's all! 

Your data is now persistently stored in a SQLite database, using my_data.index as primary key. To append or update data, replace 'create_only' with 'append' or 'upsert'. To store records without and explicit index, use 'autoindex=True'.

```bash
~/pandabase$ ls
brand_new_sqlite_db.sqlite
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
