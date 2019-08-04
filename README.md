# pandabase
[![Build Status](https://travis-ci.org/notsambeck/pandabase.svg?branch=master)](https://travis-ci.org/notsambeck/pandabase)

pandabase converts pandas DataFrames to &amp; from SQLite

### Features
* primary keys (any named index is assumed to be the PK)
* allows upsert
* also includes companda, a tool for comparing DataFrames
* replaces pd.DataFrame.to_sql and pd.read_sql
* is much simpler than pandas.sql module
* is not guaranteed to work with all database backends, but could be extended

designed for time series datasets that need to be updated over time and stored to disk,
but are used in-memory for computation.

### Design Considerations
* Minimal dependencies: SQLAlchemy and Pandas are the only requirements
* Database is the source of truth: will coerce incoming DataFrames to fit existing schema
* Reasonably fast

### License
MIT license

### Thanks
Code partially stolen from Dataset and pandas.sql
