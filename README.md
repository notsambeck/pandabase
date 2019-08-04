# pandabase
[![Build Status](https://travis-ci.org/notsambeck/pandabase.svg?branch=master)](https://travis-ci.org/notsambeck/pandabase)

pandabase converts pandas DataFrames to &amp; from SQLite

* adds explicit declaration of primary keys
* allows upsert statements 
* replaces pd.DataFrame.to_sql and pd.read_sql
* is much simpler than pandas.sql
* is not guaranteed to work with all database backends, but could be extended

designed specifically for time series datasets that need to be updated with new and stored to disk,
but are used in-memory for computation.

### Design Considerations
* Minimal dependencies: SQLAlchemy and Pandas are the only requirements
* Database is the source of truth: will coerce incoming DataFrames to fit existing schema
* Reasonably performant
