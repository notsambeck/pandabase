# pandabase
pandabase converts pandas DataFrames to &amp; from SQL databases

* adds explicit declaration of primary keys
* allows upsert statements 
* replaces pd.DataFrame.to_sql and pd.read_sql
* is much simpler than pandas sql module (but is not guaranteed to work with all database backends)

designed specifically for time series datasets that need to be updated frequently and stored to disk, but are accessed for computations in-memory.

### Design Considerations
* Minimal dependencies: SQLAlchemy and Pandas only
* Database is source of truth
    * coerce any DataFrames to fit existing schema
    * except for new columns which may be added late