"""
pandabase converts pandas DataFrames to & from SQL databases

It replaces pandas.to_sql and pandas.read_sql, and requires the user
to select a unique index. This allows upserts and makes it easier to
maintain a dataset that grows over time. Especially time series.

pandabase:
    is much simpler than pandas.io.sql
    is only compatible with newest versions of Pandas & sqlalchemy
    is not guaranteed
    definitely supports sqlite, may or may support other backends
    uses the sqlalchemy core and Pandas; has no additional dependencies

by sam beck
github.com/notsambeck/pandabase

largely copied from pandas:
https://github.com/pandas-dev/pandas
and dataset:
https://github.com/pudo/dataset/
"""
from .helpers import *

import pandas as pd
from pandas.api.types import is_string_dtype

from pytz import UTC

import sqlalchemy as sqa
from sqlalchemy import Table
from sqlalchemy.exc import IntegrityError

import logging


def to_sql(df: pd.DataFrame, *,
           table_name: str,
           con: str or sqa.engine,
           how='fail',
           strict=True, ):
    """
    Write records stored in a DataFrame to a SQL database.

    converts any datetime to UTC
    Requires a unique, named index as DataFrame.index; to insert into existing database, this must be consistent

    Parameters
    ----------
    df : DataFrame, Series
    table_name : string
        Name of SQL table.
    con : connection; database string URI < OR > sa.engine
    how : {'fail', 'upsert', 'append'}, default 'fail'
        - fail:
            If table exists, raise an error and stop.
        - append:
            If table exists, append data. Raise if index overlaps
            Create table if does not exist.
        - upsert:
            create table if needed
            if record exists: update
            else: insert
    use_index: default True; if True, use the index as pk in table, else use an automatic integer index
    strict: default False; if True, fail instead of coercing anything
    """
    ##########################################
    # 1. make connection objects; check inputs
    df = df.copy()
    engine = engine_builder(con)
    meta = sqa.MetaData()

    if how not in ('fail', 'append', 'upsert',):
        raise ValueError("'{0}' is not valid for if_exists".format(how))

    if not isinstance(df, pd.DataFrame):
        raise ValueError('to_sql() requires a DataFrame as input')

    if not df.index.is_unique:
        raise ValueError('DataFrame index is not unique.')
    if df.index.hasnans:
        raise ValueError('DataFrame index has NaN values.')
    if df.index.name is None:
        raise ValueError('DataFrame index needs to have a name (set with df.index.name = >.')

    # make a list of df columns for later:
    df_cols_dict = make_clean_columns_dict(df)

    ###########################################
    # 2. make table from db schema; table will be the reference
    if has_table(engine, table_name):
        if how == 'fail':
            raise NameError(f'Table {table_name} already exists; param if_exists is set to "fail".')

        table = Table(table_name, meta, autoload=True, autoload_with=engine)

        if how == 'upsert' and table.primary_key == PANDABASE_DEFAULT_INDEX:
            raise IOError('Cannot upsert with an automatic index')

    # 2b. unless it's a brand-new table
    else:
        logging.info(f'Creating new table {table_name}')
        table = Table(table_name, meta,
                      *[make_column(name, info) for name, info in df_cols_dict.items()
                        if info['dtype'] is not None])

    ###############################################################
    # 3. iterate over df_columns; confirm that types are compatible and all columns exist
    new_cols = []
    for name, df_col_info in df_cols_dict.items():
        if name not in table.columns:
            if df_col_info['dtype'] is not None:
                new_cols.append(make_column(name, df_col_info))
                continue

            else:
                logging.warning(f'tried to add all NaN column {name}')
                continue

        # check that dtypes and PKs match for existing columns
        col = table.columns[name]
        if col.primary_key != df_col_info['pk']:
            raise ValueError(f'Inconsistent pk for col: {name}! db: {col.primary_key} / '
                             f'df: {df_col_info["pk"]}')

        if df_col_info['dtype'] is None:
            df_col_info['dtype'] = get_column_dtype(col, pd_or_sqla='pd')

        db_sqla_dtype = get_column_dtype(col, pd_or_sqla='sqla')

        # COERCE BAD DATATYPES
        if not db_sqla_dtype == df_col_info['dtype']:
            """
            this is where things become complicated
            
            we know type of existing db_column; this is the correct type for new data
            we know the pandas dtype of df column
            
            this section of code will generally execute if the db dtype is a real type, and df dtype is 'object'
            often as a result of the df column being a non-nullable Pandas dtype (np.int64, np.bool) with Nans, 
            and the db column being the 'true' datatype (possibly plus NULL)
            
            simply coercing the column with .astype(np.bool) etc. also coerces None to 'None"
            
            solution?
            build a mask of NaN values
            coerce columns
            when inserting values to db, 
                first check if value is None, 
                then insert
                
            or:
            don't check types if values in 'object' columns are in fact stored in correct dtype (check this!)
            """
            db_pandas_dtype = get_column_dtype(col, pd_or_sqla='pd')
            if is_datetime64_any_dtype(db_pandas_dtype):
                df[name] = pd.to_datetime(df[name].values, utc=True)
            elif (
                    is_integer_dtype(df_col_info['dtype']) and is_float_dtype(db_pandas_dtype)) or (
                    is_float_dtype(df_col_info['dtype']) and is_integer_dtype(db_pandas_dtype)
            ):
                df[name] = df[name].astype(db_pandas_dtype)
            else:
                raise ValueError(
                    f'Inconsistent type for col: {name}.  '
                    f'db: {db_pandas_dtype} /'
                    f'df: {df_col_info["dtype"]}')

    # Make any new columns as needed with ALTER TABLE
    if new_cols:
        col_names_string = ", ".join([col.name for col in new_cols])
        new_column_warning = f' Table[{table_name}]:' + \
                             f' new Series [{col_names_string}] added around index {df.index[0]}'
        if strict:
            logging.warning(new_column_warning)

        for new_col in new_cols:
            with engine.begin() as conn:
                conn.execute(f'ALTER TABLE {table_name} '
                             f'ADD COLUMN {new_col.name} {new_col.type.compile(engine.dialect)}')

    df.index.name = clean_name(df.index.name)

    # convert any non-tz-aware datetimes to utc using pd.to_datetime (warn)
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]) and df[col].dt.tz != UTC:
            if strict:
                raise ValueError(f'Strict=True; column {col} is tz-naive. Please correct.')
            else:
                logging.warning(f'{col} was stored in tz-naive format; automatically converted to UTC')
            df[col] = pd.to_datetime(df[col].values, utc=True)

    with engine.begin() as con:
        meta.create_all(bind=con)

    ######################################################
    # FINALLY: either insert/fail, append/fail, or upsert

    if how in ['append', 'fail']:
        # raise if repeated index
        with engine.begin() as con:
            rows = []
            for index, row in df.iterrows():
                # append row
                rows.append(row.to_dict())
                # add index name to row
                rows[-1][df.index.name] = index
            con.execute(table.insert(), rows)

    elif how == 'upsert':
        for i in df.index:
            # check index uniqueness by attempting insert; if it fails, update
            try:
                with engine.begin() as con:
                    row = df.loc[i]
                    row[row.isna()] = None
                    values = row.to_dict()
                    values[df.index.name] = i  # set index column to i
                    insert = table.insert().values(values)
                    con.execute(insert)

            except IntegrityError:
                print('Upsert: Integrity Error on insert => do update')

                with engine.begin() as con:
                    row = df.loc[i]
                    row[row.isna()] = None
                    upsert = table.update() \
                        .where(table.c[df.index.name] == i) \
                        .values(row.to_dict())
                    con.execute(upsert)

    return table


def read_sql(table_name: str,
             con: str or sqa.engine,
             columns=None):
    """
    Convenience wrapper around pd.read_sql_query

    Reflect metadata; get Table or Table[columns]

    TODO: add range limit parameters
    :param table_name: str
    :param con: db connectable
    :param columns: list (default None => select all columns)
    """
    engine = engine_builder(con)
    meta = sqa.MetaData(bind=engine)
    table = Table(table_name, meta, autoload=True, autoload_with=engine)

    # find index column, dtypes
    index_col = None

    # make selector from columns, or select whole table
    if columns is not None:
        if index_col not in columns:
            raise NameError(f'User supplied columns do not include index col: {index_col}')
        selector = []
        for col in columns:
            selector.append(table.c[col])
        s = sqa.select(selector)

    else:
        s = sqa.select([table])

    datetime_cols = []
    other_cols = {}
    datetime_index = False

    for col in table.columns:
        dtype = get_column_dtype(col, pd_or_sqla='pd')
        if col.primary_key:
            index_col = col.name
            datetime_index = is_datetime64_any_dtype(dtype)
            continue

        if is_datetime64_any_dtype(dtype):
            datetime_cols.append(col.name)
        else:
            other_cols[col.name] = dtype

    df = pd.read_sql_query(s, engine, index_col=index_col)

    for name, dtype in other_cols.items():
        df[name] = df[name].astype(dtype)

    for name in datetime_cols:
        df[name] = pd.to_datetime(df[name].values, utc=True)

    if datetime_index:
        index = pd.to_datetime(df.index.values, utc=True)
        df.index = index
        df.index.name = index_col

    return df
