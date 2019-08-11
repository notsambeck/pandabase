"""
pandabase converts pandas DataFrames to & from SQL databases

It replaces pandas.to_sql and pandas.read_sql, and requires the user
to select a unique index. This allows upserts and makes it easier to
maintain a dataset that grows over time. Especially time series.

pandabase:
    is simpler than pandas.io.sql
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

import sqlalchemy as sqa
from sqlalchemy import Table, and_
from sqlalchemy.exc import IntegrityError
import pytz
import logging


def to_sql(df: pd.DataFrame, *,
           table_name: str,
           con: str or sqa.engine,
           auto_index=False,
           how='create_only',
           add_new_columns=False, ):
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
    auto_index: bool, default False. if True, ignore existing df.index, make a new id
    add_new_columns: bool, default False. if True, add any new columns as required by the dataframe.
    how : {'create_only', 'upsert', 'append'}, default 'create_only'
        - create_only:
            If table exists, raise an error and stop.
        - append:
            If table exists, append data. Raise if index overlaps
            Create table if does not exist.
        - upsert:
            create table if needed
            if record exists: update
            else: insert
    """
    # 1. make connection objects
    df = df.copy()

    engine = engine_builder(con)
    meta = sqa.MetaData()

    ######################
    # 2. validate inputs #
    ######################
    clean_table_name = clean_name(table_name)
    if clean_table_name != table_name:
        raise NameError(f'Illegal characters in table name: {table_name}. try: {clean_table_name}')

    if how not in ('create_only', 'append', 'upsert',):
        raise ValueError(f"{how} is not a valid value for parameter: 'how'")

    if not isinstance(df, pd.DataFrame):
        raise ValueError('to_sql() requires a DataFrame as input')

    if not auto_index:
        if not df.index.is_unique:
            raise ValueError('DataFrame index is not unique.')
        if df.index.hasnans:
            raise ValueError('DataFrame index has NaN values.')
        if is_datetime64_any_dtype(df.index):
            if df.index.tz != pytz.utc:
                raise ValueError(f'Index {df.index.name} is not UTC. Please correct.')
        if df.index.name is None:
            raise NameError('Autoindex is turned off, but df.index.name is None. Please correct.')
        df.index.name = clean_name(df.index.name)
    else:
        df.index.name = PANDABASE_DEFAULT_INDEX

    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz != pytz.utc:
                raise ValueError(f'Column {col} is not set as UTC. Please correct.')
            # else:
            # print(col, 'tzinfo =', df[col].dt.tz)

    # make a list of df columns for later:
    df_cols_dict = make_clean_columns_dict(df, autoindex=auto_index)

    #############################################
    # 3a. Make new Table from df info if needed #
    #############################################
    if not has_table(engine, table_name):
        logging.info(f'Creating new table {table_name}')
        table = Table(table_name, meta,
                      *[make_column(name, info) for name, info in df_cols_dict.items()
                        if info['dtype'] is not None])
    #######################################################################################
    # 3b. Or make Table from db schema; DB will be the source of truth for datatypes etc. #
    #######################################################################################
    else:
        if how == 'create_only':
            raise NameError(f'Table {table_name} already exists; param "how" is set to "create_only".')

        table = Table(table_name, meta, autoload=True, autoload_with=engine)

        if how == 'upsert':
            if table.primary_key == PANDABASE_DEFAULT_INDEX or auto_index:
                raise IOError('Cannot upsert with an automatic index')

        # 3. iterate over df_columns; confirm that types are compatible and all columns exist
        for col_name, df_col_info in df_cols_dict.items():
            if col_name not in table.columns:
                if df_col_info['dtype'] is None:
                    continue
                elif add_new_columns:
                    logging.info(f'adding new column to {con}:{table_name}: {col_name}')
                    add_columns_to_db(make_column(col_name, df_col_info), table_name=table_name, con=con)
                    meta.clear()
                    table = Table(table_name, meta, autoload=True, autoload_with=engine)

                else:
                    raise NameError(f'New data has at least one column that does not exist in DB: {col_name}. \n'
                                    f'Set add_new_columns to True to automatically fix.')

            # check that dtypes and PKs match for existing columns
            col = table.columns[col_name]
            if col.primary_key != df_col_info['pk']:
                raise NameError(f'Inconsistent pk for col: {col_name}! db: {col.primary_key} / '
                                f'df: {df_col_info["pk"]}')

            db_sqla_dtype = get_column_dtype(col, pd_or_sqla='sqla')

            # 3c. check datatypes
            if db_sqla_dtype == df_col_info['dtype']:
                continue

            ############################################
            # 3d. COERCE BAD DATATYPES - case by case #
            ############################################
            db_pandas_dtype = get_column_dtype(col, pd_or_sqla='pd')

            if df_col_info['dtype'] is None:
                # this does not need to be explicitly handled because when inserting None, nothing happens
                continue
            elif is_datetime64_any_dtype(db_pandas_dtype):
                df[col_name] = pd.to_datetime(df[col_name].values, utc=True)

            elif (
                    df_col_info['dtype'] == Integer and is_float_dtype(db_pandas_dtype)) or (
                    df_col_info['dtype'] == Float and is_integer_dtype(db_pandas_dtype)
            ):
                # print(f'NUMERIC DTYPE: converting df[{name}] from {df[name].dtype} to {db_pandas_dtype}')
                df[col_name] = df[col_name].astype(db_pandas_dtype)
                # print(f'new dtypes: {df.dtypes}')
            else:
                raise TypeError(
                    f'Inconsistent type for column: {col_name} \n'
                    f'db.{col_name}.dtype= {db_pandas_dtype} / '
                    f'df{col_name}.dtype= {df_col_info["dtype"]}')

    #######################################################
    # FINALLY: either insert/fail, append/fail, or upsert #
    #######################################################

    # print('DB connection begins...')
    with engine.begin() as con:
        meta.create_all(bind=con)

    if how in ['append', 'create_only']:
        # will raise IntegrityError if repeated index encountered
        with engine.begin() as con:
            rows = []
            df = df.dropna(axis=1, how='all')
            if not auto_index:
                for index, row in df.iterrows():
                    rows.append({**row.to_dict(), df.index.name: index})
                con.execute(table.insert(), rows)
            else:
                for index, row in df.iterrows():
                    rows.append({**row.to_dict()})
                con.execute(table.insert(), rows)

    elif how == 'upsert':
        with engine.begin() as con:
            for index, row in df.iterrows():
                # check index uniqueness by attempting insert; if it fails, update
                row = {**row.dropna().to_dict(), df.index.name: index}
                try:
                    insert = table.insert().values(row)
                    con.execute(insert)

                except IntegrityError:
                    upsert = table.update() \
                        .where(table.c[df.index.name] == index) \
                        .values(row)
                    con.execute(upsert)

    return table


def read_sql(table_name: str,
             con: str or sqa.engine,
             *,
             lowest=None, highest=None):
    """
    Read in a table from con as a pd.DataFrame, preserving dtypes and primary keys

    :param table_name: str
    :param con: db connectable
    :param lowest: minimum index value to select (inclusive)
    :param highest: maximum index value to select (inclusive)
    """
    engine = engine_builder(con)
    meta = sqa.MetaData(bind=engine)
    table = Table(table_name, meta, autoload=True, autoload_with=engine)

    if len(table.primary_key.columns) == 0:
        print('no index')
        assert lowest is None
        assert highest is None
        result = engine.execute(table.select())
        data = result.fetchall()

    elif len(table.primary_key.columns) == 1:
        pk = table.primary_key.columns.items()[0][1]

        if highest is None:
            if lowest is None:
                s = table.select()
            else:
                s = table.select().where(pk >= lowest)
        else:
            if lowest is None:
                s = table.select().where(pk <= highest)
            else:
                s = table.select().where(and_(pk >= lowest,
                                              pk <= highest))
        result = engine.execute(s)
        data = result.fetchall()

        if len(data) == 0:
            if not isinstance(lowest, pk.type.python_type) or not isinstance(highest, pk.type.python_type):
                raise TypeError(f'Select range is: {lowest} <= data <= {highest}; type of column is {pk.type}')
    else:
        raise NotImplementedError('pandabase is not compatible with multi-index tables')

    df = pd.DataFrame.from_records(data, columns=[col.name for col in table.columns],
                                   coerce_float=True)

    for col in table.columns:
        # deal with primary key first; never convert primary key to nullable
        if col.primary_key:
            # print(f'index column is {col.name}')
            df.index = df[col.name]
            dtype = get_column_dtype(col, pd_or_sqla='pd', index=True)
            # force all dates to utc
            if is_datetime64_any_dtype(dtype):
                # print(df.index.tz, 'PK - old...')
                df.index = pd.to_datetime(df[col.name].values, utc=True)
                # print(df.index.tz, 'PK - new')

            if col.name == PANDABASE_DEFAULT_INDEX:
                df.index.name = None
            else:
                df.index.name = col.name

            df = df.drop(columns=[col.name])
            continue
        else:
            # print(f'non-pk column: {col}')
            dtype = get_column_dtype(col, pd_or_sqla='pd')
            # force all dates to utc
            if is_datetime64_any_dtype(dtype):
                # print(df[col.name].dt.tz, 'regular col - old...')
                df[col.name] = pd.to_datetime(df[col.name].values, utc=True)
                # print(df[col.name].dt.tz, 'regular col - new')

        # convert other dtypes to nullable
        if is_bool_dtype(dtype) or is_integer_dtype(dtype):
            df[col.name] = np.array(df[col.name], dtype=float)
            df[col.name] = df[col.name].astype(pd.Int64Dtype())
        elif is_float_dtype(dtype):
            pass
        elif is_string_dtype(col):
            pass

    return df


def add_columns_to_db(new_col, table_name, con):
    # Make any new columns as needed with ALTER TABLE
    engine = engine_builder(con)
    name = clean_name(new_col.name)

    with engine.begin() as conn:
        conn.execute(f'ALTER TABLE {table_name} '
                     f'ADD COLUMN {name} {new_col.type.compile(engine.dialect)}')


def read_db_table_names(con):
    """get a list of table names from database"""
    meta = sqa.MetaData()
    meta.reflect(engine_builder(con))
    return list(meta.tables.keys())


def read_db_table_column_names(con, table_name):
    """get a list of table names from database"""
    meta = sqa.MetaData()
    meta.reflect(engine_builder(con))
    return list(meta.tables[table_name].columns)
