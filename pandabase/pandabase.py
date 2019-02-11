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

import pandas as pd
from pandas.api.types import (is_bool_dtype,
                              is_datetime64_any_dtype,
                              is_integer_dtype,
                              is_float_dtype)

from pytz import UTC

import sqlalchemy as sqa
from sqlalchemy import Table, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.exc import IntegrityError

import logging


PANDABASE_DEFAULT_INDEX_NAME = 'pandabase_index'


def engine_builder(con):
    """
    Returns a SQLAlchemy engine from a URI (if con is a string)
    else it just return con without modifying it.
    """
    if isinstance(con, str):
        con = sqa.create_engine(con)

    return con


def get_sql_dtype(series):
    """
    Take a pd.Series or column of DataFrame, return its SQLAlchemy datatype
    If it doesn't match anything, return String
    :param series: pd.Series
    :return: one of {Integer, Float, Boolean, DateTime, String}
    """
    if is_bool_dtype(series):
        return Boolean
    elif is_integer_dtype(series):
        return Integer
    elif is_float_dtype(series):
        return Float
    elif is_datetime64_any_dtype(series):
        return DateTime
    else:
        return String


def has_table(con, table_name):
    """pandas.sql.has_table()"""
    engine = engine_builder(con)
    return engine.run_callable(engine.dialect.has_table, table_name)


def to_sql(df: pd.DataFrame, *,
           index_col_name: str or None,
           table_name: str,
           con: str or sqa.engine,
           how='fail',
           strict='false', ):
    """
    Write records stored in a DataFrame to a SQL database, converting any datetime to UTC

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
    index_col_name : name of column to use as index, or None to new range_index named pandabase_index
        (Applied to both DataFrame and sql database)
    strict: default False; if True, fail instead of coercing anything
    """
    ##########################################
    # 1. make connection objects; check inputs

    engine = engine_builder(con)
    meta = sqa.MetaData()

    if how not in ('fail', 'append', 'upsert',):
        raise ValueError("'{0}' is not valid for if_exists".format(how))

    if not isinstance(df, pd.DataFrame):
        raise ValueError('to_sql() requires a DataFrame as input')

    if index_col_name is None:
        index_col_name = PANDABASE_DEFAULT_INDEX_NAME
        df = df.reindex()
        df[index_col_name] = df.index
    else:
        index_col_name = clean_name(index_col_name)

    # convert any non-tz-aware datetimes to utc using pd.to_datetime (warn)
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz != UTC:
                if strict:
                    raise ValueError(f'Strict=True; column {col} is tz-naive. Please correct manually.')
                else:
                    logging.warning(f'{col} was stored in tz-naive format; automatically converted to UTC')
                df[col] = pd.to_datetime(df[col], utc=True)

    df_columns = _make_clean_columns_dict(df, index_col_name)

    df.index = df[index_col_name]  # this raises if invalid
    if not df.index.is_unique:
        raise ValueError('DataFrame index must be unique; otherwise use index=None to add integer PK')
        # we will check that index_col_name is in db.table later (after we have reflected db)

    ############################################################################
    # 2a. get existing table metadata from db, add any new columns from df to db

    if has_table(engine, table_name):
        if how == 'fail':
            raise NameError(f'Table {table_name} already exists; param if_exists is set to "fail".')

        table = Table(table_name, meta, autoload=True, autoload_with=engine)
        if index_col_name not in [col.name for col in table.columns]:
            raise ValueError('index_col_name is not in existing database table')
        if how == 'upsert' and index_col_name == PANDABASE_DEFAULT_INDEX_NAME:
            raise IOError('Cannot upsert with a made-up index!')

        new_cols = []
        for name in df_columns.keys():
            # check dtypes and PKs match
            if name in table.columns:
                if table.columns[name].primary_key != df_columns[name].primary_key:
                    raise ValueError(f'Inconsistent pk for {name}! db: {table.columns[name].primary_key} / '
                                     f'df: {df_columns[name].primary_key}')
                # column.type is not directly comparable, use str(type) or type.python_type?
                if table.columns[name].type.python_type != df_columns[name].type.python_type:
                    raise ValueError(f'Inconsistent type for {name}! db: {table.columns[name].type} / '
                                     f'df: {df_columns[name].type}')
            # add new columns if needed (sqla does not support migrations directly)
            else:
                new_cols.append(Column(name, get_sql_dtype(df[name])))

        if len(new_cols):
            col_names_string = ", ".join([col.name for col in new_cols])
            # TODO: confirm sqla automatically adds NaN here
            new_column_warning = f' Table[{table_name}]:' + \
                                 f' new Series [{col_names_string}] added around index {df[index_col_name][0]}'
            if strict:
                raise ValueError(new_column_warning)
            else:
                logging.warning(new_column_warning)

            for new_col in new_cols:
                with engine.begin() as conn:
                    conn.execute(f'ALTER TABLE {table_name} '
                                 f'ADD COLUMN {new_col.name} {new_col.type.compile(engine.dialect)}')
            # TODO: new columns are added, but data is not written to them

    # 2b. unless it's a brand-new table
    else:
        logging.info(f'Creating new table {table_name}')
        table = Table(table_name, meta, *df_columns.values())

    #####################################
    # 3. create any new tables or columns

    with engine.begin() as con:
        meta.create_all(bind=con)

    ######################################################
    # FINALLY: either insert/fail, append/fail, or upsert

    if how in ['append', 'fail']:
        # raise if repeated index
        with engine.begin() as con:
            con.execute(table.insert(), [row.to_dict() for i, row in df.iterrows()])

    elif how == 'upsert':
        # explicitly check index uniqueness
        for i in df.index:
            try:
                with engine.begin() as con:
                    insert = table.insert().values(df.loc[i].to_dict())
                    con.execute(insert)
            except IntegrityError:
                print('sqla Integrity Error!')
            except:
                if strict:
                    raise IOError('?????????? unknown')
                print('Unknown error!')

            with engine.begin() as con:
                upsert = table.update() \
                    .where(table.c[index_col_name] == i) \
                    .values(df[[col for col in df.columns if col != index_col_name]].loc[i].to_dict())
                con.execute(upsert)

    return table


def clean_name(name):
    return name.lower().strip().replace(' ', '_')


def _make_clean_columns_dict(df: pd.DataFrame, index_col_name):
    """Take a DataFrame and index_col_name (or None), return a dictionary {column_name: new Column}"""
    columns = {}
    df.columns = [clean_name(col) for col in df.columns]

    if index_col_name is not None:
        assert index_col_name in df.columns
    else:
        columns['id'] = Column('id', Integer, primary_key=True)

    for col_name in df.columns:
        pk = index_col_name == col_name
        columns[col_name] = Column(col_name, get_sql_dtype(df[col_name]), primary_key=pk, )

    assert len(columns) > 1
    return columns


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
    dtypes = {}
    for col in table.columns:
        dtypes[col.name] = col.type.python_type
        if col.primary_key:
            index_col = col.name

    # make selector
    if columns is not None:
        if index_col not in columns:
            raise NameError(f'User supplied columns do not include index col: {index_col}')
        selector = []
        for col in columns:
            selector.append(table.c[col])
        s = sqa.select(selector)

    else:
        s = sqa.select([table])

    datetime_cols = list([key for key, value in dtypes.items() if value == pd.datetime])

    df = pd.read_sql_query(s, engine, index_col=index_col,
                           parse_dates={col: {'utc': True} for col in datetime_cols})
    if index_col != PANDABASE_DEFAULT_INDEX_NAME:
        df[index_col] = df.index

    return df
