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

from pytz import UTC

import sqlalchemy as sqa
from sqlalchemy import Table
from sqlalchemy.exc import IntegrityError

import logging


def to_sql(df: pd.DataFrame, *,
           use_index: bool,
           table_name: str,
           con: str or sqa.engine,
           how='fail',
           strict=True, ):
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
    use_index : True to use df.index, False to new range_index named pandabase_index
        (Applied to both DataFrame and sql database)
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

    if use_index is False:
        df = df.reindex()
        df.index.name = PANDABASE_DEFAULT_INDEX
    else:
        df.index.name = clean_name(df.index.name)

    # convert any non-tz-aware datetimes to utc using pd.to_datetime (warn)
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz != UTC:
                if strict:
                    raise ValueError(f'Strict=True; column {col} is tz-naive. Please correct manually.')
                else:
                    logging.warning(f'{col} was stored in tz-naive format; automatically converted to UTC')
                df[col] = pd.to_datetime(df[col], utc=True)

    df_cols_dict = make_clean_columns_dict(df)

    if not df.index.is_unique or not df.index.notna().all():
        raise ValueError('Specified DataFrame index is not unique; maybe use index=None to add integer as PK')
        # we will check that index_col_name is in db.table later (after we have reflected db)

    ############################################################################
    # 2a. get existing table metadata from db, add any new columns from df to db
    if has_table(engine, table_name):
        if how == 'fail':
            raise NameError(f'Table {table_name} already exists; param if_exists is set to "fail".')

        table = Table(table_name, meta, autoload=True, autoload_with=engine)

        if how == 'upsert' and df.index.name == PANDABASE_DEFAULT_INDEX:
            raise IOError('Cannot upsert with a made-up index!')

        new_cols = []
        for name, col_info in df_cols_dict.items():
            # check that dtypes and PKs match for existing columns
            if name in table.columns:
                if table.columns[name].primary_key != col_info['pk']:
                    raise ValueError(f'Inconsistent pk for col: {name}! db: {table.columns[name].primary_key} / '
                                     f'df: {col_info["pk"]}')

                if col_info['dtype'] is None:
                    print(f'Debug: setting {name} dtype to {table.columns[name].type}')
                    col_info['dtype'] = get_df_sql_dtype(table.columns[name])

                db_col_type = get_col_sql_dtype(table.columns[name])
                if not db_col_type == col_info['dtype']:
                    if col_info['dtype'] == String and not col_info['pk']:
                        # may be a column with NaNs; ignore
                        continue
                    if col_info['dtype'] == Integer and db_col_type == Float:
                        continue
                    if col_info['dtype'] == Float and db_col_type == Integer:
                        continue
                    raise ValueError(f'Inconsistent type for col: {name}! db: {get_col_sql_dtype(table.columns[name])}/'
                                     f'df: {col_info["dtype"]}')
            elif col_info['dtype'] is not None:
                new_cols.append(make_column(name, col_info))
            else:
                logging.warning(f'tried to add all NaN column {name}')
                continue

        if len(new_cols):
            col_names_string = ", ".join([col.name for col in new_cols])

            new_column_warning = f' Table[{table_name}]:' + \
                                 f' new Series [{col_names_string}] added around index {df.index[0]}'
            if strict:
                raise ValueError(new_column_warning)
            else:
                logging.warning(new_column_warning)

            for new_col in new_cols:
                with engine.begin() as conn:
                    conn.execute(f'ALTER TABLE {table_name} '
                                 f'ADD COLUMN {new_col.name} {new_col.type.compile(engine.dialect)}')

    # 2b. unless it's a brand-new table
    else:
        logging.info(f'Creating new table {table_name}')
        table = Table(table_name, meta,
                      *[make_column(name, info) for name, info in df_cols_dict.items()
                        if info['dtype'] is not None])

    #####################################
    # 3. create any new tables or columns

    with engine.begin() as con:
        meta.create_all(bind=con)

    ######################################################
    # FINALLY: either insert/fail, append/fail, or upsert

    if how in ['append', 'fail']:
        # raise if repeated index
        with engine.begin() as con:
            rows = []
            for index, row in df.iterrows():
                rows.append(row.to_dict())
                rows[-1][df.index.name] = index
            con.execute(table.insert(), rows)

    elif how == 'upsert':
        for i in df.index:
            # check index uniqueness by attempting insert; if it fails, update
            try:
                with engine.begin() as con:
                    row = df.loc[i]
                    values = row[row.notna()].to_dict()
                    values[df.index.name] = i
                    insert = table.insert().values(values)
                    con.execute(insert)

            except IntegrityError:
                print('Upsert: Integrity Error on insert => do update')

            with engine.begin() as con:
                row = df.loc[i]
                upsert = table.update() \
                    .where(table.c[df.index.name] == i) \
                    .values(row[row.notna()].to_dict())
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
    dtypes = {}
    for col in table.columns:
        dtypes[col.name] = col.type.python_type
        if col.primary_key:
            index_col = col.name

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

    datetime_cols = list([key for key, value in dtypes.items() if value == pd.datetime])

    df = pd.read_sql_query(s, engine, index_col=index_col,
                           parse_dates={col: {'utc': True} for col in datetime_cols})

    return df
