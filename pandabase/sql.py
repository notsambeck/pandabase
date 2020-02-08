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
from pandabase.helpers import *

import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype

import sqlalchemy as sqa
from sqlalchemy import Table, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert

import pytz
from logging import getLogger


logger = getLogger()


def to_sql(df: pd.DataFrame, *,
           table_name: str,
           con: str or sqa.engine,
           auto_index=False,
           how='create_only',
           add_new_columns=False, 
           schema: str = None):
    """
    Write records stored in DataFrame [df] to [table_name] in SQL database [con].

    Caveats:
    Converts any datetime to UTC
    Requires a unique, named index as DataFrame.index; to insert into existing database, this name must be consistent

    Args:
        df : DataFrame
        ### keyword only args follow ###
        table_name : string; Name of SQL table.
        con : connection; database string URI < OR > sqlalchemy.engine
        auto_index: bool, default False. if True, ignore existing df.index, make a new integer index
        add_new_columns: bool, default False. if True, add any new columns as required by the DataFrame.
        how : {'create_only', 'upsert', 'append'}, default 'create_only'
            - create_only:
                If table exists, raise an error and stop.
            - append:
                If table exists, append data. Raise if index overlaps
                Create table if does not exist.
            - upsert:
                create table if needed
                if record exists: update (possibly replacing values with NULL)
                else: insert
        schema: Specify the schema (if database flavor supports this, i.e. postgresql). If None, use default schema.
    
    """
    # 1. make connection objects
    df = df.copy()

    engine = engine_builder(con)
    meta = sqa.MetaData()

    ######################
    #
    #     2. validate inputs
    #
    ######################
    clean_table_name = clean_name(table_name)
    if clean_table_name != table_name:
        raise NameError(f'Illegal characters in table name: {table_name}. try: {clean_table_name}')

    if how not in ('create_only', 'append', 'upsert',):
        raise ValueError(f"Parameter how={how}; how must be in ['create_only', 'upsert', 'append']")

    if not isinstance(df, pd.DataFrame):
        raise ValueError('pandabase.to_sql() requires a DataFrame as input')

    if not auto_index:
        if not df.index.is_unique:
            raise ValueError('DataFrame.index is not unique and cannot be used as PK.')
        if is_datetime64_any_dtype(df.index):
            if df.index.tz != pytz.utc:
                raise ValueError(f'Index {df.index.name} is not UTC. Please correct.')

        if isinstance(df.index, pd.MultiIndex):
            alterations = {}
            for i, name in enumerate(df.index.names):
                if name is None:
                    raise NameError(f'One or more values in MultiIndex is unnamed: {df.index.names}')
                if clean_name(name) != name:
                    alterations[i] = clean_name(name)
            if alterations:
                new_names = list(df.index.names)
                for k, v in alterations.items():
                    new_names[k] = v
                df.index.names = new_names

        else:
            if df.index.name is None:
                raise NameError('Autoindex is False, but df.index.name is None. Set df.index.name = something')
            if df.index.hasnans:
                raise ValueError('DataFrame.index has NaN values and cannot be used as PK.')
            df.index.name = clean_name(df.index.name)

    else:
        if isinstance(df.index, pd.MultiIndex):
            raise ValueError(f'pandabase does not allow autoindex=True on a DataFrame with MultiIndex. '
                             f'Consider using df.reset_index(drop=[True or False]).')
        # otherwise: auto_index; drop index info
        df.reset_index(drop=True)
        df.index.name = PANDABASE_DEFAULT_INDEX

    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz is None:
                raise ValueError(f'Column {col} timezone must be set, maybe with '
                                 f'col.tz_localize(pytz.timezone).tz_convert(pytz.utc)')
            elif df[col].dt.tz != pytz.utc:
                raise ValueError(f'Column {col} is set, but not to UTC. Maybe correct with col.tz_convert(pytz.utc)')

    # make a list of df columns for later:
    df_cols_dict = make_clean_columns_dict(df, autoindex=auto_index)

    #####################################################
    #
    #    3a. Make new Table from df info, if it does not exist...
    #
    #####################################################
    if not has_table(engine, table_name, schema=schema):
        
        # log the creation of the table
        if schema is not None:
            log_info = f'Creating new table {schema}.{table_name}'
        else:
            log_info = f'Creating new table {table_name}'
        
        logger.info(log_info)

        # create the table
        table = Table(table_name, 
                      meta,
                      *[make_column(name, info) for name, info in df_cols_dict.items()
                        if info['dtype'] is not None],
                      schema=schema)    # schema defaults to None also in the Table class

    #######################################################################################
    #
    #    3b. Or make Table from db schema
    #    db will be the source of truth for datatypes etc. in the future
    #
    #######################################################################################
    else:
        if how == 'create_only':
            raise NameError(f'Table {table_name} already exists; param "how" is set to "create_only".')

        table = Table(table_name, meta, autoload=True, autoload_with=engine, schema=schema)

        if how == 'upsert':
            if table.primary_key == PANDABASE_DEFAULT_INDEX or auto_index:
                raise IOError('Cannot upsert with an automatically generated index')

        # 3. iterate over df_columns; confirm that types are compatible and all columns exist
        for col_name, df_col_info in df_cols_dict.items():
            if col_name not in table.columns:
                if df_col_info['dtype'] is None:
                    continue   # skip empty columns that do not exist in db
                elif add_new_columns:
                    logger.info(f'adding new column to {con}:{table_name}: {col_name}')
                    _add_columns_to_db(make_column(col_name, df_col_info),
                                       table_name=table_name, con=con, schema=schema)
                    meta.clear()
                    table = Table(table_name, 
                                  meta, 
                                  autoload=True, 
                                  autoload_with=engine,
                                  schema=schema)  # schema defaults to None also in the Table class

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

            #############################################
            #
            #    3d. COERCE INCONSISTENT DATATYPES - case by case
            #    try to change DataFrame dtypes to match existing db dtypes
            #
            #############################################
            db_pandas_dtype = get_column_dtype(col, pd_or_sqla='pd')

            if df_col_info['dtype'] is None:  # dtype is None only if the whole column is NULL
                # this does not need to be explicitly handled because when inserting None, nothing happens
                continue

            elif is_datetime64_any_dtype(db_pandas_dtype):
                pass
                # TODO: df[col_name] = pd.to_datetime(df[col_name].values, utc=True)
                # TODO: would be nice, but THIS FAILS FOR INDEX/PK. would solve cases when:
                # db.col is datetime and df.col is string
                # df.datetime_column.tz != utc

            elif (
                    df_col_info['dtype'] == Integer and is_float_dtype(db_pandas_dtype) or
                    df_col_info['dtype'] == Float and is_integer_dtype(db_pandas_dtype)
            ):
                # print(f'NUMERIC DTYPE: converting df[{name}] from {df[name].dtype} to {db_pandas_dtype}')
                try:
                    df[col_name] = df[col_name].astype(db_pandas_dtype)
                except Exception as e:
                    raise TypeError(f'Error {e} while trying to coerce float to int or vice versa, '
                                    f'e.g. cannot coerce {df[col_name][0]} to {db_pandas_dtype}')
            elif (
                    df_col_info['dtype'] == Boolean and is_integer_dtype(db_pandas_dtype) or
                    df_col_info['dtype'] == Boolean and is_float_dtype(db_pandas_dtype)
            ):
                try:
                    df[col_name] = df[col_name].astype(db_pandas_dtype)
                except Exception as e:
                    raise TypeError(f'Error {e} while trying to coerce bool float or int, '
                                    f'e.g. cannot coerce {df[col_name][0]} to {db_pandas_dtype}')

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
        _insert(table, engine, df, auto_index)

    elif how == 'upsert':
        _upsert(table, engine, df)

    return table


def _insert(table: sqa.Table,
            engine: sqa.engine,
            cleaned_data: pd.DataFrame,
            auto_index: bool):

    with engine.begin() as con:
        rows = []
        
        # remove completely null columns; convert to object due to bug inserting Int64
        df = cleaned_data.dropna(axis=1, how='all').astype('object')

        if not auto_index:
            for row in df.reset_index(drop=False).itertuples(index=False):
                rows.append(row._asdict())
            con.execute(table.insert(), rows)
        else:
            for row in df.reset_index(drop=True).itertuples(index=False):
                rows.append(row._asdict())
            con.execute(table.insert(), rows)


def _upsert(table: sqa.Table,
            engine: sqa.engine,
            cleaned_data: pd.DataFrame):
    """
    insert data into a table, replacing any rows with duplicate indices

    When upsert finds duplicate records, it overwrites ALL VALUES that are present in source DataFrame, including NaN.

    postgres - see: https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#insert-on-conflict-upsert
    """
    if isinstance(cleaned_data.index, pd.MultiIndex):
        multi = True
        names = cleaned_data.index.names
        index_elements = names
    else:
        multi = False
        names = cleaned_data.index.name
        index_elements = [names]

    cleaned_data = cleaned_data.astype('object')

    def map2none(val):
        """sqlalchemy freaks out about NaNs, so replace them with None"""
        if pd.notna(val):
            return val

    with engine.begin() as con:
        for row in cleaned_data.reset_index(drop=False).itertuples(index=False):
            # check index uniqueness by attempting insert; if it fails, update
            row = {k: map2none(v) for k, v in row._asdict().items()}
            try:

                if engine.dialect.dbapi.__name__ == 'psycopg2':
                    insert = pg_insert(table).values(row).on_conflict_do_update(
                        index_elements=index_elements,
                        set_=row
                    )
                else:
                    insert = table.insert().values(row)

                con.execute(insert)

            except sqa.exc.IntegrityError:
                if multi:
                    upsert = table.update()
                    for n in names:
                        upsert = upsert.where(table.c[n] == row[n])
                    con.execute(upsert.values(row))
                else:
                    upsert = table.update() \
                        .where(table.c[names] == row[names]) \
                        .values(row)
                    con.execute(upsert)


def read_sql(table_name: str,
             con: str or sqa.engine,
             *,
             lowest=None, highest=None, schema: str = None):
    """
    Read in a table from con as a pd.DataFrame, preserving dtypes and primary keys

    Args:
        table_name:
        con:
        lowest: minimum value of PK to select (inclusive)
        highest: maximum value of PK to select (inclusive)
            for MultiIndex tables, highest and lowest must be tuples of all primary key values (left to right) or None
        schema: Specify the schema (if database flavor supports this). If None, use default schema.

    Returns:
        DataFrame: selected data with table.primary_key(s) as index
    """
    engine = engine_builder(con)
    meta = sqa.MetaData(bind=engine)
    table = Table(table_name, 
                  meta, 
                  autoload=True, 
                  autoload_with=engine, 
                  schema=schema)   # schema defaults to None also in the Table class

    if len(table.primary_key.columns) == 0:
        print(f'Table {table_name} has no explicit PK/index (using autoindex)')
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
                raise TypeError(f'Select range is: {lowest} <= data <= {highest}; but type of column is {pk.type}. '
                                f'If e.g. cutoffs are integers, and column is float, please coerce type.')

    elif len(table.primary_key.columns) > 1:
        pks = table.primary_key.columns.items()

        s = table.select()

        for selector, sign in [(highest, 'highest'), (lowest, 'lowest')]:
            if selector is None:
                continue

            if len(selector) != len(pks):
                raise ValueError('pandabase.read_sql(multi-indexed_table) requires any values of highest, lowest '
                                 'that are not None to have __len__ equal to len(index). Use e.g. (value, None) to '
                                 'filter only on the first index dimension.')

            for i, val in enumerate(selector):
                if val is None:
                    continue
                elif sign == 'lowest':
                    s = s.where(pks[i][1] >= val)
                else:
                    s = s.where(pks[i][1] <= val)

        result = engine.execute(s)
        data = result.fetchall()

        if len(data) == 0:
            for i, (pk, col) in enumerate(table.primary_key.columns.items()):
                if isinstance(lowest[i], col.type.python_type) or isinstance(highest[i], col.type.python_type):
                    continue
                else:
                    raise TypeError(f'Select range is: {lowest[i]} <= data <= {highest[i]}; '
                                    f'but type of column is {col.type.python_type}')

    else:
        raise ValueError(f'invalid table.primary_key.columns = {table.primary_key.columns}')

    df = pd.DataFrame.from_records(data, columns=[col.name for col in table.columns],
                                   coerce_float=True)

    indices = []  # in case of multi-index, accumulate columns and assemble later
    for col in table.columns:
        # deal with primary keys separately; never convert primary key to nullable
        if col.primary_key:
            dtype = get_column_dtype(col, pd_or_sqla='pd', index=True)

            # single index
            if len(table.primary_key.columns) == 1:
                df.index = df[col.name]

                if is_datetime64_any_dtype(dtype):
                    df.index = pd.to_datetime(df[col.name].values, utc=True)

                # auto-handle legacy index name with random number suffix
                if col.name == PANDABASE_DEFAULT_INDEX or col.name[:23] == 'pandabase_default_index':
                    df.index.name = None
                else:
                    df.index.name = col.name

                df = df.drop(columns=[col.name])
                continue

            # multi-index
            else:
                indices.append(df[col.name].copy())

                if is_datetime64_any_dtype(dtype):
                    indices[-1] = pd.to_datetime(indices[-1].values, utc=True)

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

    # generate multi_index
    if indices:
        df.index = pd.MultiIndex.from_arrays(indices)

    return df


def _add_columns_to_db(new_col, table_name, con, schema=None):
    """Make new columns as needed with ALTER TABLE, as a weak substitute for migrations"""
    engine = engine_builder(con)
    name = clean_name(new_col.name)
    
    if schema is None:
        table_namespace = table_name
    else:
        table_namespace = f'{schema}.{table_name}'

    with engine.begin() as conn:
        conn.execute(f'ALTER TABLE {table_namespace} '
                     f'ADD COLUMN {name} {new_col.type.compile(engine.dialect)}')


def profiling_script(n_rows):
    db = 'sqlite:///:memory:'
    n_cols = 100
    _df = pd.DataFrame(index=range(n_rows), columns=['col_' + str(n) for n in range(n_cols)],
                       data=np.random.random((n_rows, n_cols)))
    _df.index.name = 'dex'
    to_sql(_df, con=db, table_name='table', how='append')


if __name__ == '__main__':
    profiling_script(10000)
