"""
pandabase is a tool that replaces pandas.to_sql and pandas.read_sql, allowing for easy

specifically:
* provides an easy way to move data from DataFrames to a sql database and vice versa
(intended for data science applications where the dataset may expand or be updated over time)

* definitely supports sqlite, although it may work for other dialects as well.

* uses sqlalchemy core to explicitly define primary keys, allowing for:
1. tables with meaningful indices
2. upserts

by sam beck, github.com/notsambeck
largely stolen from pandas and dataset (todo: add links)
"""

import pandas as pd
from pandas.api.types import (is_bool_dtype, is_datetime64_any_dtype,
                              is_integer_dtype, is_float_dtype)
import sqlalchemy as sa
from sqlalchemy import Table, Column, Integer, String, Float, DateTime, Boolean

import logging
logging.basicConfig(level=logging.DEBUG)


def engine_builder(con):
    """
    Returns a SQLAlchemy engine from a URI (if con is a string)
    else it just return con without modifying it.
    """
    if isinstance(con, str):
        con = sa.create_engine(con)

    return con


def _get_sql_dtype(series):
    """
    Take a pd.Series (or column of DataFrame), return SQLAlchemy datatype for column

    If it's not anything else, return a string

    :param series:
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
    return engine.run_callable(
        engine.dialect.has_table,
        table_name,
    )


def to_sql(df: pd.DataFrame,
           index_col_name: str,
           table_name: str,
           con,
           how='fail'):
    """
    Write records stored in a DataFrame to a SQL database.

    Parameters
    ----------
    df : DataFrame, Series
    table_name : string
        Name of SQL table.
    con : connection; database string URI < OR > sa.engine
    how : {'fail', 'replace', 'append'}, default 'fail'
        - fail: If table exists, do nothing.
        - append: If table exists, insert data. Create if does not exist.
        - upsert: Table must exist; if record exists ... TODO
    index_col_name : name of column to use as index, or None to use range_index
        (Applies to both DataFrame and sql database)
    """
    ############################################
    # 0. check inputs; make db connection engine

    if how not in ('fail', 'append', 'upsert', ):
        raise ValueError("'{0}' is not valid for if_exists".format(how))

    if not isinstance(df, pd.DataFrame):
        raise ValueError('to_sql() requires a DataFrame as input')

    index_col_name = clean_name(index_col_name)

    engine = engine_builder(con)
    meta = sa.MetaData()

    ####################################
    # 1. make a dict of {name: Column} from df columns
    df_columns = _make_columns_dict(df, index_col_name)

    ############################################
    # 2. get existing table metadata from db, add any new columns from df to db

    if has_table(engine, table_name):
        if how == 'fail':
            raise NameError(f'Table {table_name} already exists; if_exists set to fail.')

        table = Table(table_name, meta, autoload=True, autoload_with=engine)

        for name in df_columns.keys():
            # check dtypes and PKs match
            if name in table.columns:
                if table.columns[name].primary_key != df_columns[name].primary_key:
                    raise ValueError(f'Inconsistent pk for {name}! db: {table.columns[name].primary_key} / '
                                     f'df: {df_columns[name].primary_key}')
                if table.columns[name].type != df_columns[name].type:
                    raise ValueError(f'Inconsistent type for {name}! db: {table.columns[name].type} / '
                                     f'df: {df_columns[name].type}')
            # add new columns if needed (sqla does not support migrations directly)
            else:
                # TODO: confirm sqla automatically adds NaN here
                logging.warning(f'Table {table_name} / column {name}:'
                                f' new Series added at index {df[index_col_name]}')
                new_col = Column(name, _get_sql_dtype(df[name]))
                with engine.begin as conn:
                    conn.execute(f'ALTER TABLE {table_name} '
                                 f'ADD COLUMN {name} {new_col.type.compile(engine.dialect)}')

    # unless it's a new table
    else:
        logging.info(f'Creating table {table_name}')
        table = Table(table_name, meta, *df_columns.values())

    ###########################################################
    # 3. create tables
    with engine.begin() as conn:
        meta.create_all(bind=conn)

    ######################################################
    # FINALLY: either insert or upsert
    df.index = df[index_col_name]
    assert df.index.is_unique

    with engine.begin() as conn:
        logging.info('starting write')
        if how in ['append', 'fail']:
            # raise if repeated index
            conn.execute(table.insert(), [row.to_dict() for i, row in df.iterrows()])
            return
        if how == 'upsert':
            # explicitly handle index uniqueness
            for i in df.index:
                try:
                    upsert = table.insert().values(df.loc[i].to_dict())
                    conn.execute(upsert)
                except sa.exc.UniqueConstraintError:   # TODO fake name?
                    upsert = table.update()\
                            .where(table.c.index_col_name == i)\
                            .values(df.loc[i].to_dict())
                    conn.execute(upsert)


def clean_name(name):
    return name.lower().strip().replace(' ', '_')


def _make_columns_dict(df: pd.DataFrame, index_col_name):
    """Take a DataFrame and index_col_name (or None), return a dictionary {table_name: new Table}"""
    columns = {}
    df.columns = [clean_name(col) for col in df.columns]

    if index_col_name is not None:
        assert index_col_name in df.columns
    else:
        columns['id'] = Column('id', Integer, primary_key=True)

    for col_name in df.columns:
        pk = index_col_name == col_name
        columns[col_name] = Column(col_name, _get_sql_dtype(sample_data[col_name]), primary_key=pk, )

    assert len(columns) > 1
    return columns


if __name__ == '__main__':
    db = 'sqlite:///test_data/test.db'
    print('users table exists:', has_table(db, 'users'))

    # load sample data and parse date columns
    for file in ['./test_data/sample_data.csv.zip', './test_data/sample_data2.csv.zip']:
        sample_data = pd.read_csv(file)
        sample_data.columns = [clean_name(col) for col in sample_data.columns]
        sample_data = sample_data[sample_data.lmp_type == 'LMP']
        for df_col in sample_data.columns:
            if 'time' in df_col.lower():
                sample_data[df_col] = pd.to_datetime(sample_data[df_col])

        to_sql(sample_data, 'INTERVALSTARTTIME_GMT', 'users', db, 'append')

        print(sample_data.columns)
        print(sample_data['lmp_type'].unique())
