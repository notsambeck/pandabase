import pandas as pd
from pandas.api.types import (is_bool_dtype,
                              is_datetime64_any_dtype,
                              is_integer_dtype,
                              is_float_dtype)

import sqlalchemy as sqa
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean


PANDABASE_DEFAULT_INDEX = 'pandabase_index'


def engine_builder(con):
    """
    Returns a SQLAlchemy engine from a URI (if con is a string)
    else it just return con without modifying it.
    """
    if isinstance(con, str):
        con = sqa.create_engine(con)

    return con


def get_df_sql_dtype(series):
    """
    Take a pd.Series or column of DataFrame, return its SQLAlchemy datatype
    If it doesn't match anything, return String
    Args:
        pd.Series
    Returns:
        sqlalchemy Type or None
        one of {Integer, Float, Boolean, DateTime, String, or None (for all NaN)}
    """
    if series.isna().all():
        return None
    elif is_bool_dtype(series):
        return Boolean
    elif is_integer_dtype(series):
        return Integer
    elif is_float_dtype(series):
        return Float
    elif is_datetime64_any_dtype(series):
        return DateTime
    else:
        return String


def get_col_sql_dtype(column):
    """
    Take a sqlalchemy table.Column, return its SQLAlchemy datatype
    If it doesn't match anything, return String
    Args:
        pd.Series
    Returns:
        sqlalchemy Type or None
        one of {Integer, Float, Boolean, DateTime, String, or None (for all NaN)}
    """
    if isinstance(column.type, sqa.types.Integer):
        return Integer
    elif isinstance(column.type, sqa.types.Float):
        return Float
    elif isinstance(column.type, sqa.types.DateTime):
        return DateTime
    elif isinstance(column.type, sqa.types.Boolean):
        return Boolean
    else:
        return String


def has_table(con, table_name):
    """pandas.sql.has_table()"""
    engine = engine_builder(con)
    return engine.run_callable(engine.dialect.has_table, table_name)


def clean_name(name):
    return name.lower().strip().replace(' ', '_')


def make_clean_columns_dict(df: pd.DataFrame, index_col_name):
    """Take a DataFrame and index_col_name, return a dictionary {name: {Column info}}"""
    columns = {}
    df.columns = [clean_name(col) for col in df.columns]

    assert index_col_name in df.columns

    for col_name in df.columns:
        pk = index_col_name == col_name

        dtype = get_df_sql_dtype(df[col_name])
        if dtype is None:
            print('found dtype == None!')

        columns[col_name] = {
            'dtype': dtype,
            'pk': pk,
        }

    assert len(columns) > 1
    return columns


def make_column(name, info):
    """Make a Column from column information"""
    nullable = not info['pk']
    return Column(name, primary_key=info['pk'], type_=info['dtype'], nullable=nullable)
