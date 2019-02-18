import pandas as pd
import numpy as np
from pandas.api.types import (is_bool_dtype,
                              is_datetime64_any_dtype,
                              is_integer_dtype,
                              is_float_dtype,
                              is_string_dtype)

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


def get_df_sql_dtype(s):
    """
    Take a pd.Series or column of DataFrame, return its SQLAlchemy datatype
    If it doesn't match anything, return String
    Args:
        pd.Series
    Returns:
        sqlalchemy Type or None
        one of {Integer, Float, Boolean, DateTime, String, or None (for all NaN)}
    """
    if s.isna().all():
        return None

    if is_bool_dtype(s):
        return Boolean
    elif is_integer_dtype(s):
        return Integer
    elif is_float_dtype(s):
        return Float
    elif is_datetime64_any_dtype(s):
        return DateTime
    else:
        return String


def get_db_col_dtype(column, pd_or_sqla='pd'):
    """
    Take a sqlalchemy table.Column, return its pandas datatype
    If it doesn't match anything, return String
    Args:
        pd.Series
    Returns:
        sqlalchemy Type or None
        one of {Integer, Float, Boolean, DateTime, String, or None (for all NaN)}
    """
    def _get_sqla_type(col):
        if isinstance(col.type, sqa.types.Integer):
            return Integer
        elif isinstance(col.type, sqa.types.Float):
            return Float
        elif isinstance(col.type, sqa.types.DateTime):
            return DateTime
        elif isinstance(col.type, sqa.types.Boolean):
            return Boolean
        else:
            return String

    t = _get_sqla_type(column)
    if pd_or_sqla == 'sqla':
        return t
    elif pd_or_sqla == 'pd':
        lookup = {Integer: np.int64,
                  Float: np.float64,
                  DateTime: pd.datetime,
                  Boolean: np.bool_,
                  String: np.str_}
        return lookup[t]
    else:
        raise ValueError(f'Select pd_or_sqla: param = "pd" or "sqla"')


def has_table(con, table_name):
    """pandas.sql.has_table()"""
    engine = engine_builder(con)
    return engine.run_callable(engine.dialect.has_table, table_name)


def clean_name(name):
    return name.lower().strip().replace(' ', '_')


def make_clean_columns_dict(df: pd.DataFrame):
    """Take a DataFrame and index_col_name, return a dictionary {name: {Column info}}"""
    columns = {}
    df.columns = [clean_name(col) for col in df.columns]
    index = clean_name(df.index.name)

    for col_name in df.columns:

        dtype = get_df_sql_dtype(df[col_name])
        if dtype is None:
            print('found dtype == None!')

        columns[col_name] = {
            'dtype': dtype,
            'pk': False,
        }

    assert len(columns) > 1

    columns[index] = {'dtype': get_df_sql_dtype(df.index),
                      'pk': True}

    return columns


def make_column(name, info):
    """Make a Column from column information"""
    nullable = not info['pk']
    return Column(name, primary_key=info['pk'], type_=info['dtype'], nullable=nullable)
