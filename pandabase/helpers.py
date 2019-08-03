import pandas as pd
import numpy as np
from pandas.api.types import (is_bool_dtype,
                              is_datetime64_any_dtype,
                              is_integer_dtype,
                              is_float_dtype,
                              is_object_dtype,
                              )

import sqlalchemy as sqa
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean

PANDABASE_DEFAULT_INDEX = 'pandabase_index'


def series_is_boolean(col):
    """returns:
    None if column is all None;
    True if a pd.Series only contains True, False, and None;
    otherwise False"""
    if not isinstance(col, pd.Series):
        raise TypeError(f'series_is_boolean takes a pd.Series; got {col}::{type(col)} instead')
    elif len(col.unique()) == 1 and col.unique()[0] is None:
        # return None for all-None columns
        return None
    elif col.isna().all():
        return None
    elif is_bool_dtype(col):
        return True
    elif is_object_dtype(col):
        for val in col.unique():
            if val not in [True, False, None]:
                return False
        return True
    elif is_integer_dtype(col) or is_float_dtype(col):
        for val in col.unique():
            if pd.isna(val):
                continue
            if val not in [1, 0]:
                return False
        return True
    return False


def engine_builder(con):
    """
    Returns a SQLAlchemy engine from a URI (if con is a string)
    else it just return con without modifying it.
    """
    if isinstance(con, str):
        con = sqa.create_engine(con)

    return con


def _get_type_from_df_col(col: pd.Series):
    """
    Take a pd.Series, return its SQLAlchemy datatype
    If it doesn't match anything, return String
    Args:
        pd.Series
    Returns:
        sqlalchemy Type or None
        one of {Integer, Float, Boolean, DateTime, String, or None (for all NaN)}
    """
    if col.isna().all():
        return None

    if is_bool_dtype(col):  # or series_is_boolean:
        return Boolean
    elif is_integer_dtype(col):
        return Integer
    elif is_float_dtype(col):
        return Float
    elif is_datetime64_any_dtype(col):
        return DateTime
    else:
        return String


def _get_type_from_db_col(col: sqa.Column):
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


def get_column_dtype(column, pd_or_sqla):
    """
    Take a column (sqlalchemy table.Column or df.Series), return its dtype in Pandas or SQLA

    If it doesn't match anything else, return String

    Args:
        column: pd.Series or SQLA.table.column
        pd_or_sqla: either 'pd' or 'sqla': which kind of type to return
    Returns:
        Type or None
            if pd_or_sqla == 'sqla':
                one of {Integer, Float, Boolean, DateTime, String, or None (for all NaN)}
            if pd_or_sqla == 'pd':
                one of {np.int64, np.float64, np.datetime64, np.bool_, np.str_}
    """
    if isinstance(column, sqa.Column):
        t = _get_type_from_db_col(column)
    elif isinstance(column, (pd.Series, pd.Index)):
        t = _get_type_from_df_col(column)
    else:
        raise ValueError(f'Expected some kind of a column, got {type(column)}')

    if t is None:
        return None
    elif pd_or_sqla == 'sqla':
        return t
    elif pd_or_sqla == 'pd':
        lookup = {Integer: pd.Int64Dtype(),
                  Float: np.float64,
                  DateTime: np.datetime64,
                  Boolean: np.bool_,
                  String: np.str_}
        return lookup[t]
    else:
        raise ValueError(f'Select pd_or_sqla must equal either "pd" or "sqla"')


def has_table(con, table_name):
    """pandas.sql.has_table()"""
    engine = engine_builder(con)
    return engine.run_callable(engine.dialect.has_table, table_name)


def clean_name(name):
    """returns a standardized version of column names: lower case without spaces"""
    return name.lower().strip().replace(' ', '_')


def make_clean_columns_dict(df: pd.DataFrame):
    """Take a DataFrame and use_index, return a dictionary {name: {Column info}} (including index or not)"""
    columns = {}
    df.columns = [clean_name(col) for col in df.columns]
    index_name = clean_name(df.index.name)

    for col_name in df.columns:

        dtype = get_column_dtype(df[col_name], 'sqla')
        columns[col_name] = {
            'dtype': dtype,
            'pk': False,
        }

    assert len(columns) > 0
    columns[index_name] = {'dtype': get_column_dtype(df.index, 'sqla'),
                           'pk': True}

    return columns


def make_column(name, info):
    """Make a Column from column information"""
    nullable = not info['pk']
    return Column(name, primary_key=info['pk'], type_=info['dtype'], nullable=nullable)
