import pandas as pd
import numpy as np
from pandas.api.types import (is_bool_dtype,
                              is_datetime64_any_dtype,
                              is_integer_dtype,
                              is_float_dtype,
                              is_object_dtype,
                              )

import sqlalchemy as sqa
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, TIMESTAMP

# fake random index name in case of not explicitly indexed data
PANDABASE_DEFAULT_INDEX = 'pandabase_auto_generated_index'


def _sqa_type2pandas_type(sqa_dtype, index=False):
    """explicitly map SQLAlchemy datatypes to our Pandas dtypes. param index is to prevent nullable index"""
    if sqa_dtype == Integer:
        if index:
            return int
        else:
            return pd.Int64Dtype()
    elif sqa_dtype == Float:
        return np.float64
    elif sqa_dtype == Boolean:
        return np.bool_
    elif sqa_dtype == String:
        return np.str_
    elif sqa_dtype in [DateTime, TIMESTAMP] or isinstance(sqa_dtype, TIMESTAMP):
        return np.datetime64
    else:
        raise TypeError(f'Unknown sqlalchemy dtype: {sqa_dtype}')


def series_is_boolean(col: pd.Series or pd.Index):
    """
    returns:
        None if column is all None;
        True if a pd.Series only contains True, False, and None;
        False otherwise

    caveat: does not interpret all-zero or all-one columns as boolean"""
    if len(col.unique()) == 1 and col.unique()[0] is None:
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
        if not (False in col.unique() and True in col.unique()):
            return False
        return True
    elif is_integer_dtype(col) or is_float_dtype(col):
        for val in col.unique():
            if pd.isna(val):
                continue
            if val not in [1, 0, None]:
                return False
            if not (0 in col.unique() and 1 in col.unique()):
                return False
        return True
    return False


def engine_builder(con):
    """
    Returns a SQLAlchemy engine from a URI (if con is a string)
    if con is already a connection, return con without modifying it.
    """
    if isinstance(con, str):
        if con[:8].lower() == 'postgres':
            con = sqa.create_engine(con, connect_args={'connect_timeout': 10})
        else:
            con = sqa.create_engine(con)

    return con


def _get_type_from_df_col(col: pd.Series, index: bool):
    """
    Take a pd.Series, return its SQLAlchemy datatype
    If it doesn't match anything, return String
    Args:
        col: pd.Series to check
        index: if True, index cannot be boolean
    Returns:
        sqlalchemy Type or None
        one of {Integer, Float, Boolean, DateTime, String, or None (for all NaN)}
    """
    if col.isna().all():
        return None

    if is_bool_dtype(col):
        if index:
            raise ValueError('boolean index does not make sense')
        return Boolean
    elif not index and series_is_boolean(col):
        return Boolean
    elif is_integer_dtype(col):
        # parse purported 'integer' columns in a new table.
        # if values are all zero, make it a float for added safety - common case of a float that is often zero
        # if database table is type INTEGER, this will be parsed back to int later anyway
        if index:
            return Integer
        for val in col.unique():
            if pd.isna(val):
                continue
            if val != 0:
                return Integer
        return Float
    elif is_float_dtype(col):
        return Float
    elif is_datetime64_any_dtype(col):
        return TIMESTAMP(timezone=True)
    else:
        return String


def _get_type_from_db_col(col: sqa.Column):
    """Return Sqlalchemy type of a SQLAlchemy column. Updated to limit sqlalchemy types to this explicit list"""
    if isinstance(col.type, sqa.types.Integer):
        return Integer
    elif isinstance(col.type, sqa.types.Float):
        return Float
    elif isinstance(col.type, sqa.types.DateTime):
        return TIMESTAMP(timezone=True)
    elif isinstance(col.type, sqa.types.TIMESTAMP):
        return TIMESTAMP(timezone=True)
    elif isinstance(col.type, sqa.types.Boolean):
        return Boolean
    else:
        return String


def get_column_dtype(column, pd_or_sqla, index=False):
    """
    Take a column (sqlalchemy table.Column or df.Series), return its dtype in Pandas or SQLA

    If it doesn't match anything else, return String

    Args:
        column: pd.Series or SQLA.table.column
        pd_or_sqla: either 'pd' or 'sqla': which kind of type to return
        index: if True, column type cannot be boolean
    Returns:
        Type or None
            if pd_or_sqla == 'sqla':
                one of {Integer, Float, Boolean, DateTime, String, or None (for all-NaN column)}
            if pd_or_sqla == 'pd':
                one of {np.int64, np.float64, np.datetime64, np.bool_, np.str_}
    """
    if isinstance(column, sqa.Column):
        dtype = _get_type_from_db_col(column)
    elif isinstance(column, (pd.Series, pd.Index)):
        dtype = _get_type_from_df_col(column, index=index)
    else:
        raise ValueError(f'get_column_datatype takes a column; got {type(column)}')

    if dtype is None:
        return None

    elif pd_or_sqla == 'sqla':
        return dtype
    elif pd_or_sqla == 'pd':
        return _sqa_type2pandas_type(dtype, index=index)
    else:
        raise ValueError(f'Select pd_or_sqla must equal either "pd" or "sqla"')


def has_table(con, table_name, schema=None):
    """returns True if a table exactly named table_name exists in con"""
    engine = engine_builder(con)

    if schema is not None:
        return engine.run_callable(engine.dialect.has_table, table_name, schema=schema)
    else:
        return engine.run_callable(engine.dialect.has_table, table_name)


def clean_name(name):
    """returns a standardized version of name: lower case without spaces or special characters. May not be numeric"""
    d = {char: '_' for char in ' ()+-/*";=&|#><^%{}'}
    d['.'] = None
    d[','] = None
    table = str.maketrans(d)
    if name[0].isnumeric():
        raise NameError(f'Pandabase does not allow purely numeric names or names that start with digits.'
                        f' Illegal name: {name}')
    if '@' in name:
        raise NameError(f'At sign @ may not be a legal identifier. Please rename: {name}')
    return str(name).lower().strip().translate(table)


def make_clean_columns_dict(df: pd.DataFrame, autoindex=False):
    """Takes a DataFrame, returns a dictionary {column_names: {column info dicts}}

    if autoindex is True:
        if multi-index:
            fail; must reset_index first
        include an additional new Integer column named PANDABASE_DEFAULT_INDEX and discard df.index
    else:
        df.index.name (or .names for MultiIndex) must all be legal names (i.e. results of clean_name)

    Example:
        >>> import pandas as pd
        >>> data = {'full_name':['John Doe'],
        ...         'number_of_pets':[3],
        ...         'likes_bananas':[True], 
        ...         'dob':[pd.Timestamp('1990-01-01')]}
        >>> 
        >>> df = pd.DataFrame(data).rename_axis('id', axis = 'index')
        >>> make_clean_columns_dict(df)
        {'id': {'dtype': sqlalchemy.sql.sqltypes.Integer, 'pk': True},
         'full_name': {'dtype': sqlalchemy.sql.sqltypes.String, 'pk': False},
         'number_of_pets': {'dtype': sqlalchemy.sql.sqltypes.Integer, 'pk': False},
         'likes_bananas': {'dtype': sqlalchemy.sql.sqltypes.Boolean, 'pk': False},
         'dob': {'dtype': TIMESTAMP(timezone=True), 'pk': False}}
    
    """
    if len(df.columns) > 253:
        raise ValueError('pandabase is (currently) incompatible with data over 253 columns wide')
    columns = {}
    df.columns = [clean_name(col) for col in df.columns]

    # get index info
    if autoindex:
        if isinstance(df.index, pd.MultiIndex):
            raise ValueError(f'Must reset_index to use autoindex=True')

        index_name = PANDABASE_DEFAULT_INDEX
        columns[index_name] = {'dtype': Integer,
                               'pk': True}

    elif isinstance(df.index, pd.MultiIndex):
        indices = df[[]].reset_index(drop=False)
        for col_name in indices.columns:
            if col_name in df.columns:
                raise NameError(f'MultiIndex name is duplicate of column name: {col_name}')
            columns[col_name] = {'dtype': get_column_dtype(indices[col_name], 'sqla', index=True),
                                 'pk': True}
    else:
        if df.index.name in df.columns:
            raise NameError(f'index name is duplicate of column name: {df.index.name}')
        columns[df.index.name] = {'dtype': get_column_dtype(df.index, 'sqla', index=True),
                                  'pk': True}

    # get column info
    for col_name in df.columns:
        dtype = get_column_dtype(df[col_name], 'sqla')
        columns[col_name] = {
            'dtype': dtype,
            'pk': False,
        }
    assert len(columns) > 0

    return columns


def make_column(name, info):
    """Make a sqla.Column from column information; nullable unless it's the primary key"""
    nullable = not info['pk']
    return Column(name, primary_key=info['pk'], type_=info['dtype'], nullable=nullable)
