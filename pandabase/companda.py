"""companda compares pandas DataFrames"""

import pandas as pd
from pandas.api.types import (is_integer_dtype,
                              is_float_dtype)
from .helpers import get_column_dtype, PANDABASE_DEFAULT_INDEX


class CompandaNotEqualError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Companda(object):
    """output from companda function; evaluates as boolean + holds a message"""
    def __init__(self, equal: bool, msg=''):
        self.equal = equal
        self.message = msg

    def __bool__(self):
        if not self.equal:
            raise CompandaNotEqualError(str(self))
        else:
            return True

    def __repr__(self):
        return str(self.equal) + ": " + self.message

    @property
    def msg(self):
        return str(self)


def companda(df1: pd.DataFrame, df2: pd.DataFrame, gamma=.0001, ignore_nan=False):
    """compare two DataFrames; return a Companda object

    Args:
        df1:
        df2:
        gamma:
        ignore_nan: ignore any all_nan columns

    Returns: truthy Companda iff:
        1. columns are equal (both subsets of each other)
        2. indices are equal
        3. data is equal within decimal error gamma
    """
    if ignore_nan:
        df1 = df1.copy()
        df2 = df2.copy()
        for df in [df1, df2]:
            for col in df.columns:
                if get_column_dtype(df[col], 'pd') is None:
                    df.drop([col], axis=1, inplace=True)
    # COLUMNS
    if len(df1.columns) != len(df2.columns):
        return Companda(False, f'len(df1.cols) = {len(df1.columns)}, len(df2.cols) = {len(df2.columns)}')

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    for col in cols1:
        if col not in cols2:
            msg = f'{col} from df1 not in df2'
            return Companda(False, msg)
    for col in cols2:
        if col not in cols1:
            msg = f'{col} from df2 not in df1'
            return Companda(False, msg)

    # INDEX
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    if len(df1) != len(df2):
        return Companda(False, f'len(df1) = {len(df1)}, len(df2) = {len(df2)}')

    if df1.index.name == PANDABASE_DEFAULT_INDEX or df2.index.name == PANDABASE_DEFAULT_INDEX:
        pass
    else:
        if df1.index.name != df2.index.name:
            return Companda(False,
                            f'Different index names: {df1.index.name}, {df2.index.name}')

        index_unequal = pd.Series(df1.index != df2.index)
        if index_unequal.sum():

            return Companda(False,
                            f'Equal length indices, but {index_unequal.sum()}/{len(index_unequal)} '
                            f'index values are different.')

    # VALUES
    for col in df1.columns:
        if not isinstance(df1[col], type(df2[col])):
            return Companda(False, f"columns and indices equal, but datatypes not equal in column {col}.")

        if is_float_dtype(df1[col]) or is_integer_dtype(df1[col]):
            diff = pd.Series(pd.np.subtract(df1[col].values, df2[col].values) > pd.np.multiply(gamma, df1[col]))
            if diff.sum() == 0:
                continue
            else:
                return Companda(False, f"columns and indices equal; values not almost equal in column {col}.")
        else:
            if pd.np.array_equal(df1[col], df2[col]):
                continue
            else:
                return Companda(False, f"columns and indices equal; values not equal in column {col}.")

    return Companda(True)
