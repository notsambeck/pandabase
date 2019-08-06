"""companda compares pandas DataFrames"""

import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_datetime64_any_dtype,
)
from .helpers import get_column_dtype


class CompandaNotEqualError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Companda(object):
    """class Companda is the output from companda(df1, df2); evaluates as boolean + stores a message"""
    def __init__(self,
                 equal: bool,
                 columns_equal: bool,
                 message='', ):
        self.equal = equal
        self.columns_equal = columns_equal
        self.message = message

    def __bool__(self):
        return self.equal

    def __repr__(self):
        return f'COMPANDA: equality: {self.equal}; cols: {self.columns_equal}; {self.message}'


def companda(df1: pd.DataFrame,
             df2: pd.DataFrame,
             epsilon=.001,
             check_dtype=False,
             ignore_all_nan_columns=False,
             ignore_index=False,
             ):
    """compare two DataFrames; return a Companda object that is truth-y or false-y.

    Args:
        df1: pd.DataFrame
        df2: pd.DataFrame
        epsilon: float (allowable decimal error)
        check_dtype: bool
        ignore_all_nan_columns: ignore any all NaN (i.e. empty) columns
        ignore_index: ignore values of index

    Returns: Companda(True) if and only if:
        1. columns are equal (i.e. both subsets of each other)
        2. indices are equal
        3. data is present/absent in same locations
        4. data is equal (within decimal error gamma)
    else: returns Companda(False)
    """
    if ignore_all_nan_columns:
        df1 = df1.copy()
        df2 = df2.copy()
        for df in [df1, df2]:
            for col in df.columns:
                if get_column_dtype(df[col], 'pd') is None:
                    df.drop([col], axis=1, inplace=True)
    # COLUMNS
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    missing_from_2 = []
    for col in cols1:
        if col not in cols2:
            missing_from_2.append(col)
    missing_from_1 = []
    for col in cols2:
        if col not in cols1:
            missing_from_1.append(col)
    if missing_from_2 or missing_from_1:
        msg = f'{missing_from_2} missing from df2 and {missing_from_1} missing from df1'
        return Companda(False, False, msg)

    if len(df1.columns) != len(df2.columns):
        return Companda(False, False, f'len(df1.cols) = {len(df1.columns)}, len(df2.cols) = {len(df2.columns)}')

    # INDEX
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    if len(df1) != len(df2):
        return Companda(False, True, f'len(df1) = {len(df1)}, len(df2) = {len(df2)}')

    if not ignore_index:
        if df1.index.name != df2.index.name:
            return Companda(False, True,
                            f'Different index names: {df1.index.name}, {df2.index.name}')

        # coerce index dtype if we're not checking this
        # if not check_dtype:
        #     df1.index = df1.index.astype(type(df2.index))

        index_unequal = pd.Series(df1.index != df2.index)
        if index_unequal.sum():

            return Companda(False, True,
                            f'Equal length indices, but {index_unequal.sum()} out of {len(index_unequal)} '
                            f'index values are different. {df1.index} / {df2.index}')

    # VALUES
    for col in df1.columns:
        # datatype checks
        if check_dtype and not df1[col].dtype is df2[col].dtype:
            return Companda(False, True,
                            f"columns and indices equal, but datatypes not equal in column {col}"
                            f"::{df1[col].dtype}/{df2[col].dtype}.")

        # CHECK FOR DIFFERENT DATATYPES EXPLICITLY
        if is_float_dtype(df1[col]) or is_integer_dtype(df1[col]):
            if pd.np.array_equal(df1[col].isna(), df2[col].isna()):
                if pd.np.array_equal(df1[col].dropna(), df2[col].dropna()):
                    continue
                else:
                    diff = pd.Series(pd.np.subtract(df1.dropna()[col].values,
                                                    df2.dropna()[col].values) > epsilon, df1[col])
            else:
                print(df1)
                print(df2)
                return Companda(False, True,
                                f"columns and indices equal; values have different NaN values in {col}.")
            if diff.sum() == 0:
                continue
            else:
                return Companda(False, True,
                                f"columns and indices equal; values not almost equal in column {col}.")
        elif is_datetime64_any_dtype(df1[col]):
            if df1[col].dt.tz != df2[col].dt.tz:
                return Companda(False, True,
                                f"unequal timezones: {df1[col].dt.tz}, {df2[col].dt.tz}")
            if pd.np.array_equal(df1[col].isna(),
                                 df2[col].isna()) and pd.np.array_equal(df1[col].dropna().values,
                                                                        df2[col].dropna().values):
                continue
            else:
                print(df1)
                print(df2)
                return Companda(False, True,
                                f"columns and indices equal; datetime values different in {col}.")
        else:
            if pd.np.array_equal(df1[col].isna(),
                                 df2[col].isna()) and pd.np.array_equal(df1[col].dropna(),
                                                                        df2[col].dropna()):
                continue
            else:
                print(f'Unequal values in column {col}:')
                for i in range(len(df1)):
                    if df1[col].iloc[i] != df2[col].iloc[i]:
                        print(i, df1[col].iloc[i], df2[col].iloc[i])
                return Companda(False, True,
                                f"columns and indices equal; values not equal in column {col}. {df1[col]} {df2[col]}")

    return Companda(True, True, f'EQUAL, checked_dtype={check_dtype}, ignore_index={ignore_index}')
