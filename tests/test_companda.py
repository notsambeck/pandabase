import pytest
import pandas as pd
import numpy as np
from pandabase.companda import companda
import pytz
TZ = pytz.timezone('America/Los_Angeles')


def test_same_companda1(simple_df):
    assert companda(simple_df, simple_df)


def test_same_companda_copy1(simple_df):
    assert companda(simple_df, simple_df.copy())


def test_same_companda_select(simple_df):
    assert companda(simple_df[sorted(simple_df.columns)], simple_df)


def test_same_companda2(minimal_df):
    assert companda(minimal_df, minimal_df)


def test_same_companda_copy2(minimal_df):
    assert companda(minimal_df, minimal_df.copy())


def test_same_companda_multi(multi_index_df):
    assert companda(multi_index_df, multi_index_df)


def test_same_companda_multi_copy(multi_index_df):
    assert companda(multi_index_df, multi_index_df.copy())


def test_same_companda_cols1(minimal_df):
    df = minimal_df.copy()
    df = df.drop(['float'], axis=1)
    print(companda(df, minimal_df))
    assert not companda(df, minimal_df)


def test_same_companda_cols2(minimal_df):
    df = minimal_df.copy()
    df = df.drop(['float'], axis=1)
    print()
    print(companda(df, minimal_df))
    assert not companda(minimal_df, df)


def test_same_companda_cols3(minimal_df):
    df = minimal_df.copy()
    df = df.rename(columns={'integer': 'x'})
    print()
    print(companda(df, minimal_df))
    assert not companda(df, minimal_df)


def test_same_companda_cols4(minimal_df):
    df = minimal_df.copy()
    df = df.rename(columns={'integer': 'x'})
    print()
    print(companda(df, minimal_df))
    assert not companda(minimal_df, df)


def test_same_companda_index1(minimal_df):
    df = minimal_df.copy()
    df = df.rename(index={1: 99})
    assert not companda(df, minimal_df)


def test_same_companda_index2(minimal_df):
    df = minimal_df.copy()
    df = df.drop(1, axis=0)
    assert not companda(df, minimal_df)


def test_all_nans_ignore(df_with_all_nan_col):
    assert companda(df_with_all_nan_col, df_with_all_nan_col, ignore_all_nan_columns=True)


def test_all_nans_do_not_ignore(df_with_all_nan_col):
    assert companda(df_with_all_nan_col, df_with_all_nan_col, ignore_all_nan_columns=False)


def test_added_nans_ignore(simple_df, df_with_all_nan_col):
    assert companda(df_with_all_nan_col, simple_df, ignore_all_nan_columns=True)


def test_added_nans_do_not_ignore(simple_df, df_with_all_nan_col):
    assert not companda(df_with_all_nan_col, simple_df, ignore_all_nan_columns=False)


def test_different_companda(minimal_df, simple_df):
    assert not companda(minimal_df, simple_df)


def test_same_companda_alter_dtype(minimal_df):
    """changing between types changes equality (e.g. bool!=int)"""
    df2 = minimal_df.copy()
    # print(type(minimal_df.boolean[0]))
    # print(type(df2.boolean[0]))
    df2.boolean = df2.boolean.astype(np.int)
    # print(type(minimal_df.boolean[0]))
    # print(type(df2.boolean[0]))
    print(minimal_df.dtypes)
    print(df2.dtypes)
    assert not companda(df2, minimal_df, check_dtype=True)


def test_same_companda_epsilon1(simple_df):
    df = simple_df.copy()
    df.float = df.float.apply(lambda x: x + .0001)
    assert companda(df, simple_df)


def test_same_companda_epsilon2(simple_df):
    df = simple_df.copy()
    df.float = df.float.apply(lambda x: x + .01)
    print()
    print(companda(df, simple_df))
    assert not companda(df, simple_df)


def test_same_companda_epsilon3(simple_df):
    df = simple_df.copy()
    df.integer = df.integer.apply(lambda x: x + .01)
    print()
    print(companda(df, simple_df))
    assert not companda(df, simple_df)


def test_same_companda_epsilon4(simple_df):
    df = simple_df.copy()
    df.integer = df.integer.apply(lambda x: x + 1)
    print()
    print(companda(df, simple_df))
    assert not companda(df, simple_df)


def test_same_companda_nan(simple_df):
    df = simple_df.copy()
    df.iloc[2, 2] = np.NaN
    assert not companda(df, simple_df)


def test_same_companda_string(simple_df):
    df = simple_df.copy()
    df.loc[1, 'string'] = 'z'
    assert not companda(df, simple_df)


def test_same_companda_datetime1day(simple_df):
    df = simple_df.copy()
    df['date'] = df['date'].apply(lambda x: x + pd.Timedelta(days=1))
    assert not companda(df, simple_df)


def test_same_companda_datetime1sec(simple_df):
    df = simple_df.copy()
    df['date'] = df['date'].apply(lambda x: x + pd.Timedelta(seconds=1))
    assert not companda(df, simple_df)


def test_same_companda_datetime2(simple_df):
    df = simple_df.copy()
    df['date'] = pd.to_datetime(df['date'].values, utc=False)
    c = companda(df, simple_df)
    print(c.message)
    assert not c


def test_same_companda_datetime3(simple_df):
    df = simple_df.copy()
    df['date'] = pd.to_datetime(df['date'].values, utc=False).tz_localize(TZ)
    c = companda(df, simple_df)
    print(c.message)
    assert not c
