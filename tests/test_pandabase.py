"""
Tests for pandabase.

Note that the working directory is project root; when running tests from PyCharm,
the default may be root/tests. Can be set from Run/Debug Configurations:Templates
"""
import pytest

import pandabase as pb
from pandabase.helpers import *

import pandas as pd
from pandas.api.types import (is_bool_dtype,
                              is_datetime64_any_dtype,
                              is_integer_dtype,
                              is_float_dtype)

from sqlalchemy.exc import StatementError
from sqlalchemy import Integer, String, Float, DateTime, Boolean

import os
import logging
TEST_LOG = os.path.join('tests', 'test_log.log')
# rewrite logfile following each test
logging.basicConfig(level=logging.DEBUG, filename=TEST_LOG, filemode='w')

# Arbitrary data (partially overlapping columns, disjoint indexes) for tests
FILE1 = os.path.join('tests', 'sample_data.csv.zip')
FILE2 = os.path.join('tests', 'sample_data2.csv.zip')
FILE3 = os.path.join('tests', 'sample_data3.csv')


@pytest.fixture(scope='function')
def empty_db():
    """In-memory database fixture; not persistent"""
    return pb.engine_builder('sqlite:///:memory:')


@pytest.fixture(scope='session')
def session_db():
    """In-memory database fixture; persistent through session"""
    return pb.engine_builder('sqlite:///:memory:')


@pytest.fixture(scope='session')
def sample_csv_dfs():
    dfs = []
    for file in [FILE1, FILE2, FILE3]:
        sample_data = pd.read_csv(file)
        sample_data.columns = [pb.clean_name(col) for col in sample_data.columns]
        # if csv has non-unique index, select a subset of data w/ unique index (specific to example)
        if 'lmp_type' in sample_data.columns:
            sample_data = sample_data[sample_data.lmp_type == 'LMP']
        for df_col in sample_data.columns:
            if 'time' in df_col.lower():
                sample_data[df_col] = pd.to_datetime(sample_data[df_col])
        dfs.append(sample_data)
    return dfs


@pytest.fixture(scope='session')
def simple_df():
    """make a dumb DataFrame with multiple dtypes and integer index"""
    rows = 10
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean', 'nan'],
                      index=range(rows),)
    df.date = pd.date_range(pd.to_datetime('2001-01-01 12:00am', utc=True), periods=10, freq='d')
    df.integer = range(7, 17)
    df.float = [float(i) / 10 for i in range(10)]
    df.string = list('panda_base')
    df.boolean = [True, False] * 5
    df.nan = [None] * 10
    return df


@pytest.fixture(scope='session')
def simple_df2():
    """make a dumb DataFrame with multiple dtypes and integer index"""
    rows = 10
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean', 'nan'],
                      index=range(rows),)
    df.date = pd.date_range(pd.to_datetime('2006-01-01 12:00am', utc=True), periods=10, freq='d')
    df.integer = range(17, 27)
    df.float = [float(i) / 99 for i in range(10)]
    df.string = ['x' * i for i in range(10)]
    df.boolean = [True] * 10
    df.nan = [None] * 10
    df.loc[1, 'float'] = None
    df.loc[2, 'integer'] = None
    df.loc[3, 'datetime'] = None
    df.loc[4, 'string'] = None
    return df


# BASIC TESTS #


def test_get_sql_dtype_df(simple_df, simple_df2):
    """test that datatype functions work as expected"""
    for df in [simple_df, simple_df2]:
        assert isinstance(df.index, pd.RangeIndex)

        assert is_datetime64_any_dtype(df.date)
        assert get_df_sql_dtype(df.date) == DateTime

        # assert is_integer_dtype(df.integer)
        # assert not is_float_dtype(df.integer)
        assert get_df_sql_dtype(df.integer) == Integer

        assert is_float_dtype(df.float)
        assert not is_integer_dtype(df.float)
        assert get_df_sql_dtype(df.float) == Float

        assert not is_integer_dtype(df.string) and not is_float_dtype(df.string) and \
            not is_datetime64_any_dtype(df.string) and not is_bool_dtype(df.string)
        assert get_df_sql_dtype(df.string) == String

        assert is_bool_dtype(df.boolean)
        assert get_df_sql_dtype(df.boolean) == Boolean

        assert get_df_sql_dtype(df.nan) is None


def test_get_sql_dtype_db(simple_df, empty_db):
    """test that datatype functions work as expected"""
    df = simple_df
    table = pb.to_sql(simple_df,
                      use_index=False,
                      table_name='sample',
                      con=empty_db,
                      how='fail')
    for col in table.columns:
        if col.name == PANDABASE_DEFAULT_INDEX:
            continue
        assert get_db_col_dtype(col, pd_or_sqla='sqla') == get_df_sql_dtype(df[col.name])


# WRITE TO SQL TESTS #


def test_create_table(session_db, simple_df):
    assert not pb.has_table(session_db, 'sample')

    table = pb.to_sql(simple_df,
                      use_index=False,
                      table_name='sample',
                      con=session_db,
                      how='fail')
    assert table.columns['pandabase_index'].primary_key
    assert pb.has_table(session_db, 'sample')

    loaded = pb.read_sql('sample', con=session_db)
    # print(loaded)
    assert pb.companda(loaded, simple_df, ignore_nan=True)
    assert pb.has_table(session_db, 'sample')


def test_overwrite_table_fails(session_db, simple_df):
    assert pb.has_table(session_db, 'sample')

    with pytest.raises(NameError):
        pb.to_sql(simple_df,
                  use_index=False,
                  table_name='sample',
                  con=session_db,
                  how='fail')


@pytest.mark.parametrize('table_name, col_name', [['integer_index', 'integer'],
                                                  ['float_index', 'float'],
                                                  ['datetime_index', 'date'], ])
def test_create_table_with_index(session_db, simple_df, table_name, col_name):
    df = simple_df.copy()
    df.index = df[col_name]
    df.drop(col_name, axis=1, inplace=True)

    table = pb.to_sql(df,
                      use_index=True,
                      table_name=table_name,
                      con=session_db,
                      how='fail')

    assert table.columns[col_name].primary_key
    assert pb.has_table(session_db, table_name)

    loaded = pb.read_sql(table_name, con=session_db)
    c = pb.companda(loaded, df, ignore_nan=True)
    if not c:
        raise ValueError(c.msg)


def test_upsert_fails_no_index(session_db, simple_df):
    table_name = 'sample'
    with pytest.raises(IOError):
        pb.to_sql(simple_df,
                  use_index=False,
                  table_name=table_name,
                  con=session_db,
                  how='upsert')


def test_upsert(session_db):
    table_name = 'integer_index'
    assert pb.has_table(session_db, table_name)
    df = pb.read_sql(table_name, con=session_db)

    df.loc[1, 'float'] = 999
    df.loc[2, 'string'] = 'fitty'
    df.loc[111, 'float'] = 9999

    pb.to_sql(df,
              use_index=True,
              table_name=table_name,
              con=session_db,
              how='upsert')

    loaded = pb.read_sql(table_name, con=session_db)

    assert loaded.loc[1, 'float'] == 999
    assert loaded.loc[111, 'float'] == 9999
    assert loaded.loc[2, 'string'] == 'fitty'
    assert loaded.loc[111, 'string'] is None


def test_upsert_fails(session_db, simple_df):
    df = simple_df.copy()
    df.index = df.integer
    df.drop('integer', axis=1, inplace=True)

    print(df.index.name)

    assert pb.has_table(session_db, 'integer_index')

    df.loc[1, 'float'] = 'x'
    df.loc[3, 'date'] = 'cat'

    for i in range(1, 4):
        with pytest.raises(StatementError):
            print(i)
            pb.to_sql(df[i:i+1],
                      use_index=True,
                      table_name='integer_index',
                      con=session_db,
                      how='upsert')
