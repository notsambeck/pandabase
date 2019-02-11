"""
Tests for pandabase.

Note that the working directory is project root; when running tests from PyCharm,
the default may be root/tests. Can be set from Run/Debug Configurations:Templates
"""
import pytest

import pandabase as pb

import pandas as pd
from pandas.api.types import (is_bool_dtype,
                              is_datetime64_any_dtype,
                              is_integer_dtype,
                              is_float_dtype)

from sqlalchemy.exc import IntegrityError
from sqlalchemy import Table, Column, Integer, String, Float, DateTime, Boolean

import os
import logging
TEST_LOG = os.path.join('tests', 'test_log.log')
# rewrite logfile following each test
logging.basicConfig(level=logging.DEBUG, filename=TEST_LOG, filemode='w')

# Arbitrary data (partially overlapping columns, disjoint indexes) for tests
FILE1 = os.path.join('tests', 'sample_data.csv.zip')
FILE2 = os.path.join('tests', 'sample_data2.csv.zip')
FILE3 = os.path.join('tests', 'sample_data3.csv')


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
def basic_df():
    """make a dumb DataFrame with sample data of 4 classes and integer index"""

    cols = 5
    rows = 10
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean'],
                      index=range(rows),)
    df.date = pd.date_range(pd.to_datetime('2001-01-01 12:00am', utc=True), periods=10, freq='d')
    df.integer = range(10)
    df.float = [float(i) / 10 for i in range(10)]
    df.string = list('panda_base')
    df.boolean = [True, False] * 5
    return df


# BASIC TESTS #


def test_basic_df(basic_df):
    """test that datatype functions work as expected"""
    df = basic_df
    assert type(basic_df.index) is pd.RangeIndex

    assert is_datetime64_any_dtype(df.date)
    assert pb.get_sql_dtype(df.date) == DateTime

    assert is_integer_dtype(df.integer)
    assert not is_float_dtype(df.integer)
    assert pb.get_sql_dtype(df.integer) == Integer

    assert is_float_dtype(df.float)
    assert not is_integer_dtype(df.float)
    assert pb.get_sql_dtype(df.float) == Float

    assert not is_integer_dtype(df.string) and not is_float_dtype(df.string) and \
        not is_datetime64_any_dtype(df.string) and not is_bool_dtype(df.string)
    assert pb.get_sql_dtype(df.string) == String

    assert is_bool_dtype(df.boolean)
    assert pb.get_sql_dtype(df.boolean) == Boolean


# WRITE TO SQL TESTS #


def test_create_table(session_db, basic_df):
    assert not pb.has_table(session_db, 'sample')

    table = pb.to_sql(basic_df,
                      index_col_name=None,
                      table_name='sample',
                      con=session_db,
                      how='fail')
    assert table.columns['pandabase_index'].primary_key
    assert pb.has_table(session_db, 'sample')

    loaded = pb.read_sql('sample', con=session_db)
    assert pb.companda(loaded, basic_df)
    assert pb.has_table(session_db, 'sample')


def test_overwrite_table_fails(session_db, basic_df):
    assert pb.has_table(session_db, 'sample')

    with pytest.raises(NameError):
        pb.to_sql(basic_df,
                  index_col_name=None,
                  table_name='sample',
                  con=session_db,
                  how='fail')


@pytest.mark.parametrize('table_name, col_name', [['s1', 'integer'],
                                                  ['s2', 'float'],
                                                  ['s3', 'date'], ])
def test_create_table_with_index(session_db, basic_df, table_name, col_name):
    table = pb.to_sql(basic_df,
                      index_col_name=col_name,
                      table_name=table_name,
                      con=session_db,
                      how='fail')
    assert table.columns[col_name].primary_key

    assert pb.has_table(session_db, table_name)
    loaded = pb.read_sql(table_name, con=session_db)
    c = pb.companda(loaded, basic_df)
    if not c:
        raise ValueError(c.msg)


def test_upsert_fails_no_index(session_db, basic_df):
    table_name = 'sample'
    with pytest.raises(IOError):
        pb.to_sql(basic_df,
                  index_col_name=None,
                  table_name=table_name,
                  con=session_db,
                  how='upsert')


def test_upsert(session_db):
    """TODO - this still fails for upserting incomplete records"""
    table_name = 's1'
    assert pb.has_table(session_db, table_name)
    df = pb.read_sql(table_name, con=session_db)

    df.loc[1, 'float'] = 999

    pb.to_sql(df,
              index_col_name='integer',
              table_name=table_name,
              con=session_db,
              how='upsert')

    loaded = pb.read_sql(table_name, con=session_db)
    assert loaded.loc[1, 'float'] == 999
