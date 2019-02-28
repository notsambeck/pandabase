"""
Tests for pandabase.

Note that the working directory is project root; when running tests from PyCharm,
the default may be root/tests. Can be set from Run/Debug Configurations:Templates
"""
import pytest

import pandabase as pb
from pandabase.helpers import *
from pandabase.companda import companda

import pandas as pd
from pandas.api.types import (is_bool_dtype,
                              is_datetime64_any_dtype,
                              is_integer_dtype,
                              is_float_dtype)

from sqlalchemy import Integer, String, Float, DateTime, Boolean
from sqlalchemy.exc import IntegrityError

import os
import logging

pd.set_option('display.max_colwidth', 8)

TABLE_NAME = 'pre_loaded_table'
INDEX_NAME = 'user_integer_index'
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


@pytest.fixture(scope='function')
def full_db(empty_db, simple_df):
    """In-memory database fixture; not persistent"""
    pb.to_sql(simple_df,
              table_name=TABLE_NAME,
              con=empty_db,
              how='fail')
    return empty_db


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


@pytest.fixture(scope='function')
def simple_df():
    """make a basic DataFrame with multiple dtypes, integer index"""
    rows = 6
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean'],
                      index=range(rows),)
    df.index.name = INDEX_NAME

    df.date = pd.date_range(pd.to_datetime('2001-01-01 12:00am', utc=True), periods=rows, freq='d')
    df.integer = range(1, rows+1)
    df.float = [float(i) / 10 for i in range(rows)]
    df.string = list('panda_base')[:rows]
    df.boolean = [True, False] * (rows//2)

    return df


@pytest.fixture(scope='function')
def df_with_nan_col():
    """make a dumb DataFrame with multiple dtypes and integer index"""
    rows = 10
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean', 'nan'],
                      index=range(rows),)
    df.index.name = INDEX_NAME

    df.date = pd.date_range(pd.to_datetime('2001-01-01 12:00am', utc=True), periods=10, freq='d')
    df.integer = range(1, rows+1)
    df.float = [float(i) / 10 for i in range(10)]
    df.string = list('panda_base')
    df.boolean = [True, False] * 5
    df.nan = [None] * 10

    print(df)

    return df


@pytest.fixture(scope='session')
def simple_df2():
    """make a dumb DataFrame with multiple dtypes and integer index, index is valid but columns.hasnans"""
    rows = 10
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean', 'nan'],
                      index=range(rows),)
    df.index.name = INDEX_NAME

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


def test_get_sql_dtype_df(df_with_nan_col):
    """test that datatype functions work as expected"""
    df = df_with_nan_col

    assert isinstance(df.index, pd.RangeIndex)

    assert is_datetime64_any_dtype(df.date)
    assert get_column_dtype(df.date, 'sqla') == DateTime
    assert get_column_dtype(df.date, 'pd') == np.datetime64

    # assert is_integer_dtype(df.integer)
    # assert not is_float_dtype(df.integer)
    assert get_column_dtype(df.integer, 'sqla') == Integer
    assert get_column_dtype(df.integer, 'pd') == pd.Int64Dtype()

    assert is_float_dtype(df.float)
    assert not is_integer_dtype(df.float)
    assert get_column_dtype(df.float, 'sqla') == Float

    assert get_column_dtype(df.string, 'sqla') == String

    assert is_bool_dtype(df.boolean)
    assert get_column_dtype(df.boolean, 'sqla') == Boolean

    assert get_column_dtype(df.nan, 'pd') is None


def test_get_sql_dtype_db(simple_df, empty_db):
    """test that datatype functions work as expected"""
    df = simple_df
    table = pb.to_sql(simple_df,
                      table_name='sample',
                      con=empty_db,
                      how='fail')

    for col in table.columns:
        if col.primary_key:
            assert get_column_dtype(col, 'sqla') == get_column_dtype(df.index, 'sqla')
            continue
        assert get_column_dtype(col, 'sqla') == get_column_dtype(df[col.name], 'sqla')


def test_read_table(full_db, simple_df):
    """read pre-written table with pb.read_sql"""
    assert has_table(full_db, TABLE_NAME)

    df = pb.read_sql(TABLE_NAME, full_db)

    orig_dict = make_clean_columns_dict(simple_df)
    df_dict = make_clean_columns_dict(df)
    for key in orig_dict.keys():
        if key == 'nan':
            # column of all NaN values is skipped
            continue
        assert orig_dict[key] == df_dict[key]
    assert companda(df, simple_df)


def test_create_table(session_db, simple_df):
    """add a new table, read it back, check equality"""
    table = pb.to_sql(simple_df,
                      table_name='sample',
                      con=session_db,
                      how='fail')

    # print(table.columns)
    assert table.columns[INDEX_NAME].primary_key

    loaded = pb.read_sql('sample', con=session_db)
    # print(loaded)
    assert pb.companda(loaded, simple_df, ignore_nan=True)
    assert pb.has_table(session_db, 'sample')


@pytest.mark.parametrize('how', ['fail', 'append'])
def test_overwrite_table_fails(full_db, simple_df, how):
    """Try to append/insert rows with conflicting indices"""
    table_name = TABLE_NAME
    assert pb.has_table(full_db, table_name)

    with pytest.raises(Exception):
        pb.to_sql(simple_df,
                  table_name=table_name,
                  con=full_db,
                  how=how)


@pytest.mark.parametrize('table_name, index_col_name', [['integer_index', 'integer'],
                                                        ['float_index', 'float'],
                                                        ['datetime_index', 'date'], ])
def test_create_table_with_index(session_db, simple_df, table_name, index_col_name):
    """create new tables in empty db, using different col types as index"""
    df = simple_df.copy()
    df.index = df[index_col_name]
    df.drop(index_col_name, axis=1, inplace=True)

    table = pb.to_sql(df,
                      table_name=table_name,
                      con=session_db,
                      how='fail')

    assert table.columns[index_col_name].primary_key
    assert pb.has_table(session_db, table_name)

    loaded = pb.read_sql(table_name, con=session_db)
    c = pb.companda(loaded, df, ignore_nan=True)
    if not c:
        raise ValueError(c.msg)


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_new_rows(full_db, simple_df, how):
    """upsert or append new complete rows"""
    assert pb.has_table(full_db, TABLE_NAME)

    df = simple_df.copy()
    df.index = df.index + 100

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=full_db,
              how=how)

    loaded = pb.read_sql(TABLE_NAME, con=full_db)
    # print('loaded post-upsert by pandabase:')
    # print(loaded)

    assert loaded.isna().sum().sum() == 0
    assert companda(simple_df, loaded.loc[simple_df.index])
    assert companda(df, loaded.loc[df.index])


def test_upsert_complete_rows(full_db):
    """upsert, changing individual values"""
    assert pb.has_table(full_db, TABLE_NAME)
    df = pb.read_sql(TABLE_NAME, con=full_db)

    df.loc[1, 'float'] = 9.9
    df.loc[2, 'integer'] = 999
    df.loc[3, 'string'] = 'nah'
    df.loc[4, 'date'] = pd.to_datetime('1968-01-01', utc=True)

    # check that these values still exist
    assert df.loc[1, 'integer'] == 2

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=full_db,
              how='upsert')

    loaded = pb.read_sql(TABLE_NAME, con=full_db)
    assert companda(df, loaded)

    loaded_pd = pd.read_sql(TABLE_NAME, con=full_db, index_col=INDEX_NAME)
    assert companda(df, loaded_pd)


def test_upsert_incomplete_rows(full_db):
    """upsert new rows with only 1 of 5 values (and index)"""
    assert pb.has_table(full_db, TABLE_NAME)
    df = pb.read_sql(TABLE_NAME, con=full_db)
    types = df.dtypes

    df.loc[11, 'float'] = 9.9
    df.loc[12, 'integer'] = 999
    df.loc[13, 'string'] = 'nah'
    df.loc[14, 'date'] = pd.to_datetime('1968-01-01', utc=True)

    for col in df.columns:
        print(col)
        assert types[col] == df.dtypes[col]

    # check that these values still exist
    assert df.loc[1, 'integer'] == 2
    assert pd.isna(df.loc[11, 'integer'])

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=full_db,
              how='upsert')

    loaded = pb.read_sql(TABLE_NAME, con=full_db)
    assert companda(df, loaded)

    loaded_pd = pd.read_sql(TABLE_NAME, con=full_db, index_col=INDEX_NAME)
    assert companda(df, loaded_pd)


def test_upsert_coerce_float(full_db):
    """insert an integer into float column"""
    assert pb.has_table(full_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['float'], data=[[1]])
    df.index.name = INDEX_NAME
    types = df.dtypes

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=full_db,
              how='upsert')

    for col in df.columns:
        print(col)
        assert types[col] == df.dtypes[col]

    loaded = pb.read_sql(TABLE_NAME, con=full_db)
    assert loaded.loc[1, 'float'] == 1.0


def test_upsert_coerce_integer(full_db):
    """insert an integer into float column"""
    assert pb.has_table(full_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['integer'], data=[[1.0]])
    df.index.name = INDEX_NAME
    types = df.dtypes

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=full_db,
              how='upsert')

    for col in df.columns:
        print(col)
        assert types[col] == df.dtypes[col]

    loaded = pb.read_sql(TABLE_NAME, con=full_db)
    assert loaded.loc[1, 'integer'] == 1


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_fails_wrong_index_name(full_db):
    assert pb.has_table(full_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['date'], data=[['x']])
    df.index_name = 'not_a_real_name'

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=TABLE_NAME,
                  con=full_db,
                  how='how')


@pytest.mark.parametrize('how', ['upsert', 'append'])
def test_upsert_fails_invalid_float(full_db, how):
    assert pb.has_table(full_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['float'], data=[['x']])

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=TABLE_NAME,
                  con=full_db,
                  how=how)


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_fails_invalid_date(full_db, how):
    assert pb.has_table(full_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['date'], data=[['x']])

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=TABLE_NAME,
                  con=full_db,
                  how=how)


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_fails_wrong_index_name(full_db, how):
    assert pb.has_table(full_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['date'], data=[['x']])

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=TABLE_NAME,
                  con=full_db,
                  how=how)

