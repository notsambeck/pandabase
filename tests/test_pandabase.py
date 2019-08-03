"""
Tests for pandabase.

Note that the working directory is project root; when running tests from PyCharm,
the default may be root/tests. Can be set from Run/Debug Configurations:Templates
"""
import pytest

import os
import logging

from sqlalchemy import Integer, String, Float, DateTime, Boolean

import pandas as pd
from pandas.api.types import (is_bool_dtype,
                              is_datetime64_any_dtype,
                              is_integer_dtype,
                              is_float_dtype,
                              )

import pandabase as pb
from pandabase.helpers import *
from pandabase.companda import companda

import pytz
UTC = pytz.utc


pd.set_option('display.max_colwidth', 8)

TABLE_NAME = 'sample_table_name'
SAMPLE_INDEX_NAME = 'sample_named_integer_index'
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
    print('empty db fixture setup...')
    return pb.engine_builder('sqlite:///:memory:')


@pytest.fixture(scope='function')
def pre_loaded_db(empty_db, simple_df):
    """In-memory database fixture; not persistent"""
    print('full db fixture setup...')
    pb.to_sql(simple_df,
              table_name=TABLE_NAME,
              con=empty_db,
              how='create_only')
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
def minimal_df():
    """make a basic DataFrame with multiple dtypes, range index"""
    rows = 6
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean'])

    df.date = pd.date_range(pd.to_datetime('2001-01-01 12:00am', utc=True), periods=rows, freq='d', tz=UTC)
    df.integer = range(rows)
    df.float = [float(i) / 10 for i in range(rows)]
    df.string = list('panda_base')[:rows]
    df.boolean = [True, False] * (rows//2)

    assert df.date[0].tzinfo is not None
    return df


@pytest.fixture(scope='function')
def simple_df():
    """make a basic DataFrame with multiple dtypes, integer index"""
    rows = 6
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean'],
                      index=range(rows),)
    df.index.name = SAMPLE_INDEX_NAME

    df.date = pd.date_range(pd.to_datetime('2001-01-01 12:00am', utc=True), periods=rows, freq='d', tz=UTC)
    df.integer = range(1, rows+1)
    df.float = [float(i) / 10 for i in range(rows)]
    df.string = list('panda_base')[:rows]
    df.boolean = [True, False] * (rows//2)

    assert df.date[0].tzinfo is not None
    return df


@pytest.fixture(scope='function')
def df_with_all_nan_col():
    """make a dumb DataFrame with multiple dtypes and integer index"""
    rows = 10
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean', 'nan'],
                      index=range(rows),)
    df.index.name = SAMPLE_INDEX_NAME

    df.date = pd.date_range(pd.to_datetime('2001-01-01 12:00am', utc=True), periods=10, freq='d')
    df.integer = range(1, rows+1)
    df.float = [float(i) / 10 for i in range(10)]
    df.string = list('panda_base')
    df.boolean = [True, False] * 5
    df.nan = [None] * 10

    return df


@pytest.fixture(scope='session')
def simple_df_with_nans():
    """make a dumb DataFrame with multiple dtypes and integer index, index is valid but columns.hasnans"""
    rows = 10
    df = pd.DataFrame(columns=['date', 'integer', 'float', 'string', 'boolean', 'nan'],
                      index=range(rows),)
    df.index.name = SAMPLE_INDEX_NAME

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


# ========== BASIC TESTS ======== #


@pytest.mark.parametrize('df, how', [
    (pd.DataFrame(index=[2, 2], data=['x', 'y']), 'create_only'),
    (pd.DataFrame(index=[1, None], columns=['a'], data=['x', 'y']), 'create_only'),
    (pd.Series([1, 2, 3]), 'create_only'),
    (pd.DataFrame(index=[1, 2, 3], columns=['a'], data=['x', 'y', 'a']), 'fake_mode'),
])
def test_invalid_inputs(df, how, empty_db):
    with pytest.raises(ValueError):
        pb.to_sql(df, table_name='blank', con=empty_db, how=how)


def test_get_sql_dtype_df(df_with_all_nan_col):
    """test that datatype functions work as expected"""
    df = df_with_all_nan_col

    assert isinstance(df.index, pd.RangeIndex)

    assert is_datetime64_any_dtype(df.date)
    assert get_column_dtype(df.date, 'sqla') == DateTime
    assert get_column_dtype(df.date, 'pd') == np.datetime64

    assert is_integer_dtype(df.integer)
    assert not is_float_dtype(df.integer)

    assert get_column_dtype(df.integer, 'sqla') == Integer
    assert get_column_dtype(df.integer, 'pd') == pd.Int64Dtype()

    assert is_float_dtype(df.float)
    assert not is_integer_dtype(df.float)
    assert get_column_dtype(df.float, 'sqla') == Float

    assert get_column_dtype(df.string, 'sqla') == String

    assert is_bool_dtype(df.boolean)
    assert get_column_dtype(df.boolean, 'sqla') == Boolean

    assert get_column_dtype(df.nan, 'pd') is None


def test_get_sql_dtype_from_db(simple_df, empty_db):
    """test that datatype extraction functions work"""
    df = simple_df
    table = pb.to_sql(df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    for col in table.columns:
        if col.primary_key:
            assert get_column_dtype(col, 'sqla') == get_column_dtype(df.index, 'sqla')
            continue
        assert get_column_dtype(col, 'sqla') == get_column_dtype(df[col.name], 'sqla')


def test_get_sql_dtype_from_db_nans(simple_df_with_nans, empty_db):
    """test that datatype extraction functions work as expected"""
    df = simple_df_with_nans
    table = pb.to_sql(df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    for col in table.columns:
        if col.primary_key:
            assert get_column_dtype(col, 'sqla') == get_column_dtype(df.index, 'sqla')
            continue
        assert get_column_dtype(col, 'sqla') == get_column_dtype(df[col.name], 'sqla')


@pytest.mark.parametrize('series, expected', [
    [pd.Series([True, False, True, ], dtype=int), True],
    [pd.Series([True, False, True, ], dtype=float), True],
    [pd.Series([True, False, None, ], dtype=float), True],
    [pd.Series([True, False, True, ]), True],
    [pd.Series([True, False, None, ]), True],
    [pd.Series([True, False, 4.4, ]), False],
    [pd.Series([True, False, 1.0, ]), True],
    [pd.Series([True, False, 0, ]), True],
    [pd.Series([1, 0, 0, ]), True],
    [pd.Series([1, 0, 2, ]), False],
    [pd.Series([1, 0, None, ]), True],
    [pd.Series([None, None, None, ]), None],
    # [pd.Series([np.NaN], dtype=bool), None],   # doesn't work - bool coerces NaN to False
    [pd.Series([np.NaN], dtype=str), None],
    [pd.Series([np.NaN], dtype=float), None],
    [pd.Series([0, 1, pd.to_datetime('2017-01-12')]), False],
    [pd.Series([0, 1, pd.to_datetime('2000')]), False],
    [pd.Series([pd.to_datetime('2000'), pd.to_datetime('2017-01-12')]), False],
    [pd.Series(np.array([True, False, np.NaN, ]), dtype='Int64'), True],    # Int64 is broken
    [pd.Series([np.NaN], dtype=pd.Int64Dtype()), None],                     # Int64 is broken
])
def test_series_is_boolean(series, expected):
    assert isinstance(series, pd.Series)
    print(series)
    assert series_is_boolean(series) == expected


def test_read_table(pre_loaded_db, simple_df):
    """read pre-written table containing simple_df, using pb.read_sql"""
    assert has_table(pre_loaded_db, TABLE_NAME)

    df = pb.read_sql(TABLE_NAME, pre_loaded_db)

    orig_columns = make_clean_columns_dict(simple_df)
    loaded_columns = make_clean_columns_dict(df)
    for key in orig_columns.keys():
        print(key)
        if key == 'nan':
            # column of all NaN values is skipped
            continue
        assert orig_columns[key] == loaded_columns[key]
    assert companda(df, simple_df)


def test_create_read_table_no_index(empty_db, minimal_df):
    """add a new minimal table"""
    table = pb.to_sql(minimal_df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    # print(table.columns)
    assert table.columns[PANDABASE_DEFAULT_INDEX].primary_key
    loaded = pb.read_sql('sample', con=empty_db)
    assert pb.has_table(empty_db, 'sample')
    assert pb.companda(loaded, minimal_df)


def test_create_read_table_index(session_db, simple_df):
    """add a new table, read it back, check equality"""
    table = pb.to_sql(simple_df,
                      table_name='sample',
                      con=session_db,
                      how='create_only')

    # print(table.columns)
    assert table.columns[SAMPLE_INDEX_NAME].primary_key

    loaded = pb.read_sql('sample', con=session_db)
    # print(loaded)
    assert pb.companda(loaded, simple_df, ignore_all_nan_columns=True)
    assert pb.has_table(session_db, 'sample')


@pytest.mark.parametrize('how', ['create_only', 'append'])
def test_overwrite_table_fails(pre_loaded_db, simple_df, how):
    """Try to append/insert rows with conflicting indices"""
    table_name = TABLE_NAME
    assert pb.has_table(pre_loaded_db, table_name)

    with pytest.raises(Exception):
        pb.to_sql(simple_df,
                  table_name=table_name,
                  con=pre_loaded_db,
                  how=how)


@pytest.mark.parametrize('table_name, index_col_name', [['integer_index_table', 'integer'],
                                                        ['float_index_table', 'float'],
                                                        ['datetime_index_table', 'date'], ])
def test_create_table_with_different_index(session_db, simple_df, table_name, index_col_name):
    """create new tables in empty db, using different col types as index"""
    df = simple_df.copy()
    df.index = df[index_col_name]
    df.drop(index_col_name, axis=1, inplace=True)

    table = pb.to_sql(df,
                      table_name=table_name,
                      con=session_db,
                      how='create_only')

    assert table.columns[index_col_name].primary_key
    assert pb.has_table(session_db, table_name)

    loaded = pb.read_sql(table_name, con=session_db)
    c = pb.companda(loaded, df, ignore_all_nan_columns=True)
    if not c:
        raise ValueError(c.msg)


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_new_rows(pre_loaded_db, simple_df, how):
    """upsert or append new complete rows"""
    assert pb.has_table(pre_loaded_db, TABLE_NAME)

    df = simple_df.copy()
    df.index = df.index + 100

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=pre_loaded_db,
              how=how)

    loaded = pb.read_sql(TABLE_NAME, con=pre_loaded_db)
    # print('loaded post-upsert by pandabase:')
    # print(loaded)

    assert loaded.isna().sum().sum() == 0
    assert companda(simple_df, loaded.loc[simple_df.index])
    assert companda(df, loaded.loc[df.index])


def test_upsert_complete_rows(pre_loaded_db):
    """upsert, changing individual values"""
    assert pb.has_table(pre_loaded_db, TABLE_NAME)
    df = pb.read_sql(TABLE_NAME, con=pre_loaded_db)

    df.loc[1, 'float'] = 9.9
    df.loc[2, 'integer'] = 999
    df.loc[3, 'string'] = 'nah'
    df.loc[4, 'date'] = pd.to_datetime('1968-01-01', utc=True)

    # check that these values still exist
    assert df.loc[1, 'integer'] == 2

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=pre_loaded_db,
              how='upsert')

    loaded = pb.read_sql(TABLE_NAME, con=pre_loaded_db)
    assert companda(df, loaded)

    loaded_pd = pd.read_sql(TABLE_NAME, con=pre_loaded_db, index_col=SAMPLE_INDEX_NAME)
    assert companda(df, loaded_pd)


def test_upsert_incomplete_rows(pre_loaded_db):
    """upsert new rows with only 1 of 5 values (and index)"""
    assert pb.has_table(pre_loaded_db, TABLE_NAME)
    df = pb.read_sql(TABLE_NAME, con=pre_loaded_db)

    df.loc[11, 'float'] = 9.9
    df.loc[12, 'integer'] = 999
    df.loc[13, 'string'] = 'nah'
    df.loc[14, 'date'] = pd.to_datetime('1968-01-01', utc=True)

    # check that these values exist
    assert df.loc[1, 'integer'] == 2
    assert pd.isna(df.loc[11, 'integer'])
    assert df.loc[13, 'string'] == 'nah'

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=pre_loaded_db,
              how='upsert')

    # check against pandabase read
    loaded = pb.read_sql(TABLE_NAME, con=pre_loaded_db)
    assert companda(df, loaded)

    # check against pandas read
    loaded_pd = pd.read_sql(TABLE_NAME, con=pre_loaded_db, index_col=SAMPLE_INDEX_NAME)
    assert companda(df, loaded_pd)


def test_upsert_coerce_float(pre_loaded_db):
    """insert an integer into float column"""
    assert pb.has_table(pre_loaded_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['float'], data=[[1.0]])
    df.index.name = SAMPLE_INDEX_NAME
    types = df.dtypes

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=pre_loaded_db,
              how='upsert')

    for col in df.columns:
        print(col)
        assert types[col] == df.dtypes[col]

    loaded = pb.read_sql(TABLE_NAME, con=pre_loaded_db)
    assert loaded.loc[1, 'float'] == 1.0


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_coerce_integer(pre_loaded_db, how):
    """insert an integer into float column"""
    assert pb.has_table(pre_loaded_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['integer'], data=[[77.0]])
    df.index.name = SAMPLE_INDEX_NAME
    types = df.dtypes

    pb.to_sql(df,
              table_name=TABLE_NAME,
              con=pre_loaded_db,
              how='upsert')

    for col in df.columns:
        print(col)
        assert types[col] == df.dtypes[col]

    loaded = pb.read_sql(TABLE_NAME, con=pre_loaded_db)
    assert loaded.loc[1, 'integer'] == 77


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_new_column_fails(pre_loaded_db, how):
    """insert into a new column"""
    assert pb.has_table(pre_loaded_db, TABLE_NAME)

    df = pd.DataFrame(index=[101], columns=['new_column'], data=[[1.1]])
    df.index.name = SAMPLE_INDEX_NAME
    assert df.loc[101, 'new_column'] == 1.1

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=TABLE_NAME,
                  con=pre_loaded_db,
                  how=how,
                  strict=False)


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_fails_wrong_index_name(pre_loaded_db, how):
    assert pb.has_table(pre_loaded_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['date'], data=[['x']])
    df.index_name = 'not_a_real_name'

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=TABLE_NAME,
                  con=pre_loaded_db,
                  how=how)


@pytest.mark.parametrize('how', ['upsert', 'append'])
def test_upsert_fails_invalid_float(pre_loaded_db, how):
    assert pb.has_table(pre_loaded_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['float'], data=[['x']])

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=TABLE_NAME,
                  con=pre_loaded_db,
                  how=how)


@pytest.mark.parametrize('how', ['upsert', 'append'])
def test_upsert_fails_invalid_bool(pre_loaded_db, how):
    assert pb.has_table(pre_loaded_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['bool'], data=[['x']])

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=TABLE_NAME,
                  con=pre_loaded_db,
                  how=how)


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_fails_invalid_date(pre_loaded_db, how):
    assert pb.has_table(pre_loaded_db, TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['date'], data=[['x']])

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=TABLE_NAME,
                  con=pre_loaded_db,
                  how=how)
