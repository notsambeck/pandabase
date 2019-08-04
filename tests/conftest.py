import pandas as pd
import pytest

import pandabase as pb
from pandas import set_option
from os.path import join
from logging import basicConfig, DEBUG
from types import SimpleNamespace

import pytz
UTC = pytz.utc

# pd.set_option('display.max_colwidth', 12)
set_option('expand_frame_repr', True)

TABLE_NAME = 'sample_table_name'
SAMPLE_INDEX_NAME = 'sample_named_integer_index'
TEST_LOG = join('tests', 'test_log.log')
# rewrite logfile following each test
basicConfig(level=DEBUG, filename=TEST_LOG, filemode='w')

# Arbitrary data (partially overlapping columns, disjoint indexes) for tests
FILE1 = join('tests', 'sample_data.csv.zip')
FILE2 = join('tests', 'sample_data2.csv.zip')
FILE3 = join('tests', 'sample_data3.csv')


@pytest.fixture(scope='session')
def constants():
    d = {
        'FILE1': FILE1,
        'FILE2': FILE2,
        'FILE3': FILE3,
        'TABLE_NAME': TABLE_NAME,
        'SAMPLE_INDEX_NAME': SAMPLE_INDEX_NAME,
        'TEST_LOG': TEST_LOG,
    }
    return SimpleNamespace(**d)


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

    assert df.date[0].tzinfo == UTC
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
