"""
Tests for pandabase.

Note that the working directory is project root; when running tests from PyCharm,
the default may be root/tests. Can be set from Run/Debug Configurations:Templates
"""
import pytest
import pandas as pd
import pandabase as pb
from sqlalchemy.exc import IntegrityError
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
def mem_con():
    """In-memory database fixture; persistent through session"""
    return pb.engine_builder('sqlite:///:memory:')


@pytest.fixture(scope='session')
def sample_dfs():
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


# BASIC TESTS #


def test_dir_exists():
    # print(os.listdir())
    assert os.path.exists(FILE1)
    assert os.path.exists(FILE2)


def test_has_table_false(mem_con):
    assert pb.has_table(mem_con, 'sample') is False


# WRITE TO SQL TESTS #


def test_create_table(mem_con, sample_dfs):
    pk = 'intervalstarttime_gmt'
    table = pb.to_sql(sample_dfs[0],
                      index_col_name=pk,
                      table_name='sample',
                      con=mem_con,
                      how='fail')
    print(table.columns[pk].primary_key)
    assert table.columns[pk].primary_key
    assert pb.has_table(mem_con, 'sample') is True


def test_create_table_again_fail(mem_con, sample_dfs):
    pk = 'intervalstarttime_gmt'
    with pytest.raises(NameError):
        pb.to_sql(sample_dfs[0],
                  index_col_name=pk,
                  table_name='sample',
                  con=mem_con,
                  how='fail')


def test_append_fails(mem_con, sample_dfs):
    # can't append if indexes overlap
    pk = 'intervalstarttime_gmt'
    with pytest.raises(IntegrityError):
        pb.to_sql(sample_dfs[0],
                  index_col_name=pk,
                  table_name='sample',
                  con=mem_con,
                  how='append')


def test_upsert(mem_con, sample_dfs):
    pk = 'intervalstarttime_gmt'
    pb.to_sql(sample_dfs[0],
              index_col_name=pk,
              table_name='sample',
              con=mem_con,
              how='upsert')


def test_append_new_index(mem_con, sample_dfs):
    pk = 'intervalstarttime_gmt'
    pb.to_sql(sample_dfs[1],
              index_col_name=pk,
              table_name='sample',
              con=mem_con,
              how='append')


def test_append_new_cols_new_index(mem_con, sample_dfs):
    pk = 'intervalstarttime_gmt'
    pb.to_sql(sample_dfs[2],
              index_col_name=pk,
              table_name='sample',
              con=mem_con,
              how='append')


def test_upsert_fails_different_index(mem_con, sample_dfs):
    with pytest.raises(ValueError):
        pb.to_sql(sample_dfs[0],
                  index_col_name='INTERVALENDTIME_GMT',
                  table_name='sample',
                  con=mem_con,
                  how='upsert')


# READ FROM SQL TESTS #


def test_read_from_full_table(mem_con, sample_dfs):
    db = pb.read_sql('sample', mem_con)
    print(db.head())
    print(db.info())
    csv = sum([len(df) for df in sample_dfs])
    assert len(db) == csv
