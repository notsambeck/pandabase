"""
Tests for pandabase.

Note that the working directory is project root; when running tests from PyCharm,
the default may be root/tests. Can be set from Run/Debug Configurations:Templates
"""
import pytest

import pandabase as pb
from pandabase.helpers import *
from pandabase.companda import companda

import pytz
UTC = pytz.utc


@pytest.mark.parametrize('df, how', [
    (pd.DataFrame(index=[2, 2], data=['x', 'y']), 'create_only'),
    (pd.DataFrame(index=[1, None], columns=['a'], data=['x', 'y']), 'create_only'),
    (pd.Series([1, 2, 3]), 'create_only'),
    (pd.DataFrame(index=[1, 2, 3], columns=['a'], data=['x', 'y', 'a']), 'fake_mode'),
])
def test_invalid_inputs(df, how, empty_db):
    """all these inputs are invalid"""
    with pytest.raises(ValueError):
        pb.to_sql(df, table_name='blank', con=empty_db, how=how)


def test_get_sql_dtype_from_db(simple_df, empty_db):
    """test that datatype extraction functions work in pandas and SQL"""
    df = simple_df
    table = pb.to_sql(df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    for col in table.columns:
        if col.primary_key:
            # different syntax for index
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


def test_read_pandas_table_pandas(pandabase_loaded_db, simple_df, constants):
    """baseline: read pre-written table containing simple_df, using pd.read_sql_table"""
    assert has_table(pandabase_loaded_db, constants.TABLE_NAME)

    loaded_df = pd.read_sql_table(constants.TABLE_NAME,
                                  con=pandabase_loaded_db,
                                  index_col=constants.SAMPLE_INDEX_NAME,
                                  parse_dates='dates')

    # sqlite does not store TZ info. So we will convert
    loaded_df['date'] = pd.to_datetime(loaded_df['date'], utc=True)

    orig_columns = make_clean_columns_dict(simple_df)
    loaded_columns = make_clean_columns_dict(loaded_df)
    for key in orig_columns.keys():
        print(key)
        if key == 'nan':
            # column of all NaN values is skipped
            continue
        assert orig_columns[key] == loaded_columns[key]
    assert companda(loaded_df, simple_df)


def test_read_pandas_table(pandas_loaded_db, simple_df, constants):
    """using pandabase.read_sql:
    read pandas-written table containing simple_df,

    this test fails because: when pandas writes the entry, it does not create
    an explicit primary key. the table is treated as a multiindex"""
    assert has_table(pandas_loaded_db, constants.TABLE_NAME)

    df = pb.read_sql(constants.TABLE_NAME, pandas_loaded_db)

    # line up pk since Pandas doesn't deal with it well
    simple_df[simple_df.index.name] = simple_df.index
    simple_df.index.name = None
    orig_columns = make_clean_columns_dict(simple_df)

    loaded_columns = make_clean_columns_dict(df)
    for key in orig_columns.keys():
        print(key)
        if key == 'nan':
            continue
        assert orig_columns[key] == loaded_columns[key]
    assert companda(df, simple_df)


def test_create_table_no_index_load_pandas(empty_db, minimal_df):
    """add a new minimal table, read with Pandas"""
    table = pb.to_sql(minimal_df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    # print(table.columns)
    assert table.columns[PANDABASE_DEFAULT_INDEX].primary_key
    assert pb.has_table(empty_db, 'sample')

    loaded = pd.read_sql_table('sample', con=empty_db, index_col=PANDABASE_DEFAULT_INDEX)
    # pandas doesn't know about default index
    loaded.index.name = None
    # pandas doesn't know stored as UTC w/o timezone info
    loaded.date = pd.to_datetime(loaded.date, utc=True)

    assert pb.companda(loaded, minimal_df)


def test_create_read_table_no_index(empty_db, minimal_df):
    """add a new minimal table & read it back with pandabase"""
    table = pb.to_sql(minimal_df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    # print(table.columns)
    assert table.columns[PANDABASE_DEFAULT_INDEX].primary_key
    loaded = pb.read_sql('sample', con=empty_db)

    assert pb.has_table(empty_db, 'sample')
    assert pb.companda(loaded, minimal_df)


def test_create_read_table_index(session_db, simple_df, constants):
    """add a new table with explicit index, read it back with pandabase, check equality"""
    table = pb.to_sql(simple_df,
                      table_name='sample',
                      con=session_db,
                      how='create_only')

    # print(table.columns)
    assert table.columns[constants.SAMPLE_INDEX_NAME].primary_key
    assert pb.has_table(session_db, 'sample')

    loaded = pb.read_sql('sample', con=session_db)
    assert pb.companda(loaded, simple_df, ignore_all_nan_columns=True)


@pytest.mark.parametrize('how', ['create_only', 'append'])
def test_overwrite_table_fails(pandabase_loaded_db, simple_df, how, constants):
    """Try to append/insert rows with conflicting indices"""
    table_name = constants.TABLE_NAME
    assert pb.has_table(pandabase_loaded_db, table_name)

    with pytest.raises(Exception):
        pb.to_sql(simple_df,
                  table_name=table_name,
                  con=pandabase_loaded_db,
                  how=how)


@pytest.mark.parametrize('table_name, index_col_name', [['integer_index_table', 'integer'],
                                                        ['float_index_table', 'float'],
                                                        ['datetime_index_table', 'date'], ])
def test_create_table_with_different_index_pandas(session_db, simple_df, table_name, index_col_name):
    """create new tables in empty db, using different col types as index, read with Pandas"""
    df = simple_df.copy()
    df.index = df[index_col_name]
    df = df.drop(index_col_name, axis=1)

    table = pb.to_sql(df,
                      table_name=table_name,
                      con=session_db,
                      how='create_only')

    assert table.columns[index_col_name].primary_key
    assert pb.has_table(session_db, table_name)

    # read with PANDAS
    loaded = pd.read_sql_table(table_name, con=session_db, index_col=index_col_name)

    # make an integer index, since pd.read_sql_table doesn't know to do this
    new_index = loaded.index.name
    loaded[new_index] = loaded.index
    if isinstance(loaded[new_index].iloc[0], str):
        print('converting')
        loaded[new_index] = loaded[new_index].apply(lambda x: float(x))
    loaded.index = loaded[new_index]
    loaded = loaded.drop(new_index, axis=1)

    # pandas doesn't know about UTC
    if 'date' in loaded.columns:
        print('converting date to UTC')
        loaded.date = pd.to_datetime(loaded.date, utc=True)
    else:
        print('making new UTC index (Fake!)')
        loaded.index = df.index

    c = pb.companda(loaded, df, ignore_all_nan_columns=True)
    if not c:
        raise ValueError(c.msg)


@pytest.mark.parametrize('table_name, index_col_name', [['integer_index_table1', 'integer'],
                                                        ['float_index_table1', 'float'],
                                                        ['datetime_index_table1', 'date'], ])
def test_create_read_table_with_different_index(session_db, simple_df, table_name, index_col_name):
    """create new tables in empty db, using different col types as index, read with pandabase"""
    orig_df = simple_df.copy()
    orig_df.index = orig_df[index_col_name]
    print(orig_df[index_col_name])
    print(orig_df.index)
    orig_df = orig_df.drop(index_col_name, axis=1)

    table = pb.to_sql(orig_df,
                      table_name=table_name,
                      con=session_db,
                      how='create_only')

    assert table.columns[index_col_name].primary_key
    assert pb.has_table(session_db, table_name)

    loaded = pb.read_sql(table_name, con=session_db)
    c = pb.companda(loaded, orig_df, ignore_all_nan_columns=True)
    if not c:
        raise ValueError(c.msg)


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_new_rows(pandabase_loaded_db, simple_df, how, constants):
    """upsert or append new complete rows"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = simple_df.copy()
    df.index = df.index + 100

    pb.to_sql(df,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how=how)

    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    # print('loaded post-upsert by pandabase:')
    # print(loaded)

    assert loaded.isna().sum().sum() == 0
    assert companda(simple_df, loaded.loc[simple_df.index])
    assert companda(df, loaded.loc[df.index])


def test_upsert_complete_rows(pandabase_loaded_db, constants):
    """upsert, changing individual values"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)
    df = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    assert df.date.dt.tz == UTC

    df.loc[778, 'float'] = 9.9
    df.loc[779, 'integer'] = 999
    df.loc[780, 'string'] = 'nah'
    df.loc[781, 'date'] = pd.to_datetime('1968-01-01', utc=True)

    # check that all values still exist
    assert df.loc[1, 'integer'] == 778
    assert df.date.dt.tz == UTC

    pb.to_sql(df,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how='upsert')

    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    assert companda(df, loaded)


def test_upsert_incomplete_rows(pandabase_loaded_db, constants):
    """upsert new rows with only 1 of 5 values (and index)"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)
    df = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)

    df.loc[11, 'float'] = 9.9
    df.loc[12, 'integer'] = 999
    df.loc[13, 'string'] = 'nah'
    df.loc[14, 'date'] = pd.to_datetime('1968-01-01', utc=True)

    # check that these values exist
    assert df.loc[1, 'integer'] == 778
    assert pd.isna(df.loc[11, 'integer'])
    assert df.loc[13, 'string'] == 'nah'

    pb.to_sql(df,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how='upsert')

    # check against pandabase read
    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    assert companda(df, loaded)


def test_upsert_coerce_float(pandabase_loaded_db, constants):
    """insert an integer into float column"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['float'], data=[[1.0]])
    df.index.name = constants.SAMPLE_INDEX_NAME
    types = df.dtypes

    pb.to_sql(df,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how='upsert')

    for col in df.columns:
        print(col)
        assert types[col] == df.dtypes[col]

    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    assert loaded.loc[1, 'float'] == 1.0


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_coerce_integer(pandabase_loaded_db, how, constants):
    """insert an integer into float column"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['integer'], data=[[77.0]])
    df.index.name = constants.SAMPLE_INDEX_NAME
    types = df.dtypes

    pb.to_sql(df,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how='upsert')

    for col in df.columns:
        print(col)
        assert types[col] == df.dtypes[col]

    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    assert loaded.loc[1, 'integer'] == 77


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_new_column_fails(pandabase_loaded_db, how, constants):
    """insert into a new column"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[101], columns=['new_column'], data=[[1.1]])
    df.index.name = constants.SAMPLE_INDEX_NAME
    assert df.loc[101, 'new_column'] == 1.1

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how,
                  strict=False)


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_fails_wrong_index_name(pandabase_loaded_db, how, constants):
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['date'], data=[['x']])
    df.index_name = 'not_a_real_name'

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how)


@pytest.mark.parametrize('how', ['upsert', 'append'])
def test_upsert_fails_invalid_float(pandabase_loaded_db, how, constants):
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['float'], data=[['x']])

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how)


@pytest.mark.parametrize('how', ['upsert', 'append'])
def test_upsert_fails_invalid_bool(pandabase_loaded_db, how, constants):
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['bool'], data=[['x']])

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how)


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_fails_invalid_date(pandabase_loaded_db, how, constants):
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['date'], data=[['x']])

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how)
