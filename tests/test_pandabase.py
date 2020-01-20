"""
Tests for pandabase.

Note that the working directory is must be project_root_dir; when running tests from (for example) PyCharm,
the default may be project_root_dir/tests. In Pycharm this default can be set from Run/Debug Configurations:Templates

see, e.g.:
https://pythonspot.com/python-database-postgresql/
for postgres configuration commands to run postgres tests (use command 'pytest --run-postgres')
"""
import pytest

import pandabase as pb
from pandabase.helpers import *
from pandabase.companda import companda

import pytz
UTC = pytz.utc
LA_TZ = pytz.timezone('America/Los_Angeles')    # test timezone

bad_df = pd.DataFrame(index=[2, 2], data=['x', 'y'])
bad_df.index.name = 'bad_index'

bad_df2 = pd.DataFrame(index=[1, None], columns=['a'], data=['x', 'y'])
bad_df2.index.name = 'bad_index'


@pytest.mark.parametrize('df, how', [
    (bad_df, 'create_only'),
    (bad_df2, 'create_only'),
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
            assert_sqla_types_equivalent(get_column_dtype(col, 'sqla'), get_column_dtype(df.index, 'sqla'))
            continue
        assert_sqla_types_equivalent(get_column_dtype(col, 'sqla'), get_column_dtype(df[col.name], 'sqla'))


def assert_sqla_types_equivalent(type1, type2):
    """weak equality test"""
    assert str(type1) == str(type2)


def test_get_sql_dtype_from_db_nans(simple_df_with_nans, empty_db):
    """test that datatype extraction functions work as expected"""
    df = simple_df_with_nans
    table = pb.to_sql(df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    for col in table.columns:
        if col.primary_key:
            # different syntax for index
            assert_sqla_types_equivalent(get_column_dtype(col, 'sqla'), get_column_dtype(df.index, 'sqla'))
            continue
        assert_sqla_types_equivalent(get_column_dtype(col, 'sqla'), get_column_dtype(df[col.name], 'sqla'))


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
        assert_sqla_types_equivalent(orig_columns[key], loaded_columns[key])
    assert companda(loaded_df, simple_df)


def test_select_pandas_table(pandas_loaded_db, simple_df, constants):
    """using pandabase.read_sql:
    read pandas-written table containing simple_df,

    this test is weird because: when pandas writes the entry, it does not create
    an explicit primary key. the table is treated as a multiindex. hence code in middle"""
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
        assert_sqla_types_equivalent(orig_columns[key], loaded_columns[key])
    assert companda(df, simple_df)


def test_create_table_no_index_load_pandas(empty_db, minimal_df):
    """add a new minimal table, read with Pandas"""
    table = pb.to_sql(minimal_df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only',
                      auto_index=True,
                      )

    # print(table.columns)
    assert table.columns[PANDABASE_DEFAULT_INDEX].primary_key
    assert pb.has_table(empty_db, 'sample')

    loaded = pd.read_sql_table('sample', con=empty_db, index_col=PANDABASE_DEFAULT_INDEX)
    # pandas doesn't know about default index
    loaded.index.name = None
    # pandas doesn't know stored as UTC w/o timezone info
    loaded.date = pd.to_datetime(loaded.date, utc=True)

    assert pb.companda(loaded, minimal_df, ignore_index=True)


def test_create_read_table_no_index(empty_db, minimal_df):
    """add a new minimal table & read it back with pandabase"""
    table = pb.to_sql(minimal_df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only',
                      auto_index=True,
                      )

    # print(table.columns)
    assert table.columns[PANDABASE_DEFAULT_INDEX].primary_key
    loaded = pb.read_sql('sample', con=empty_db)

    assert pb.has_table(empty_db, 'sample')
    assert pb.companda(loaded, minimal_df, ignore_index=True)


def test_create_table_multi_index(empty_db, multi_index_df):
    """add a new minimal table & read it back with pandabase"""
    table = pb.to_sql(multi_index_df,
                      table_name='sample_mi',
                      con=empty_db,
                      how='create_only',
                      )

    # print(table.columns)
    assert table.columns['this'].primary_key
    assert table.columns['that'].primary_key

    loaded = pb.read_sql(con=empty_db, table_name='sample_mi')
    print('\n', loaded)

    assert companda(multi_index_df, loaded)


@pytest.mark.parametrize('how, qty', [('create_only', 3000),
                                      ('upsert', 1000)])
def test_write_time(empty_db, how, qty):
    """test that write times are semi-acceptably fast"""
    start = pd.datetime.utcnow()
    pb.to_sql(pd.DataFrame(index=range(qty), columns=['a', 'b', 'c'], data=pd.np.random.random((qty, 3))),
              table_name='sample',
              con=empty_db,
              how=how,
              auto_index=True,
              )
    end = pd.datetime.utcnow()
    assert end-start < pd.Timedelta(seconds=2)


def test_create_select_table_index(session_db, simple_df, constants):
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


def test_create_select_table_range_int_index(empty_db, simple_df, constants):
    """add a new table with explicit index, read it back with pandabase, check equality"""
    table = pb.to_sql(simple_df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    # print(table.columns)
    assert table.columns[constants.SAMPLE_INDEX_NAME].primary_key
    assert pb.has_table(empty_db, 'sample')

    loaded0 = pb.read_sql('sample', con=empty_db, lowest=1, highest=0)
    print(loaded0)
    assert len(loaded0) == 0

    loaded = pb.read_sql('sample', con=empty_db,
                         lowest=simple_df.index[0],
                         highest=simple_df.index[-1])
    assert pb.companda(loaded, simple_df, ignore_all_nan_columns=True)


def test_create_select_table_range_datetime_index(empty_db, simple_df, constants):
    """add a new table with explicit index, read it back with pandabase, check equality"""
    simple_df.index = simple_df.date
    simple_df = simple_df.drop('date', axis=1)

    table = pb.to_sql(simple_df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    # print(table.columns)
    assert table.columns['date'].primary_key
    assert pb.has_table(empty_db, 'sample')

    loaded0 = pb.read_sql('sample', con=empty_db,
                          lowest=simple_df.index[-1],
                          highest=simple_df.index[0])
    print(loaded0)
    assert len(loaded0) == 0

    loaded = pb.read_sql('sample', con=empty_db,
                         lowest=simple_df.index[0],
                         highest=simple_df.index[-1])
    assert pb.companda(loaded, simple_df, ignore_all_nan_columns=True)


def test_select_table_range_fails_different_index(empty_db, simple_df, constants):
    """add a new table with explicit index, read it back with pandabase, check equality"""
    simple_df.index = simple_df.date
    simple_df = simple_df.drop('date', axis=1)

    table = pb.to_sql(simple_df,
                      table_name='sample',
                      con=empty_db,
                      how='create_only')

    with pytest.raises(Exception):
        _ = pb.read_sql('sample', con=empty_db,
                        lowest=0,
                        highest=12)


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


@pytest.mark.parametrize('unique_index_name', [True, False])
def test_append_bad_pk_fails(pandabase_loaded_db, simple_df, constants, unique_index_name):
    """Try to append rows with conflicting index columns"""
    table_name = constants.TABLE_NAME
    assert pb.has_table(pandabase_loaded_db, table_name)

    simple_df.index = simple_df['integer']
    if unique_index_name:
        simple_df[constants.SAMPLE_INDEX_NAME] = simple_df.integer
        simple_df = simple_df.drop('integer', axis=1)

    with pytest.raises(NameError):
        pb.to_sql(simple_df,
                  table_name=table_name,
                  con=pandabase_loaded_db,
                  how='append')


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
        raise ValueError(c.message)


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
        raise ValueError(c.message)


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


def test_upsert_individual_values1(pandabase_loaded_db, constants):
    """upsert to update rows with only 1 of 5 values (and index) from full dataframe"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    df2 = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df2.columns:
        df2[col] = df2[col].astype(df[col].dtype)

    df2.loc[df2.index[0], 'float'] = 9.9
    df2.loc[df2.index[1], 'integer'] = 999
    df2.loc[df2.index[2], 'string'] = 'nah'
    df2.loc[df2.index[3], 'date'] = pd.to_datetime('1968-01-01', utc=True)

    pb.to_sql(df2,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how='upsert')

    # check against pandabase read
    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)

    df.loc[df.index[0], 'float'] = 9.9
    df.loc[df.index[1], 'integer'] = 999
    df.loc[df.index[2], 'string'] = 'nah'
    df.loc[df.index[3], 'date'] = pd.to_datetime('1968-01-01', utc=True)

    assert companda(df, loaded)


def test_upsert_individual_values2(pandabase_loaded_db, constants):
    """upsert to update rows with only 1 of 5 values (and index) from incomplete DataFrame"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    df2 = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df2.columns:
        df2[col] = df2[col].astype(df[col].dtype)

    df2.loc[df2.index[0], 'float'] = 9.9
    df2.loc[df2.index[3], 'date'] = pd.to_datetime('1968-01-01', utc=True)

    pb.to_sql(pd.DataFrame(index=df2.index[:1], columns=['float'], data=[9.9]),
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how='upsert')
    pb.to_sql(pd.DataFrame(index=df2.index[3:4], columns=['date'], data=[pd.to_datetime('1968-01-01', utc=True)]),
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how='upsert')

    # check against pandabase read
    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)

    df.loc[df.index[0], 'float'] = 9.9
    df.loc[df.index[3], 'date'] = pd.to_datetime('1968-01-01', utc=True)

    assert companda(df, loaded)


@pytest.mark.parametrize('col_to_duplicate', ['integer', 'float', 'date', 'string'])
def test_upsert_new_cols(pandabase_loaded_db, constants, col_to_duplicate):
    """upsert new rows with only 1 of 5 values (and index)"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)
    df = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    df['bonus_col'] = df[col_to_duplicate].copy()

    pb.to_sql(df,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how='upsert',
              add_new_columns=True)

    # check against pandabase read
    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    assert companda(df, loaded)
    assert 'bonus_col' in df.columns


def test_upsert_coerce_float(pandabase_loaded_db, constants):
    """insert an integer into float column"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['float'], data=[[2]])
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
    assert loaded.loc[1, 'float'] == 2.0
    assert isinstance(loaded.loc[1, 'float'], float)


def test_coerce_integer(pandabase_loaded_db, constants):
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
        assert types[col] == df.dtypes[col]

    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    assert loaded.loc[1, 'integer'] == 77


def test_coerce_bool(pandabase_loaded_db, constants):
    """insert a bool into float column"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['float'], data=[[True]])
    df.index.name = constants.SAMPLE_INDEX_NAME
    types = df.dtypes

    pb.to_sql(df,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how='upsert')

    for col in df.columns:
        assert types[col] == df.dtypes[col]

    loaded = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)
    assert loaded.loc[1, 'float'] == 1


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_new_column_fails(pandabase_loaded_db, how, constants):
    """insert into a new column"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[101], columns=['new_column'], data=[[1.1]])
    df.index.name = constants.SAMPLE_INDEX_NAME
    assert df.loc[101, 'new_column'] == 1.1

    with pytest.raises(NameError):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how, )


def test_new_column_all_nan(pandabase_loaded_db, df_with_all_nan_col, constants):
    """insert into a new column"""
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df_with_all_nan_col.index = range(100,100+len(df_with_all_nan_col))
    df_with_all_nan_col.index.name = constants.SAMPLE_INDEX_NAME

    pb.to_sql(df_with_all_nan_col,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              auto_index=False,
              how='append', )


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_fails_wrong_index_name(pandabase_loaded_db, how, constants):
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['date'], data=[['x']])
    df.index.name = 'not_a_real_name'

    with pytest.raises(NameError):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how)


@pytest.mark.parametrize('how', ['upsert', 'append'])
def test_upsert_fails_invalid_float(pandabase_loaded_db, how, constants):
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['float'], data=[['x']])
    df.index.name = constants.SAMPLE_INDEX_NAME

    with pytest.raises(TypeError):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how)


def test_auto_index_add_valid_bool(minimal_df, empty_db, constants):
    pb.to_sql(minimal_df,
              table_name=constants.TABLE_NAME,
              con=empty_db,
              how='create_only',
              auto_index=True, )
    assert pb.has_table(empty_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[101, 102, 103],
                      columns=['boolean'],
                      data=[True, False, None])

    pb.to_sql(df,
              table_name=constants.TABLE_NAME,
              con=empty_db,
              how='append',
              auto_index=True, )

    df = pb.read_sql(constants.TABLE_NAME, con=empty_db)

    # Int64Dtype is a fine way to store nullable boolean values
    # Stored in database as boolean or NULL so the data can only be 0, 1, or None
    assert is_bool_dtype(df.boolean) or is_integer_dtype(df.boolean)

    # assume values were loaded in order:
    x = len(df)
    assert df.loc[x - 2, 'boolean']
    assert not df.loc[x - 1, 'boolean']
    assert pd.np.isnan(df.loc[x, 'boolean'])
    with pytest.raises(KeyError):
        _ = df.loc[x + 1, 'boolean']


@pytest.mark.parametrize('how', ['upsert', 'append'])
def test_upsert_valid_bool(pandabase_loaded_db, how, constants):
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[101, 102, 103],
                      columns=['boolean'],
                      data=[True, False, None])
    df.index.name = constants.SAMPLE_INDEX_NAME

    pb.to_sql(df,
              table_name=constants.TABLE_NAME,
              con=pandabase_loaded_db,
              how=how)

    df = pb.read_sql(constants.TABLE_NAME, con=pandabase_loaded_db)

    # Int64Dtype is a fine way to store nullable boolean values
    # Stored in database as boolean or NULL so the data can only be 0, 1, or None
    assert is_bool_dtype(df.boolean) or is_integer_dtype(df.boolean)
    assert df.loc[101, 'boolean']
    assert not df.loc[102, 'boolean']
    assert pd.np.isnan(df.loc[103, 'boolean'])
    with pytest.raises(KeyError):
        _ = df.loc[104, 'boolean']


@pytest.mark.parametrize('how', ['append', 'upsert'])
def test_add_fails_invalid_date(pandabase_loaded_db, how, constants):
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=[1], columns=['date'], data=[['x']])
    df.index.name = constants.SAMPLE_INDEX_NAME

    with pytest.raises((ValueError, TypeError, sqa.exc.StatementError)):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how)


@pytest.mark.parametrize('how, tz',
                         [['append', None],
                          ['upsert', None],
                          ['append', LA_TZ],
                          ['upsert', LA_TZ], ]
                         )
def test_add_fails_invalid_timezone(pandabase_loaded_db, how, constants, tz):
    assert pb.has_table(pandabase_loaded_db, constants.TABLE_NAME)

    df = pd.DataFrame(index=range(5),
                      columns=['date'],
                      data=pd.date_range('2019-06-06', periods=5, freq='h', tz=tz))
    df.index.name = constants.SAMPLE_INDEX_NAME

    print(df.date)

    with pytest.raises(ValueError):
        pb.to_sql(df,
                  table_name=constants.TABLE_NAME,
                  con=pandabase_loaded_db,
                  how=how)


def test_append_auto_index(empty_db, minimal_df):
    """add a new minimal table; add it again"""
    pb.to_sql(minimal_df,
              table_name='sample',
              con=empty_db,
              auto_index=True,
              how='create_only')
    table2 = pb.to_sql(minimal_df,
                       table_name='sample',
                       con=empty_db,
                       auto_index=True,
                       how='append')

    assert table2.columns[PANDABASE_DEFAULT_INDEX].primary_key
    loaded = pb.read_sql('sample', con=empty_db)

    assert pb.has_table(empty_db, 'sample')
    double_df = pd.concat([minimal_df, minimal_df], ignore_index=True)
    assert pb.companda(loaded, double_df, ignore_index=True)
    assert len(loaded) == len(minimal_df) * 2


def test_upsert_auto_index_fails(empty_db, minimal_df):
    """add a new minimal table w/o index; trying to add again should fail"""
    pb.to_sql(minimal_df,
              table_name='sample',
              con=empty_db,
              auto_index=True,
              how='create_only')
    with pytest.raises(IOError):
        pb.to_sql(minimal_df,
                  table_name='sample',
                  con=empty_db,
                  auto_index=True,
                  how='upsert')


@pytest.mark.parametrize('actually_do', [True, False])
def test_add_column_to_database(pandabase_loaded_db, actually_do, constants):
    """possibly add new column to db"""
    name = 'a_new_column'
    col = sqa.Column(name, primary_key=False, type_=Integer, nullable=True)
    if actually_do:
        pb.add_columns_to_db(col, table_name=constants.TABLE_NAME, con=pandabase_loaded_db)
    df = pb.read_sql(table_name=constants.TABLE_NAME, con=pandabase_loaded_db)

    if actually_do:
        assert name in df.columns
        assert is_integer_dtype(df[name])
    else:
        assert name not in df.columns


def test_drop_table(pandabase_loaded_db):
    names = pb.get_db_table_names(pandabase_loaded_db)
    for name in names:
        assert pb.has_table(pandabase_loaded_db, table_name=name)
        pb.drop_db_table(con=pandabase_loaded_db, table_name=name)
        assert not pb.has_table(pandabase_loaded_db, table_name=name)


def test_get_tables(pandabase_loaded_db, constants):
    names = pb.get_db_table_names(pandabase_loaded_db)
    assert len(names) == 1
    assert names[0] == constants.TABLE_NAME


def test_get_columns(pandabase_loaded_db, simple_df, constants):
    cols = pb.get_table_column_names(pandabase_loaded_db, constants.TABLE_NAME)
    assert len(cols) == len(simple_df.columns) + 1
    assert constants.SAMPLE_INDEX_NAME in [col.name for col in cols]


def test_describe_db(pandabase_loaded_db, constants):
    desc = pb.describe_database(pandabase_loaded_db)
    assert len(desc) == 1                        # 1 table in sample db
    assert desc[constants.TABLE_NAME]['min'] == 0    # min
    assert desc[constants.TABLE_NAME]['max'] == 5    # max
    assert desc[constants.TABLE_NAME]['count'] == 6    # count
