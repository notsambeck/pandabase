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
LA_TZ = pytz.timezone('America/Los_Angeles')  # test timezone

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
    assert len(desc) == 1  # 1 table in sample db
    assert desc[constants.TABLE_NAME]['min'] == 0  # min
    assert desc[constants.TABLE_NAME]['max'] == 5  # max
    assert desc[constants.TABLE_NAME]['count'] == 6  # count
