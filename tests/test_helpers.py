from pandabase.helpers import *
import pytest
from pytz import utc


def test_get_sql_dtype_df(df_with_all_nan_col):
    """test that datatype functions work as expected"""
    df = df_with_all_nan_col

    assert isinstance(df.index, pd.RangeIndex)

    assert is_datetime64_any_dtype(df.date)
    assert isinstance(get_column_dtype(df.date, 'sqla'), TIMESTAMP)
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
    [pd.Series([0, 0, 0, ]), False],
    [pd.Series([1, 1, 1, ]), False],
    [pd.Series([None, None, None, ]), None],
    # [pd.Series([np.NaN], dtype=bool), None],   # doesn't work - bool coerces NaN to False
    [pd.Series([np.NaN], dtype=str), None],
    [pd.Series([np.NaN], dtype=float), None],
    [pd.Series([0, 1, pd.to_datetime('2017-01-12')]), False],
    [pd.Series([0, 1, pd.to_datetime('2000')]), False],
    [pd.Series([pd.to_datetime('2000'), pd.to_datetime('2017-01-12')]), False],
    [pd.Series(np.array([True, False, np.NaN, ]), dtype='Int64'), True],  # Int64 is broken
    [pd.Series([np.NaN], dtype=pd.Int64Dtype()), None],  # Int64 is broken
])
def test_series_is_boolean(series, expected):
    assert isinstance(series, pd.Series)
    print(series)
    assert series_is_boolean(series) == expected


@pytest.mark.parametrize('name, cleaned', [['abc', 'abc'],
                                           ['a b c', 'a_b_c'],
                                           ['(a{b}c)', '_a_b_c_'],
                                           ['aaa-', 'aaa_'],
                                           ['weather_33.68_-117.87', 'weather_3368__11787']])
def test_clean_name(name, cleaned):
    assert clean_name(name) == cleaned


def test_make_clean_columns_dict_single_index():
    """test make_clean_columns_dict works for a single index"""
    data = {'full_name': ['John Doe'],
            'number_of_pets': [3],
            'likes_bananas': [True],
            'dob': [pd.Timestamp('1990-01-01', tzinfo=utc)]}
    df = pd.DataFrame(data).rename_axis('id', axis='index')
    cols = make_clean_columns_dict(df)

    res = {
        'id': {'dtype': Integer, 'pk': True},
        'full_name': {'dtype': String, 'pk': False},
        'number_of_pets': {'dtype': Integer, 'pk': False},
        'likes_bananas': {'dtype': Boolean, 'pk': False},
        'dob': {'dtype': TIMESTAMP(timezone=True), 'pk': False}
    }

    assert cols.keys() == res.keys()

    for k in cols.keys():
        if isinstance(cols[k]['dtype'], TIMESTAMP):  # Equality comparison fails for TIMESTAMP == TIMESTAMP
            assert isinstance(res[k]['dtype'], TIMESTAMP)
            assert cols[k]['pk'] == res[k]['pk']
        else:
            assert cols[k] == res[k]


@pytest.mark.parametrize('index_value, index_type', [
    [1, Integer],
    [1.1, Float],
    ['goat', String],
    [pd.Timestamp('1990-01-01', tzinfo=utc), TIMESTAMP],
])
def test_make_clean_columns_dict_multi_index(index_value, index_type):
    """test make_clean_columns_dict works for a multi index"""
    data = {'full_name': ['John Doe'],
            'number_of_pets': [3],
            'likes_bananas': [True],
            'dob': [pd.Timestamp('1990-01-01', tzinfo=utc)]}
    df = pd.DataFrame(data)
    df.index = pd.MultiIndex.from_arrays([[1], [index_value]], names=['category_id', 'member_id'])

    cols = make_clean_columns_dict(df, autoindex=False)

    res = {
        'category_id': {'dtype': Integer, 'pk': True},
        'member_id': {'dtype': index_type, 'pk': True},
        'full_name': {'dtype': String, 'pk': False},
        'number_of_pets': {'dtype': Integer, 'pk': False},
        'likes_bananas': {'dtype': Boolean, 'pk': False},
        'dob': {'dtype': TIMESTAMP(timezone=True), 'pk': False}
    }

    assert cols.keys() == res.keys()

    for k in cols.keys():
        if isinstance(cols[k]['dtype'], TIMESTAMP):  # Equality comparison fails for TIMESTAMP == TIMESTAMP
            # TODO: why can't we check that isinstance(res[k], TIMESTAMP) ???
            assert cols[k]['pk'] == res[k]['pk']
        else:
            assert cols[k] == res[k]


def test_make_clean_cols_from_full_df(multi_index_df):
    df = multi_index_df
    x = make_clean_columns_dict(df, autoindex=False)
    for pk in ['this', 'that']:
        assert x[pk]['pk']
        assert x[pk]['dtype'] in [Float, Integer]
    for col in ['date', 'float', 'integer']:
        assert not x[col]['pk']
