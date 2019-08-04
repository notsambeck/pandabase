from pandabase.helpers import *
import pytest


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
