import pytest
import pandabase as pb


@pytest.fixture(scope='session')
def test_con():
    return pb.engine_builder('sqlite:///:memory:')


def test_has_table(test_con):
    assert pb.has_table(test_con, 'rt5') is False
