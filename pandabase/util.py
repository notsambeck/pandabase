from sqlalchemy import Table
from pandabase.helpers import *


def drop_db_table(table_name, con, schema=None):
    """Drop table [table_name] from con (or con.schema, if schema kwarg is supplied)
    utility function to avoid calling SQL/SQLA directly during maintenance, etc."""
    engine = engine_builder(con)
    meta = sqa.MetaData(engine)
    meta.reflect(bind=engine, schema=schema)

    if schema is None:
        table_namespace = table_name
    else:
        table_namespace = f'{schema}.{table_name}'

    t = meta.tables[table_namespace]

    with engine.begin():
        t.drop()


def get_db_table_names(con, schema=None):
    """get a list of table names from con (or con.schema, if schema kwarg is supplied)"""
    meta = sqa.MetaData()
    engine = engine_builder(con)
    meta.reflect(bind=engine, schema=schema)
    return list(meta.tables.keys())


def get_table_column_names(con, table_name, schema=None):
    """get a list of column names from con, table_name (or con.schema if schema kwarg is supplied)"""
    meta = sqa.MetaData()
    engine = engine_builder(con)
    meta.reflect(bind=engine, schema=schema)

    if schema is None:
        table_namespace = table_name
    else:
        table_namespace = f'{schema}.{table_name}'

    return list(meta.tables[table_namespace].columns)


def describe_database(con, schema=None):
    """
    Returns a description of con (or con.schema if schema kwarg is supplied): {table_names: {table_info_dicts}}

    Args:
        con: string URI or db engine
        schema: Specify the schema (if database flavor supports this). If None, use default schema.

    Returns:
        {'table_name_1': {'min': min, 'max': max, 'count': count},
         ... }
    """
    engine = engine_builder(con)
    meta = sqa.MetaData()
    meta.reflect(bind=engine, schema=schema)

    res = {}

    for table_name in meta.tables:
        try:
            table = Table(table_name, meta, autoload=True, autoload_with=engine)
            index = table.primary_key.columns
            if len(index) == 1:
                index = index.items()[0][0]
                minim = engine.execute(sqa.select([sqa.func.min(sqa.text(index))]).select_from(table)).scalar()
                maxim = engine.execute(sqa.select([sqa.func.max(sqa.text(index))]).select_from(table)).scalar()
                count = engine.execute(sqa.select([sqa.func.count()]).select_from(table)).scalar()
                res[table_name] = {'min': minim, 'max': maxim, 'count': count}
            else:
                count = engine.execute(sqa.select([sqa.func.count()]).select_from(table)).scalar()
                res[table_name] = {'index_type': 'multi', 'index_cols': str(index.keys()), 'count': count}

        except Exception as e:
            print(f'failed to describe table: {table_name} due to {e}')

    return res

