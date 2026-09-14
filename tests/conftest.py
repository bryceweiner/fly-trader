"""Every test runs against fly_trader_test. DATABASE_URL is forced before any fly_trader import so
the refuse-production guard in fly_trader.db.connection can never see the real database."""
import os

os.environ["DATABASE_URL"] = "postgresql:///fly_trader_test"

import pytest  # noqa: E402

TEST_URL = "postgresql:///fly_trader_test"


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    from fly_trader.db import schema
    schema.ensure_database(TEST_URL)
    schema.apply_schema(TEST_URL)
    yield


@pytest.fixture
def db_conn():
    from fly_trader.db.connection import connect
    conn = connect(TEST_URL)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
