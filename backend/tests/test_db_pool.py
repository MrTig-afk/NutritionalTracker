"""get_db after a Neon cold start: a dead pooled connection is dropped on its own, the pool is never replaced.

Run:  venv/Scripts/python -m unittest backend.tests.test_db_pool -v
No database: a stand-in pool hands out stand-in connections.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import main  # noqa: E402


class Conn:
    def __init__(self, dead=False):
        self.dead, self.autocommit = dead, True

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        if self.dead:
            raise main.psycopg2.OperationalError("server closed the connection unexpectedly")

    def close(self):
        pass


class Pool:
    def __init__(self, *conns):
        self.conns, self.put = list(conns), []

    def getconn(self):
        return self.conns.pop(0)

    def putconn(self, conn, close=False):
        self.put.append((conn, close))


class ColdStart(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(main, "_pool", None)
        p.start()
        self.addCleanup(p.stop)

    def test_a_dead_connection_is_dropped_and_the_same_pool_hands_out_a_live_one(self):
        dead, live = Conn(dead=True), Conn()
        main._pool = pool = Pool(dead, live)
        with mock.patch.object(main, "notify_admin") as alert:
            self.assertIs(main.get_db(None), live)
        self.assertEqual(pool.put, [(dead, True)])   # closed and handed back to the pool it came from
        self.assertIs(main._pool, pool)              # never replaced: other threads hold its connections
        self.assertFalse(live.autocommit)
        alert.assert_not_called()

    def test_two_dead_connections_alert_and_raise_without_leaking_either(self):
        a, b = Conn(dead=True), Conn(dead=True)
        main._pool = pool = Pool(a, b)
        with mock.patch.object(main, "notify_admin") as alert, self.assertRaises(main.psycopg2.OperationalError):
            main.get_db(None)
        self.assertEqual(pool.put, [(a, True), (b, True)])
        alert.assert_called_once()

    def test_a_full_pool_is_replaced_without_a_database_down_alert(self):
        full = Pool()
        full.getconn = mock.Mock(side_effect=main.psycopg2.pool.PoolError("connection pool exhausted"))
        live = Conn()
        fresh = Pool(live)
        main._pool = full
        with mock.patch.object(main, "DATABASE_URL", "postgres://stand-in"), \
                mock.patch.object(main.psycopg2.pool, "ThreadedConnectionPool", return_value=fresh), \
                mock.patch.object(main, "notify_admin") as alert:
            self.assertIs(main.get_db(None), live)
        self.assertIs(main._pool, fresh)   # leaked connections went with the old pool; nothing was closed under anyone
        alert.assert_not_called()

    def test_a_connection_from_a_retired_pool_is_closed_on_release(self):
        conn = Conn()
        conn.close = mock.Mock()
        conn.rollback = mock.Mock()
        main._pool = pool = Pool()
        pool.putconn = mock.Mock(side_effect=main.psycopg2.pool.PoolError("trying to put unkeyed connection"))
        main.release_db(conn)   # used to raise and turn a finished request into a 500
        conn.close.assert_called_once()

    def test_release_with_no_pool_closes_instead_of_opening_one(self):
        conn = Conn()
        conn.close = mock.Mock()
        main._pool = None   # retired, and Neon may be unreachable: a release must never connect
        with mock.patch.object(main.psycopg2.pool, "ThreadedConnectionPool") as new_pool:
            main.release_db(conn)
        new_pool.assert_not_called()
        conn.close.assert_called_once()

    def test_a_pool_that_cannot_be_created_alerts(self):
        with mock.patch.object(main, "get_pool", side_effect=main.psycopg2.OperationalError("could not connect")), \
                mock.patch.object(main, "notify_admin") as alert, self.assertRaises(main.psycopg2.OperationalError):
            main.get_db(None)
        alert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
