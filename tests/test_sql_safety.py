import pytest

from backend.nodes import is_safe_select


def test_safe_select_query():
    assert is_safe_select("SELECT airport_name, weight_tons FROM air_cargo_data")


def test_rejects_insert():
    assert not is_safe_select("INSERT INTO air_cargo_data VALUES (1)")


def test_rejects_delete():
    assert not is_safe_select("DELETE FROM air_cargo_data")


def test_rejects_drop():
    assert not is_safe_select("DROP TABLE air_cargo_data")


def test_rejects_non_select():
    assert not is_safe_select("UPDATE air_cargo_data SET weight_tons = 0")
