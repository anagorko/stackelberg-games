"""
Unit tests for setting generator functions.
"""

import stackelberg_games.patrolling as sgp


def test_basilico_et_al():
    setting = sgp.basilico_et_al()
    assert setting is not None


def test_john_et_al():
    setting = sgp.john_et_al()
    assert setting is not None


def test_port_gdynia():
    setting = sgp.port_gdynia()
    assert setting is not None
