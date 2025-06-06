"""
Example unit test to verify directory structure.
"""
import pytest


@pytest.mark.unit
def test_unit_directory_works():
    """Verify unit test directory is properly set up."""
    assert True


@pytest.mark.unit
def test_basic_math():
    """Basic test for unit testing setup."""
    assert 2 + 2 == 4
    assert 3 * 3 == 9