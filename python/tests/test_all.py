import pytest
import vector_search


def test_sum_as_string():
    assert vector_search.sum_as_string(1, 1) == "2"
