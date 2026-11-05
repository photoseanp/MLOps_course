import pytest
from hypothesis import given
from hypothesis import strategies as st

from src import add


@pytest.mark.unit
def test_add() -> None:
    assert add(2, 3) == 5


@pytest.mark.property
@given(a=st.integers(), b=st.integers())
def test_add_is_commutative(a: int, b: int) -> None:
    assert add(a, b) == add(b, a)
