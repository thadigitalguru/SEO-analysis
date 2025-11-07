import pandas as pd
import pytest
from src.utils import ensure_cols


def test_ensure_cols_ok():
    df = pd.DataFrame({"a": [1], "b": [2]})
    ensure_cols(df, ["a", "b"])  # should not raise


def test_ensure_cols_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError) as e:
        ensure_cols(df, ["a", "b"])  # b missing
    assert "Missing columns" in str(e.value)
