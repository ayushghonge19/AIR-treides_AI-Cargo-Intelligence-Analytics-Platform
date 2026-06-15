import pandas as pd
import pytest

from backend.ingestion import normalize_headers, REQUIRED_COLUMNS


def test_normalize_airport_name_header():
    df = pd.DataFrame({"Airport Name": ["DEL"], "Weight (Tons)": [10.5]})
    normalized = normalize_headers(df)
    assert "airport_name" in normalized.columns
    assert "weight_tons" in normalized.columns


def test_normalize_headers_strips_and_lowercases():
    df = pd.DataFrame(
        {
            " Month ": ["2024-01-01"],
            "Export/Import": ["Export"],
        }
    )
    normalized = normalize_headers(df)
    assert "month" in normalized.columns
    assert "export_import" in normalized.columns


def test_required_columns_constant():
    assert REQUIRED_COLUMNS == [
        "month",
        "airport_name",
        "airline",
        "commodity_type",
        "export_import",
        "weight_tons",
    ]
