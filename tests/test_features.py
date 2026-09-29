"""Tests for package-level feature scaffolding."""

from src.features import __doc__ as feature_doc


def test_feature_package_exists() -> None:
    """Feature package should remain available for future engineering modules."""
    assert isinstance(feature_doc, str)
