"""Tests for LLM JSON parsing utilities."""

import pytest
from src.illustrator.utils import parse_llm_json


def test_parse_llm_json_clean():
    """Test parsing clean JSON."""
    json_str = '{"key": "value", "nested": {"arr": [1, 2, 3]}}'
    result = parse_llm_json(json_str)
    assert result["key"] == "value"
    assert result["nested"]["arr"] == [1, 2, 3]


def test_parse_llm_json_code_fences():
    """Test parsing JSON with code fences."""
    json_str = '```json\n{"key": "value"}\n```'
    result = parse_llm_json(json_str)
    assert result["key"] == "value"


def test_parse_llm_json_embedded():
    """Test extracting JSON from text."""
    json_str = 'Here is the JSON: {"key": "value"} Hope that helps!'
    result = parse_llm_json(json_str)
    assert result["key"] == "value"


def test_parse_llm_json_cleanup():
    """Test cleaning up common JSON errors."""
    # Trailing characters
    json_str = '{"key": "value"}...'
    result = parse_llm_json(json_str)
    assert result["key"] == "value"
    
    # Missing quotes
    json_str = '{"key: "value"}'
    with pytest.raises(ValueError):
        # This should still fail because it's not easily fixable
        parse_llm_json(json_str)


def test_parse_llm_json_no_json():
    """Test handling when no JSON is found."""
    json_str = 'This text has no JSON content.'
    with pytest.raises(ValueError) as excinfo:
        parse_llm_json(json_str)
    assert "No JSON found" in str(excinfo.value)