"""Tests for robust LLM JSON parsing helpers."""

import pytest

from illustrator.utils import parse_llm_json, extract_json_from_text


def test_parse_simple_json_object():
    text = '{"a": 1, "b": [2,3]}'
    obj = parse_llm_json(text)
    assert isinstance(obj, dict)
    assert obj["a"] == 1


def test_parse_code_fenced_json():
    text = """
```json
{
  "key": "value",
  "list": [1, 2]
}
```
"""
    obj = parse_llm_json(text)
    assert obj["key"] == "value"
    assert obj["list"] == [1, 2]


def test_extract_first_json_from_mixed_text():
    text = """
Some prose before.
Here is the data:
{
  "x": 10,
  "y": 20
}
Trailing text that should be ignored.
"""
    extracted = extract_json_from_text(text)
    assert extracted.strip().startswith('{')
    obj = parse_llm_json(text)
    assert obj["x"] == 10 and obj["y"] == 20


def test_parse_json_array():
    text = "Prose then array: [1, 2, 3]"
    arr = parse_llm_json(text)
    assert arr == [1, 2, 3]

