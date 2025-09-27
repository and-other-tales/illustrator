"""Test module for validation helpers."""

import pytest
import uuid
from datetime import datetime
from src.illustrator.utils.validation_helpers import ensure_chapter_required_fields, validate_manuscript_before_save


def test_ensure_chapter_required_fields():
    """Test that missing chapter fields are properly filled."""
    data = {
        'chapters': [
            {
                'title': 'Chapter 1',
                'number': 1,
                'content': 'Test content'
            },
            {
                'title': 'Chapter 2',
                'content': 'More test content'
            }
        ]
    }
    
    result = ensure_chapter_required_fields(data)
    
    # Check that all chapters have IDs and summaries
    for chapter in result['chapters']:
        assert 'id' in chapter
        assert 'summary' in chapter
        assert chapter['summary'] == f"Summary for {chapter['title']}"
        assert chapter['id'].startswith('ch-')


def test_ensure_chapter_required_fields_with_empty_chapters():
    """Test handling of empty chapters list."""
    data = {'chapters': []}
    
    result = ensure_chapter_required_fields(data)
    
    assert result['chapters'] == []


def test_ensure_chapter_required_fields_no_chapters():
    """Test handling when chapters key is missing."""
    data = {'metadata': {'title': 'Test'}}
    
    result = ensure_chapter_required_fields(data)
    
    assert 'chapters' in result
    assert result['chapters'] == []


def test_validate_manuscript_before_save():
    """Test validation of entire manuscript before saving."""
    data = {
        'chapters': [
            {
                'title': 'Chapter 1',
                'content': 'Test content'
            }
        ]
    }
    
    result = validate_manuscript_before_save(data)
    
    # Check metadata was added
    assert 'metadata' in result
    assert 'title' in result['metadata']
    assert 'total_chapters' in result['metadata']
    assert result['metadata']['total_chapters'] == 1
    assert 'created_at' in result['metadata']
    
    # Check chapter fields
    assert 'id' in result['chapters'][0]
    assert 'summary' in result['chapters'][0]