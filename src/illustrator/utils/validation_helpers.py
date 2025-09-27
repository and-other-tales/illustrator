"""Helper utilities for validating and fixing model data."""

import uuid
import re
import json
from typing import Dict, Any, List


def ensure_chapter_required_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures that all required fields for chapters are present in the data.
    If fields are missing, it will add them with default values.
    
    Args:
        data: Manuscript data dictionary
        
    Returns:
        Updated manuscript data with all required fields
    """
    if 'chapters' not in data:
        data['chapters'] = []
        return data
    
    for chapter in data['chapters']:
        if 'id' not in chapter:
            # Generate a deterministic ID based on title and number
            title = chapter.get('title', '')
            number = chapter.get('number', 0)
            chapter['id'] = f"ch-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{title}-{number}')}"
        
        if 'summary' not in chapter:
            chapter['summary'] = f"Summary for {chapter.get('title', 'Untitled Chapter')}"
    
    return data


def validate_manuscript_before_save(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and fixes manuscript data before saving to ensure all required fields
    are present and properly formatted.
    
    Args:
        data: Manuscript data to validate
        
    Returns:
        Validated and fixed manuscript data
    """
    # Ensure metadata exists
    if 'metadata' not in data:
        data['metadata'] = {}
    
    metadata = data['metadata']
    
    # Validate metadata fields
    if 'title' not in metadata:
        metadata['title'] = "Untitled Manuscript"
    
    if 'total_chapters' not in metadata:
        metadata['total_chapters'] = len(data.get('chapters', []))
    
    if 'created_at' not in metadata:
        from datetime import datetime
        metadata['created_at'] = datetime.now().isoformat()
    
    # Validate chapters
    data = ensure_chapter_required_fields(data)
    
    return data