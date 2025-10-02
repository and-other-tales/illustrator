"""Helper utilities for validating and fixing model data."""

import uuid
import re
import json
from typing import Dict, Any, List


def ensure_chapter_required_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures that all required fields for chapters are present in the data.
    If fields are missing or None, it will add them with default values.
    
    Args:
        data: Manuscript data dictionary
        
    Returns:
        Updated manuscript data with all required fields
    """
    if 'chapters' not in data:
        data['chapters'] = []
        return data
    
    for chapter in data['chapters']:
        # Handle missing or None ID
        if 'id' not in chapter or chapter.get('id') is None:
            # Generate a deterministic ID based on title and number
            title = chapter.get('title', '')
            number = chapter.get('number', 0)
            chapter['id'] = f"ch-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{title}-{number}')}"
        
        # Handle missing or None summary
        if 'summary' not in chapter or chapter.get('summary') is None:
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


def extract_json_from_text(s: str) -> str:
    """Extract JSON string from text, handling common LLM output formats."""
    if s is None:
        return None

    # Strip code fences if present
    if s.startswith("```"):
        # remove starting fence with optional language
        s = re.sub(r"^```(json|JSON)?\s*", "", s)
        # remove trailing fence
        s = re.sub(r"\s*```\s*$", "", s)

    # Quick path: already starts with JSON
    if s.startswith("{") or s.startswith("["):
        return s

    # Fallback: search for the first {..} or [..] block
    obj_match = re.search(r"\{[\s\S]*\}", s)
    arr_match = re.search(r"\[[\s\S]*\]", s)

    # Prefer object over array if both found and object starts earlier
    candidates = []
    if obj_match:
        candidates.append((obj_match.start(), obj_match.group(0)))
    if arr_match:
        candidates.append((arr_match.start(), arr_match.group(0)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def parse_llm_json(text: str) -> Any:
    """Parse JSON from LLM output robustly.

    Returns Python object or None if parsing fails.
    """
    candidate = extract_json_from_text(text)
    if candidate is None:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Attempt progressive cleanup strategies

        # Strategy 1: Remove trailing ellipses or stray characters
        try:
            cleaned = candidate.strip().rstrip('.').rstrip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Fix unclosed quotes
        try:
            cleaned = re.sub(r'([{,]\s*"[^"]*?)(\s*[},])', r'\1"\2', candidate)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Fix missing commas
        try:
            cleaned = re.sub(r'"\s*}\s*"', '", "', candidate)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # If all strategies fail, return None
        return None