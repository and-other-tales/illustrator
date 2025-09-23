"""Utility functions for the manuscript illustrator."""

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from PIL import Image


def split_model_and_provider(model: str) -> Dict[str, str]:
    """Split model string into provider and model components."""
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return {"provider": provider, "model": model_name}
    return {"model": model}


def save_image_from_base64(
    image_data: str,
    output_path: Path,
    format: str = "PNG"
) -> bool:
    """Save base64 encoded image data to file."""
    try:
        # Decode base64 data
        image_bytes = base64.b64decode(image_data)

        # Create PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        image.save(output_path, format=format)
        return True

    except Exception:
        return False


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")


def save_config(config: Dict[str, Any], config_file: str) -> bool:
    """Save configuration to JSON file."""
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False


def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are available."""
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "google": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and os.getenv("GOOGLE_PROJECT_ID")),
        "huggingface": bool(os.getenv("HUGGINGFACE_API_KEY")),
    }


def get_output_directory(manuscript_title: str) -> Path:
    """Get the output directory for a manuscript."""
    safe_title = "".join(c for c in manuscript_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title.replace(' ', '_')

    output_dir = Path("illustrator_output") / safe_title
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def count_tokens_estimate(text: str) -> int:
    """Rough estimate of token count for text."""
    # Simple estimation: ~4 characters per token
    return len(text) // 4


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to maximum length, preserving word boundaries."""
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    last_space = truncated.rfind(' ')

    if last_space > max_length * 0.8:  # If we can preserve most of the text
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def format_file_size(bytes_size: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def create_prompt_variations(base_prompt: str, variations: int = 3) -> list[str]:
    """Create variations of a base prompt for diverse image generation."""
    variation_modifiers = [
        ["cinematic", "dramatic lighting", "high contrast"],
        ["soft", "dreamy", "watercolor style"],
        ["bold", "vibrant colors", "dynamic composition"],
        ["moody", "atmospheric", "film noir"],
        ["ethereal", "fantasy art", "magical realism"],
    ]

    prompts = [base_prompt]  # Include original

    for i in range(min(variations - 1, len(variation_modifiers))):
        modifiers = variation_modifiers[i]
        modified_prompt = f"{base_prompt}, {', '.join(modifiers)}"
        prompts.append(modified_prompt)

    return prompts[:variations]


# -----------------------
# LLM JSON parsing helpers
# -----------------------

def extract_json_from_text(text: str) -> str | None:
    """Extract the first JSON object or array from arbitrary LLM text.

    Handles common cases like code fences and leading/trailing prose.
    """
    if not text:
        return None

    s = text.strip()

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

    Returns Python object or raises ValueError on failure.
    """
    candidate = extract_json_from_text(text)
    if candidate is None:
        raise ValueError("No JSON found in LLM output")

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

        # Strategy 2: Fix unterminated strings by finding and closing them
        try:
            # Find the position of the JSON decode error
            lines = candidate.split('\n')
            if hasattr(e, 'lineno') and hasattr(e, 'colno'):
                line_idx = e.lineno - 1
                col_idx = e.colno - 1

                if line_idx < len(lines):
                    line = lines[line_idx]
                    # Check if we have an unterminated string
                    if col_idx < len(line) and '"' in line[:col_idx]:
                        # Count unmatched quotes before the error position
                        quote_count = 0
                        escaped = False
                        for i, char in enumerate(line[:col_idx + 1]):
                            if char == '\\' and not escaped:
                                escaped = True
                            elif char == '"' and not escaped:
                                quote_count += 1
                            else:
                                escaped = False

                        # If we have an odd number of quotes, we have an unterminated string
                        if quote_count % 2 == 1:
                            # Try to close the unterminated string
                            lines[line_idx] = line[:col_idx] + '"' + line[col_idx:]
                            fixed_candidate = '\n'.join(lines)
                            return json.loads(fixed_candidate)
        except (json.JSONDecodeError, AttributeError, IndexError):
            pass

        # Strategy 3: Try to salvage partial JSON by truncating at the error point
        try:
            if hasattr(e, 'pos'):
                truncated = candidate[:e.pos]
                # Try to close any open structures
                open_braces = truncated.count('{') - truncated.count('}')
                open_brackets = truncated.count('[') - truncated.count(']')

                # Add closing brackets/braces as needed
                for _ in range(open_brackets):
                    truncated += ']'
                for _ in range(open_braces):
                    truncated += '}'

                return json.loads(truncated)
        except (json.JSONDecodeError, AttributeError):
            pass

        # If all strategies fail, raise the original error
        raise ValueError(f"Failed to parse JSON after cleanup attempts: {e}")


def enforce_prompt_length(provider: str, prompt: str) -> str:
    """Clamp prompt length per provider to conservative char limits.

    This is a pragmatic safeguard; providers enforce their own limits.
    """
    limits = {
        'dalle': 1800,        # OpenAI Images prompt (approx)
        'imagen4': 2200,      # Vertex Imagen
        'imagen': 2200,
        'flux': 2400,
    }
    max_len = limits.get(provider.lower(), 2000)
    if len(prompt) <= max_len:
        return prompt
    # Trim on word boundary when possible
    cut = prompt[:max_len]
    sp = cut.rfind(' ')
    if sp > max_len * 0.6:
        return cut[:sp] + '...'
    return cut + '...'
