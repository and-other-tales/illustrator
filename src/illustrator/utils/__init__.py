"""Utility functions for the illustrator application."""

# Import and re-export from validation_helpers
from .validation_helpers import (
    ensure_chapter_required_fields,
    validate_manuscript_before_save,
    parse_llm_json,
    extract_json_from_text
)

# Import from parent module to avoid circular imports
import sys
import importlib
parent_utils = importlib.import_module('illustrator.utils')
sys.modules[__name__].split_model_and_provider = parent_utils.split_model_and_provider
sys.modules[__name__].save_image_from_base64 = parent_utils.save_image_from_base64
sys.modules[__name__].get_output_directory = parent_utils.get_output_directory
sys.modules[__name__].count_tokens_estimate = parent_utils.count_tokens_estimate
sys.modules[__name__].truncate_text = parent_utils.truncate_text
sys.modules[__name__].validate_api_keys = parent_utils.validate_api_keys
sys.modules[__name__].format_file_size = parent_utils.format_file_size
sys.modules[__name__].load_config = parent_utils.load_config
sys.modules[__name__].save_config = parent_utils.save_config
sys.modules[__name__].create_prompt_variations = parent_utils.create_prompt_variations

# Define which symbols should be exported
__all__ = [
    # From validation_helpers
    'ensure_chapter_required_fields',
    'validate_manuscript_before_save',
    'parse_llm_json',
    'extract_json_from_text',
    # From utils.py
    'split_model_and_provider',
    'save_image_from_base64',
    'get_output_directory',
    'count_tokens_estimate',
    'truncate_text',
    'validate_api_keys',
    'format_file_size',
    'load_config',
    'save_config',
    'create_prompt_variations'
]