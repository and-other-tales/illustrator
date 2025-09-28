"""Utility functions for the illustrator application."""

# Import from validation_helpers
from .validation_helpers import (
    ensure_chapter_required_fields,
    validate_manuscript_before_save,
    parse_llm_json,
    extract_json_from_text
)

# Import directly from the parent module's utils.py file
# We need to use a relative import to avoid circular dependencies
import os
import sys
import importlib.util

# Get the path to the parent module's utils.py file
parent_dir = os.path.dirname(os.path.dirname(__file__))
utils_path = os.path.join(parent_dir, 'utils.py')

# Load the utils.py module directly
spec = importlib.util.spec_from_file_location('utils_direct', utils_path)
utils_direct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_direct)

# Export functions from utils.py
split_model_and_provider = utils_direct.split_model_and_provider
save_image_from_base64 = utils_direct.save_image_from_base64
get_output_directory = utils_direct.get_output_directory
count_tokens_estimate = utils_direct.count_tokens_estimate
truncate_text = utils_direct.truncate_text
validate_api_keys = utils_direct.validate_api_keys
format_file_size = utils_direct.format_file_size
load_config = utils_direct.load_config
save_config = utils_direct.save_config
create_prompt_variations = utils_direct.create_prompt_variations

# Define __all__ to specify what gets exported
__all__ = [
    'ensure_chapter_required_fields',
    'validate_manuscript_before_save',
    'parse_llm_json',
    'extract_json_from_text',
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