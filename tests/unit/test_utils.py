"""Unit tests for utility functions."""

import base64
import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from illustrator.utils import (
    count_tokens_estimate,
    create_prompt_variations,
    format_file_size,
    get_output_directory,
    load_config,
    save_config,
    save_image_from_base64,
    split_model_and_provider,
    truncate_text,
    validate_api_keys,
)


class TestModelAndProviderSplitting:
    """Test model and provider splitting functionality."""

    def test_split_model_with_provider(self):
        """Test splitting model string with provider."""
        result = split_model_and_provider("openai/gpt-4")
        assert result == {"provider": "openai", "model": "gpt-4"}

        result = split_model_and_provider("anthropic/claude-3-sonnet")
        assert result == {"provider": "anthropic", "model": "claude-3-sonnet"}

    def test_split_model_without_provider(self):
        """Test splitting model string without provider."""
        result = split_model_and_provider("gpt-4")
        assert result == {"model": "gpt-4"}

        result = split_model_and_provider("claude-3-sonnet")
        assert result == {"model": "claude-3-sonnet"}

    def test_split_model_empty_string(self):
        """Test splitting empty model string."""
        result = split_model_and_provider("")
        assert result == {"model": ""}


class TestImageSaving:
    """Test image saving functionality."""

    def test_save_valid_image(self):
        """Test saving a valid base64 encoded image."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_image.png"
            result = save_image_from_base64(img_b64, output_path)

            assert result is True
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_save_invalid_base64(self):
        """Test saving invalid base64 data."""
        invalid_b64 = "not_valid_base64_data"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_image.png"
            result = save_image_from_base64(invalid_b64, output_path)

            assert result is False
            assert not output_path.exists()

    def test_save_image_creates_directory(self):
        """Test that saving creates parent directories."""
        # Create a simple test image
        img = Image.new('RGB', (50, 50), color='blue')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "nested" / "test.png"
            result = save_image_from_base64(img_b64, output_path)

            assert result is True
            assert output_path.exists()
            assert output_path.parent.exists()


class TestConfigurationFiles:
    """Test configuration file handling."""

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config_data = {
            "api_key": "test-key",
            "model": "gpt-4",
            "settings": {
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            config_file = tmp_file.name

        try:
            # Test saving
            result = save_config(config_data, config_file)
            assert result is True

            # Test loading
            loaded_config = load_config(config_file)
            assert loaded_config == config_data
            assert loaded_config["api_key"] == "test-key"
            assert loaded_config["settings"]["temperature"] == 0.7

        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        result = load_config("nonexistent_file.json")
        assert result == {}

    def test_load_invalid_json_config(self):
        """Test loading invalid JSON configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_file.write("invalid json content {")
            config_file = tmp_file.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_config(config_file)
        finally:
            Path(config_file).unlink(missing_ok=True)


class TestAPIKeyValidation:
    """Test API key validation."""

    def test_validate_api_keys_with_mock_env(self):
        """Test API key validation with mocked environment variables."""
        import os
        from unittest.mock import patch

        # Mock environment variables
        mock_env = {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json",
            "GOOGLE_PROJECT_ID": "test-project",
            "HUGGINGFACE_API_KEY": "test-hf-key"
        }

        with patch.dict(os.environ, mock_env, clear=True):
            result = validate_api_keys()
            assert result["openai"] is True
            assert result["anthropic"] is True
            assert result["google"] is True
            assert result["huggingface"] is True

    def test_validate_api_keys_missing(self):
        """Test API key validation with missing keys."""
        import os
        from unittest.mock import patch

        # Mock empty environment
        with patch.dict(os.environ, {}, clear=True):
            result = validate_api_keys()
            assert result["openai"] is False
            assert result["anthropic"] is False
            assert result["google"] is False
            assert result["huggingface"] is False


class TestDirectoryUtilities:
    """Test directory utility functions."""

    def test_get_output_directory(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            import os
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                output_dir = get_output_directory("My Test Novel")
                expected_path = Path("illustrator_output") / "My_Test_Novel"

                assert output_dir == expected_path
                assert output_dir.exists()
                assert output_dir.is_dir()

            finally:
                os.chdir(original_cwd)

    def test_get_output_directory_special_chars(self):
        """Test output directory with special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            import os
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                output_dir = get_output_directory("My Novel! @#$%^&*()")
                # Special characters should be filtered out
                assert "!" not in str(output_dir)
                assert "@" not in str(output_dir)
                assert output_dir.exists()

            finally:
                os.chdir(original_cwd)


class TestTextUtilities:
    """Test text processing utilities."""

    def test_count_tokens_estimate(self):
        """Test token counting estimation."""
        text = "This is a test sentence with multiple words."
        token_count = count_tokens_estimate(text)

        # Rough estimate: ~4 characters per token
        expected = len(text) // 4
        assert abs(token_count - expected) <= 1

    def test_truncate_text_within_limit(self):
        """Test text truncation when within limit."""
        text = "Short text"
        result = truncate_text(text, 50)
        assert result == text

    def test_truncate_text_exceeds_limit(self):
        """Test text truncation when exceeding limit."""
        text = "This is a very long sentence that definitely exceeds the character limit."
        result = truncate_text(text, 30)

        assert len(result) <= 33  # 30 + "..."
        assert result.endswith("...")
        assert result != text

    def test_truncate_text_preserves_words(self):
        """Test that truncation preserves word boundaries when possible."""
        text = "This is a test sentence with many words here."
        result = truncate_text(text, 20)

        # Should cut at word boundary if possible
        if "..." in result:
            truncated_part = result.replace("...", "").strip()
            # Should not end with partial word (unless no space found)
            if " " in truncated_part:
                assert truncated_part.split()[-1] not in text.split()[:-1]


class TestFileUtilities:
    """Test file utility functions."""

    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(100) == "100.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(1024 ** 4) == "1.0 TB"

    def test_format_file_size_fractional(self):
        """Test file size formatting with fractional values."""
        assert format_file_size(1536) == "1.5 KB"  # 1.5 * 1024
        assert format_file_size(1024 * 1024 + 512 * 1024) == "1.5 MB"


class TestPromptUtilities:
    """Test prompt generation utilities."""

    def test_create_prompt_variations(self):
        """Test creating prompt variations."""
        base_prompt = "A beautiful landscape"
        variations = create_prompt_variations(base_prompt, 3)

        assert len(variations) == 3
        assert variations[0] == base_prompt  # Original should be first

        for variation in variations[1:]:
            assert base_prompt in variation
            assert len(variation) > len(base_prompt)

    def test_create_prompt_variations_limit(self):
        """Test prompt variations with limit exceeding available modifiers."""
        base_prompt = "Test prompt"
        variations = create_prompt_variations(base_prompt, 10)

        # Should not exceed the number of available modifier sets + original
        assert len(variations) <= 6  # 5 modifier sets + original
        assert variations[0] == base_prompt

    def test_create_prompt_variations_single(self):
        """Test creating single prompt variation."""
        base_prompt = "Single test prompt"
        variations = create_prompt_variations(base_prompt, 1)

        assert len(variations) == 1
        assert variations[0] == base_prompt