"""Comprehensive unit tests for the CLI module."""

import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
from uuid import uuid4

from illustrator.cli import ManuscriptCLI
from illustrator.models import Chapter, ChapterAnalysis, ManuscriptMetadata, SavedManuscript, EmotionalMoment, EmotionalTone, ImageProvider


class TestManuscriptCLI:
    """Test cases for the ManuscriptCLI class."""

    @pytest.fixture
    def cli(self):
        """Create a CLI instance for testing."""
        return ManuscriptCLI()

    @pytest.fixture
    def sample_metadata(self):
        """Sample manuscript metadata for testing."""
        return ManuscriptMetadata(
            title="Test Novel",
            author="Test Author",
            genre="Fantasy",
            total_chapters=2,
            created_at=datetime.now().isoformat()
        )

    @pytest.fixture
    def sample_chapter(self):
        """Sample chapter for testing."""
        return Chapter(
            title="Chapter 1",
            content="This is test chapter content with emotional depth.",
            number=1,
            word_count=10
        )

    @pytest.fixture
    def sample_analysis(self, sample_chapter):
        """Sample chapter analysis for testing."""
        from illustrator.models import IllustrationPrompt
        emotional_moment = EmotionalMoment(
            text_excerpt="emotional depth",
            start_position=10,
            end_position=25,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.8,
            context="This is a test chapter with emotional depth."
        )
        illustration_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="A mystical forest scene",
            style_modifiers=["digital painting"],
            technical_params={}
        )
        return ChapterAnalysis(
            chapter=sample_chapter,
            emotional_moments=[emotional_moment],
            dominant_themes=["Growth", "Discovery"],
            setting_description="A mystical forest",
            character_emotions={"protagonist": [EmotionalTone.JOY]},
            illustration_prompts=[illustration_prompt]
        )

    def test_cli_initialization(self, cli):
        """Test CLI initialization."""
        assert cli.context is None
        assert cli.chapters == []
        assert cli.manuscript_metadata is None
        assert cli.completed_analyses == []

    @patch.dict(os.environ, {
        'DEFAULT_IMAGE_PROVIDER': 'dalle',
        'OPENAI_API_KEY': 'test_key',
        'ANTHROPIC_API_KEY': 'test_key'
    })
    def test_setup_environment_dalle(self, cli):
        """Test environment setup for DALL-E provider."""
        cli.setup_environment()
        # Should not raise an exception

    @patch.dict(os.environ, {
        'DEFAULT_IMAGE_PROVIDER': 'imagen4',
        'GOOGLE_APPLICATION_CREDENTIALS': 'test_creds',
        'GOOGLE_PROJECT_ID': 'test_project',
        'ANTHROPIC_API_KEY': 'test_key'
    })
    def test_setup_environment_imagen4(self, cli):
        """Test environment setup for Imagen4 provider."""
        cli.setup_environment()
        # Should not raise an exception

    @patch.dict(os.environ, {
        'DEFAULT_IMAGE_PROVIDER': 'flux',
        'HUGGINGFACE_API_KEY': 'test_key',
        'ANTHROPIC_API_KEY': 'test_key'
    })
    def test_setup_environment_flux(self, cli):
        """Test environment setup for Flux provider."""
        cli.setup_environment()
        # Should not raise an exception

    @patch.dict(os.environ, {'DEFAULT_IMAGE_PROVIDER': 'dalle'}, clear=True)
    def test_setup_environment_missing_openai_key(self, cli):
        """Test environment setup with missing OpenAI key."""
        with patch('sys.exit') as mock_exit, \
             patch('illustrator.cli.console'), \
             patch('illustrator.cli.load_dotenv'), \
             patch('illustrator.cli.find_dotenv'):
            cli.setup_environment()
            mock_exit.assert_called_with(1)

    @patch.dict(os.environ, {'DEFAULT_IMAGE_PROVIDER': 'imagen4'}, clear=True)
    def test_setup_environment_missing_google_creds(self, cli):
        """Test environment setup with missing Google credentials."""
        with patch('sys.exit') as mock_exit, \
             patch('illustrator.cli.console'), \
             patch('illustrator.cli.load_dotenv'), \
             patch('illustrator.cli.find_dotenv'):
            cli.setup_environment()
            mock_exit.assert_called_with(1)

    @patch('illustrator.cli.console')
    def test_display_welcome(self, mock_console, cli):
        """Test welcome message display."""
        cli.display_welcome()
        mock_console.print.assert_called_once()

    @patch('illustrator.cli.Prompt.ask')
    def test_get_manuscript_metadata(self, mock_prompt, cli):
        """Test manuscript metadata collection."""
        mock_prompt.side_effect = ["Test Title", "Test Author", "Fantasy"]

        metadata = cli.get_manuscript_metadata()

        assert metadata.title == "Test Title"
        assert metadata.author == "Test Author"
        assert metadata.genre == "Fantasy"
        assert metadata.total_chapters == 0
        assert metadata.created_at is not None

    def test_load_style_config_success(self, cli, tmp_path):
        """Test successful style config loading."""
        config_data = {
            "style_name": "Test Style",
            "art_style": "watercolor",
            "color_palette": "warm"
        }

        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        with patch('illustrator.cli.console'):
            result = cli.load_style_config(str(config_file))

        assert result["art_style"] == "Test Style"
        assert result["image_provider"] == "dalle"
        assert "style_config" in result

    def test_load_style_config_error(self, cli):
        """Test style config loading with file error."""
        with pytest.raises(Exception):
            cli.load_style_config("nonexistent_file.json")

    @patch('illustrator.cli.Prompt.ask')
    @patch('illustrator.cli.Path.exists')
    def test_get_user_preferences_with_predefined_styles(self, mock_exists, mock_prompt, cli):
        """Test user preferences with predefined styles."""
        mock_exists.return_value = True
        mock_prompt.side_effect = ["1"]  # Select first predefined style

        with patch.object(cli, 'load_style_config') as mock_load:
            mock_load.return_value = {"image_provider": "dalle"}
            result = cli.get_user_preferences()

        mock_load.assert_called_once()

    @patch('illustrator.cli.Prompt.ask')
    @patch('illustrator.cli.Path.exists')
    def test_get_user_preferences_custom_style(self, mock_exists, mock_prompt, cli):
        """Test user preferences with custom style."""
        mock_exists.return_value = False
        mock_prompt.side_effect = ["1", "digital painting", "warm", "Monet"]

        result = cli.get_user_preferences()

        assert result["image_provider"] == "dalle"
        assert result["art_style"] == "digital painting"
        assert result["color_palette"] == "warm"
        assert result["artistic_influences"] == "Monet"

    @patch('illustrator.cli.input')
    @patch('illustrator.cli.Prompt.ask')
    @patch('illustrator.cli.console')
    def test_input_chapter_success(self, mock_console, mock_prompt, mock_input, cli):
        """Test successful chapter input."""
        mock_prompt.return_value = "Test Chapter"
        mock_input.side_effect = ["Line 1", "Line 2", EOFError()]

        chapter = cli.input_chapter(1)

        assert chapter is not None
        assert chapter.title == "Test Chapter"
        assert chapter.content == "Line 1\nLine 2"
        assert chapter.number == 1
        assert chapter.word_count == 4

    @patch('illustrator.cli.input')
    @patch('illustrator.cli.Prompt.ask')
    @patch('illustrator.cli.console')
    def test_input_chapter_empty_content(self, mock_console, mock_prompt, mock_input, cli):
        """Test chapter input with empty content."""
        mock_prompt.return_value = "Test Chapter"
        mock_input.side_effect = EOFError()

        chapter = cli.input_chapter(1)

        assert chapter is None

    @patch('illustrator.cli.Confirm.ask')
    def test_confirm_continue_true(self, mock_confirm, cli):
        """Test continue confirmation returns True."""
        mock_confirm.return_value = True
        assert cli.confirm_continue() is True

    @patch('illustrator.cli.Confirm.ask')
    def test_confirm_continue_false(self, mock_confirm, cli):
        """Test continue confirmation returns False."""
        mock_confirm.return_value = False
        assert cli.confirm_continue() is False

    @pytest.mark.asyncio
    async def test_process_chapters_no_chapters(self, cli):
        """Test processing with no chapters."""
        with patch('illustrator.cli.console') as mock_console:
            await cli.process_chapters({})
            mock_console.print.assert_called_with("[red]No chapters to process![/red]")

    @pytest.mark.asyncio
    @patch('illustrator.graph.graph')
    @patch('langgraph.store.memory.InMemoryStore')
    @patch('illustrator.cli.ManuscriptContext')
    async def test_process_chapters_success(self, mock_context, mock_store, mock_graph, cli, sample_chapter, sample_metadata):
        """Test successful chapter processing."""
        cli.chapters = [sample_chapter]
        cli.manuscript_metadata = sample_metadata

        # Mock the compiled graph
        mock_compiled = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_compiled.ainvoke.return_value = {
            "current_analysis": {"chapter": sample_chapter, "emotional_moments": []},
            "generated_images": []
        }

        style_prefs = {"image_provider": "dalle", "art_style": "digital"}

        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "test-uuid"
            await cli.process_chapters(style_prefs)

    @pytest.mark.asyncio
    async def test_save_generated_images_no_images(self, cli, sample_chapter, sample_metadata):
        """Test image saving with no images."""
        cli.manuscript_metadata = sample_metadata
        await cli._save_generated_images([], sample_chapter, 1)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_save_generated_images_success(self, cli, sample_chapter, sample_metadata, tmp_path):
        """Test successful image saving."""
        cli.manuscript_metadata = sample_metadata

        # Mock generated images with base64 data
        import base64
        test_image_data = base64.b64encode(b"fake image data").decode()
        generated_images = [{
            "image_data": f"data:image/png;base64,{test_image_data}",
            "emotional_moment": "test moment",
            "metadata": {"format": "png"}
        }]

        with patch('illustrator.cli.Path.mkdir'), \
             patch('builtins.open', mock_open()) as mock_file:
            await cli._save_generated_images(generated_images, sample_chapter, 1)

        # Verify files were written
        assert mock_file.call_count >= 2  # Image file + metadata file

    def test_display_results_summary_no_chapters(self, cli):
        """Test results summary with no chapters."""
        cli.display_results_summary()
        # Should complete without error

    def test_display_results_summary_with_data(self, cli, sample_chapter, sample_analysis):
        """Test results summary with chapter data."""
        cli.chapters = [sample_chapter]
        cli.completed_analyses = [sample_analysis]

        with patch('illustrator.cli.console') as mock_console:
            cli.display_results_summary()
            mock_console.print.assert_called()

    def test_save_results_no_data(self, cli):
        """Test save results with no data."""
        cli.save_results()
        # Should complete without error

    def test_save_results_success(self, cli, sample_chapter, sample_metadata, sample_analysis):
        """Test successful results saving."""
        cli.manuscript_metadata = sample_metadata
        cli.chapters = [sample_chapter]
        cli.completed_analyses = [sample_analysis]

        with patch('illustrator.cli.Path.mkdir'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json, \
             patch('illustrator.cli.console'):
            cli.save_results()

        mock_json.assert_called()

    def test_save_manuscript_draft_no_data(self, cli):
        """Test save draft with no data."""
        with pytest.raises(ValueError):
            cli.save_manuscript_draft()

    def test_save_manuscript_draft_success(self, cli, sample_chapter, sample_metadata):
        """Test successful draft saving."""
        cli.manuscript_metadata = sample_metadata
        cli.chapters = [sample_chapter]

        with patch('illustrator.cli.Path.mkdir'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json, \
             patch('illustrator.cli.console'):
            result = cli.save_manuscript_draft("test_name")

        assert "test_name.json" in result
        mock_json.assert_called_once()

    def test_list_saved_manuscripts_no_dir(self, cli):
        """Test listing manuscripts with no directory."""
        with patch('illustrator.cli.Path.exists', return_value=False):
            result = cli.list_saved_manuscripts()
            assert result == []

    def test_list_saved_manuscripts_success(self, cli, tmp_path):
        """Test successful manuscript listing."""
        # Create test manuscript file
        manuscript_data = {
            "metadata": {"title": "Test", "author": "Author", "genre": "Fantasy", "total_chapters": 1, "created_at": "2023-01-01"},
            "chapters": [],
            "saved_at": "2023-01-01T00:00:00",
            "file_path": "test.json"
        }

        test_file = tmp_path / "test.json"
        with open(test_file, 'w') as f:
            json.dump(manuscript_data, f)

        with patch('illustrator.cli.Path') as mock_path:
            mock_saved_dir = MagicMock()
            mock_saved_dir.exists.return_value = True
            mock_saved_dir.glob.return_value = [test_file]
            mock_path.return_value = mock_saved_dir

            with patch('builtins.open', mock_open(read_data=json.dumps(manuscript_data))):
                result = cli.list_saved_manuscripts()

        assert len(result) == 1
        assert result[0].metadata.title == "Test"

    def test_load_manuscript_draft_success(self, cli):
        """Test successful manuscript loading."""
        manuscript_data = {
            "metadata": {"title": "Test", "author": "Author", "genre": "Fantasy", "total_chapters": 1, "created_at": "2023-01-01"},
            "chapters": [{"title": "Ch1", "content": "content", "number": 1, "word_count": 1}],
            "saved_at": "2023-01-01T00:00:00",
            "file_path": "test.json"
        }

        with patch('builtins.open', mock_open(read_data=json.dumps(manuscript_data))), \
             patch('illustrator.cli.console'):
            result = cli.load_manuscript_draft("test.json")

        assert result is True
        assert cli.manuscript_metadata.title == "Test"
        assert len(cli.chapters) == 1

    def test_load_manuscript_draft_error(self, cli):
        """Test manuscript loading with error."""
        with patch('builtins.open', side_effect=FileNotFoundError), \
             patch('illustrator.cli.console'):
            result = cli.load_manuscript_draft("nonexistent.json")

        assert result is False

    def test_display_saved_manuscripts_menu_no_manuscripts(self, cli):
        """Test manuscript menu with no manuscripts."""
        with patch.object(cli, 'list_saved_manuscripts', return_value=[]):
            result = cli.display_saved_manuscripts_menu()
            assert result is None

    @patch('illustrator.cli.Prompt.ask')
    def test_display_saved_manuscripts_menu_quit(self, mock_prompt, cli):
        """Test manuscript menu with quit option."""
        mock_prompt.return_value = 'q'

        manuscript = MagicMock()
        manuscript.metadata.title = "Test"
        manuscript.saved_at = "2023-01-01T00:00:00"

        with patch.object(cli, 'list_saved_manuscripts', return_value=[manuscript]):
            result = cli.display_saved_manuscripts_menu()
            assert result is None

    @patch('illustrator.cli.Prompt.ask')
    def test_display_saved_manuscripts_menu_select(self, mock_prompt, cli):
        """Test manuscript menu with selection."""
        mock_prompt.return_value = '1'

        manuscript = MagicMock()
        manuscript.metadata.title = "Test"
        manuscript.saved_at = "2023-01-01T00:00:00"
        manuscript.file_path = "test.json"

        with patch.object(cli, 'list_saved_manuscripts', return_value=[manuscript]):
            result = cli.display_saved_manuscripts_menu()
            assert result == "test.json"


class TestCLICommands:
    """Test CLI command functions."""

    @patch('illustrator.cli.ManuscriptCLI')
    def test_analyze_command_list_saved(self, mock_cli_class):
        """Test analyze command with list-saved flag."""
        from illustrator.cli import analyze
        from click.testing import CliRunner

        runner = CliRunner()
        with patch('illustrator.cli.console'):
            result = runner.invoke(analyze, ['--list-saved'])

        # Should complete without error
        assert result.exit_code == 0

    @patch('illustrator.cli.ManuscriptCLI')
    def test_analyze_command_load(self, mock_cli_class):
        """Test analyze command with load option."""
        from illustrator.cli import analyze
        from click.testing import CliRunner

        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_file = f.name

        try:
            mock_instance = MagicMock()
            mock_instance.load_manuscript_draft.return_value = True
            mock_cli_class.return_value = mock_instance

            runner = CliRunner()
            result = runner.invoke(analyze, ['--load', temp_file])

            assert result.exit_code == 0
        finally:
            os.unlink(temp_file)

    @patch('illustrator.web.app.run_server')
    def test_start_command(self, mock_run_server):
        """Test start command."""
        from illustrator.cli import start
        from click.testing import CliRunner

        runner = CliRunner()
        with patch('illustrator.cli.console'), \
             patch('webbrowser.open'), \
             patch('threading.Thread'):
            result = runner.invoke(start, ['--host', '127.0.0.1', '--port', '8000'])

        mock_run_server.assert_called_once()

    def test_main_function(self):
        """Test main function."""
        from illustrator.cli import main

        with patch('illustrator.cli.cli') as mock_cli:
            main()
            mock_cli.assert_called_once()


class TestCLIHelperFunctions:
    """Test CLI helper functions and edge cases."""

    def test_cli_with_keyboard_interrupt(self):
        """Test CLI handling of keyboard interrupt."""
        from illustrator.cli import analyze
        from click.testing import CliRunner

        runner = CliRunner()
        with patch('illustrator.cli.ManuscriptCLI.setup_environment', side_effect=KeyboardInterrupt):
            result = runner.invoke(analyze)

        assert result.exit_code == 0

    def test_cli_with_general_exception(self):
        """Test CLI handling of general exceptions."""
        from illustrator.cli import analyze
        from click.testing import CliRunner

        runner = CliRunner()
        with patch('illustrator.cli.ManuscriptCLI.setup_environment', side_effect=Exception("Test error")):
            result = runner.invoke(analyze)

        assert result.exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])