"""Comprehensive unit tests for chapter analysis routes."""

import json
import tempfile
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from illustrator.models import SavedManuscript, ManuscriptMetadata, Chapter, EmotionalMoment, EmotionalTone
from illustrator.web.routes.chapters import (
    load_manuscript_by_id,
    save_manuscript,
    load_chapter_analysis,
    count_chapter_images,
    SAVED_MANUSCRIPTS_DIR
)
from illustrator.web.app import app


class TestChapterAnalysisRoutes:
    """Test class for chapter analysis functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_manuscripts_dir = self.temp_dir / "manuscripts"
        self.test_analysis_dir = self.temp_dir / "analysis"
        self.test_images_dir = self.temp_dir / "generated_images"

        self.test_manuscripts_dir.mkdir(parents=True, exist_ok=True)
        self.test_analysis_dir.mkdir(parents=True, exist_ok=True)
        self.test_images_dir.mkdir(parents=True, exist_ok=True)

        # Create test manuscript with chapters
        self.test_manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test Novel",
                author="Test Author",
                genre="Fantasy",
                total_chapters=2,
                created_at=datetime.now().isoformat()
            ),
            chapters=[
                Chapter(
                    title="The Beginning",
                    content="This is the first chapter of our epic tale. The hero felt great joy as he embarked on his journey. Dark shadows lurked in the forest, creating a sense of fear and mystery.",
                    number=1,
                    word_count=30
                ),
                Chapter(
                    title="The Journey Continues",
                    content="The adventure deepens as our heroes face new challenges. Tension filled the air as they approached the castle. Love bloomed between the characters despite the danger.",
                    number=2,
                    word_count=28
                )
            ],
            saved_at=datetime.now().isoformat(),
            file_path=str(self.test_manuscripts_dir / "test_novel.json")
        )

        # Save test manuscript to file
        self.manuscript_file = self.test_manuscripts_dir / "test_novel.json"
        with open(self.manuscript_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_manuscript.model_dump(), f, indent=2, ensure_ascii=False)

        self.manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(self.manuscript_file)))
        self.chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.manuscript_id}_1"))

        # Test client
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_load_chapter_analysis_existing(self):
        """Test loading existing chapter analysis."""
        # Create actual test analysis file
        analysis_data = {
            "emotional_analysis": {
                "total_moments": 3,
                "moments": [
                    {
                        "text_excerpt": "felt great joy",
                        "emotional_tones": ["joy"],
                        "intensity_score": 0.8,
                        "visual_potential": 0.7
                    }
                ]
            },
            "summary": {
                "analysis_timestamp": datetime.now().isoformat()
            }
        }

        analysis_file = self.test_analysis_dir / f"chapter_{self.chapter_id}_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f)

        # Use actual file system for this test
        import illustrator.web.routes.chapters as chapters_module
        original_path = chapters_module.Path

        try:
            # Temporarily replace Path in the module
            chapters_module.Path = lambda x: self.test_output_dir if x == "illustrator_output" else original_path(x)

            result = load_chapter_analysis(self.chapter_id)

            assert result is not None
            assert result["emotional_analysis"]["total_moments"] == 3
            assert len(result["emotional_analysis"]["moments"]) == 1

        finally:
            # Restore original Path
            chapters_module.Path = original_path

    def test_load_chapter_analysis_nonexistent(self):
        """Test loading non-existent chapter analysis."""
        import illustrator.web.routes.chapters as chapters_module
        original_path = chapters_module.Path

        try:
            # Temporarily replace Path in the module
            chapters_module.Path = lambda x: self.test_output_dir if x == "illustrator_output" else original_path(x)

            result = load_chapter_analysis("nonexistent_id")

            assert result is None

        finally:
            # Restore original Path
            chapters_module.Path = original_path

    def test_count_chapter_images_filesystem(self):
        """Test counting chapter images from filesystem."""
        # Create test image files
        test_files = [
            "chapter_1_scene_1.png",
            "chapter_1_scene_2.png",
            "chapter_2_scene_1.png",
            "other_file.png"
        ]

        for filename in test_files:
            (self.test_images_dir / filename).touch()

        # Use actual filesystem for this test
        import illustrator.web.routes.chapters as chapters_module
        original_path = chapters_module.Path

        try:
            # Mock only the Path constructor to return our test directory
            def mock_path_constructor(path_str):
                if path_str == "illustrator_output/generated_images":
                    return self.test_images_dir
                return original_path(path_str)

            chapters_module.Path = mock_path_constructor

            # Mock database service to force filesystem fallback
            with patch('illustrator.web.routes.chapters.IllustrationService', side_effect=Exception("DB error")):
                count = count_chapter_images(self.manuscript_id, 1)

            # Should find 2 images for chapter 1
            assert count == 2

        finally:
            chapters_module.Path = original_path

    def test_count_chapter_images_database_fallback(self):
        """Test counting images with database fallback to filesystem."""
        # Create test image files
        test_files = ["chapter_1_scene_1.png", "chapter_1_scene_2.png"]
        for filename in test_files:
            (self.test_images_dir / filename).touch()

        import illustrator.web.routes.chapters as chapters_module
        original_path = chapters_module.Path

        try:
            def mock_path_constructor(path_str):
                if path_str == "illustrator_output/generated_images":
                    return self.test_images_dir
                return original_path(path_str)

            chapters_module.Path = mock_path_constructor

            # Mock database service to raise exception, triggering filesystem fallback
            with patch('illustrator.web.routes.chapters.IllustrationService', side_effect=Exception("Database error")):
                count = count_chapter_images(self.manuscript_id, 1)
                assert count == 2

        finally:
            chapters_module.Path = original_path

    @patch('illustrator.web.routes.chapters.EmotionalAnalyzer')
    @patch('illustrator.web.routes.chapters.LiterarySceneDetector')
    @patch('illustrator.web.routes.chapters.NarrativeAnalyzer')
    @patch('generate_scene_illustrations.ComprehensiveSceneAnalyzer')
    @patch('langchain.chat_models.init_chat_model')
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    def test_analyze_chapter_endpoint_success(
        self,
        mock_init_chat_model,
        mock_comprehensive,
        mock_narrative,
        mock_scene_detector,
        mock_emotional
    ):
        """Test successful chapter analysis endpoint."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm

        # Mock analyzers
        mock_emotional_instance = MagicMock()
        mock_emotional.return_value = mock_emotional_instance

        mock_scene_instance = MagicMock()
        mock_scene_detector.return_value = mock_scene_instance

        mock_narrative_instance = MagicMock()
        mock_narrative.return_value = mock_narrative_instance

        mock_comprehensive_instance = MagicMock()
        mock_comprehensive.return_value = mock_comprehensive_instance

        # Mock analysis results
        mock_emotional_moments = [
            MagicMock(
                text_excerpt="felt great joy",
                emotional_tones=[EmotionalTone.JOY],
                intensity_score=0.8,
                visual_potential=0.7,
                context="Beginning of journey",
                start_position=0,
                end_position=20
            )
        ]
        mock_emotional_instance.analyze_chapter_with_scenes = AsyncMock(
            return_value=mock_emotional_moments
        )

        mock_scenes = [
            MagicMock(
                scene_type="opening",
                primary_characters=["hero"],
                location="forest",
                time_context="dawn",
                emotional_tone="hopeful",
                text="This is the first chapter...",
                start_position=0,
                end_position=50
            )
        ]
        mock_scene_instance.extract_scenes = AsyncMock(return_value=mock_scenes)

        mock_narrative = MagicMock(
            structure_type="hero_journey",
            pacing="steady",
            tension_points=[],
            character_arcs=[],
            themes=["adventure", "growth"],
            narrative_devices=["foreshadowing"]
        )
        mock_narrative_instance.analyze_structure = AsyncMock(return_value=mock_narrative)

        mock_comprehensive_instance.analyze_chapter_comprehensive = AsyncMock(
            return_value=mock_emotional_moments[:1]
        )

        # Patch the SAVED_MANUSCRIPTS_DIR
        with patch('illustrator.web.routes.chapters.SAVED_MANUSCRIPTS_DIR', self.test_manuscripts_dir):
            with patch('illustrator.web.routes.chapters.Path') as mock_path:
                # Mock analysis directory creation
                mock_analysis_dir = MagicMock()
                mock_analysis_dir.mkdir = MagicMock()
                mock_analysis_file = MagicMock()
                mock_path.return_value = mock_analysis_dir
                mock_analysis_dir.__truediv__ = MagicMock(return_value=mock_analysis_file)

                # Mock file writing
                mock_file = MagicMock()
                mock_analysis_file.open = MagicMock(return_value=mock_file)
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=None)

                response = self.client.post(f"/api/chapters/{self.chapter_id}/analyze")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["chapter_id"] == self.chapter_id
        assert data["chapter_title"] == "The Beginning"
        assert "analysis" in data

        analysis = data["analysis"]
        assert "emotional_analysis" in analysis
        assert "scene_analysis" in analysis
        assert "narrative_analysis" in analysis
        assert "illustration_potential" in analysis
        assert "statistics" in analysis
        assert "summary" in analysis

    def test_analyze_chapter_endpoint_missing_api_key(self):
        """Test chapter analysis endpoint without API key."""
        with patch('illustrator.web.routes.chapters.SAVED_MANUSCRIPTS_DIR', self.test_manuscripts_dir):
            with patch.dict('os.environ', {}, clear=True):
                response = self.client.post(f"/api/chapters/{self.chapter_id}/analyze")

        assert response.status_code == 503
        assert "Anthropic API key is required" in response.json()["detail"]

    def test_analyze_chapter_endpoint_chapter_not_found(self):
        """Test chapter analysis endpoint with non-existent chapter."""
        nonexistent_id = str(uuid.uuid4())

        with patch('illustrator.web.routes.chapters.SAVED_MANUSCRIPTS_DIR', self.test_manuscripts_dir):
            with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
                response = self.client.post(f"/api/chapters/{nonexistent_id}/analyze")

        assert response.status_code == 404
        assert "Chapter not found" in response.json()["detail"]

    @patch('illustrator.web.routes.chapters.EmotionalAnalyzer')
    @patch('langchain.chat_models.init_chat_model')
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    def test_analyze_chapter_endpoint_analysis_failure(
        self,
        mock_init_chat_model,
        mock_emotional
    ):
        """Test chapter analysis endpoint with analysis failure."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_init_chat_model.return_value = mock_llm

        # Mock analyzer to raise exception
        mock_emotional_instance = MagicMock()
        mock_emotional.return_value = mock_emotional_instance
        mock_emotional_instance.analyze_chapter_with_scenes = AsyncMock(
            side_effect=Exception("Analysis failed")
        )

        with patch('illustrator.web.routes.chapters.SAVED_MANUSCRIPTS_DIR', self.test_manuscripts_dir):
            response = self.client.post(f"/api/chapters/{self.chapter_id}/analyze")

        assert response.status_code == 500
        assert "Failed to analyze chapter" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__])