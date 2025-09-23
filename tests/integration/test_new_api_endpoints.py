"""Integration tests for new API endpoints (chapter analysis and manuscript export)."""

import json
import tempfile
import shutil
import uuid
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from illustrator.models import SavedManuscript, ManuscriptMetadata, Chapter, EmotionalTone
from illustrator.web.app import app


class TestNewAPIEndpointsIntegration:
    """Integration tests for new API endpoints."""

    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_manuscripts_dir = self.temp_dir / "saved_manuscripts"
        self.test_output_dir = self.temp_dir / "illustrator_output"
        self.test_analysis_dir = self.test_output_dir / "analysis"
        self.test_exports_dir = self.test_output_dir / "exports"

        # Create test directories
        self.test_manuscripts_dir.mkdir(parents=True, exist_ok=True)
        self.test_analysis_dir.mkdir(parents=True, exist_ok=True)
        self.test_exports_dir.mkdir(parents=True, exist_ok=True)

        # Patch the directory paths in routes
        self.manuscripts_patcher = patch(
            'illustrator.web.routes.manuscripts.SAVED_MANUSCRIPTS_DIR',
            self.test_manuscripts_dir
        )
        self.chapters_patcher = patch(
            'illustrator.web.routes.chapters.SAVED_MANUSCRIPTS_DIR',
            self.test_manuscripts_dir
        )

        self.manuscripts_patcher.start()
        self.chapters_patcher.start()

        # Create comprehensive test manuscript
        self.test_manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Epic Fantasy Novel",
                author="Jane Doe",
                genre="High Fantasy",
                total_chapters=3,
                created_at=datetime.now().isoformat()
            ),
            chapters=[
                Chapter(
                    title="The Awakening",
                    content="""The hero awoke to sunlight streaming through ancient windows. Joy filled their heart as they remembered the prophecy.

                    But dark shadows lurked in the corners of the room, and fear crept into their mind. The mystical artifact glowed with an otherworldly light.

                    Tension mounted as footsteps echoed in the hallway. Who could be approaching at this early hour?""",
                    number=1,
                    word_count=50
                ),
                Chapter(
                    title="The Journey Begins",
                    content="""The path ahead was treacherous, filled with unknown dangers. Love bloomed between the companions despite the perilous circumstances.

                    Anger flared when they discovered the betrayal. The villain's laugh echoed through the mountain pass.

                    Mystery surrounded the ancient ruins they discovered, holding secrets of ages past.""",
                    number=2,
                    word_count=45
                ),
                Chapter(
                    title="The Final Confrontation",
                    content="""The final battle commenced with thunderous roars. Sadness overwhelmed the hero as their mentor fell.

                    But hope remained alive in their heart. The magical sword blazed with righteous fire.

                    Victory was achieved, but at a great cost. The world was saved, yet forever changed.""",
                    number=3,
                    word_count=40
                )
            ],
            saved_at=datetime.now().isoformat(),
            file_path=str(self.test_manuscripts_dir / "epic_fantasy.json")
        )

        # Save manuscript to file
        self.manuscript_file = self.test_manuscripts_dir / "epic_fantasy.json"
        with open(self.manuscript_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_manuscript.model_dump(), f, indent=2, ensure_ascii=False)

        self.manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(self.manuscript_file)))
        self.chapter_1_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.manuscript_id}_1"))
        self.chapter_2_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.manuscript_id}_2"))

    def teardown_method(self):
        """Clean up test environment."""
        self.manuscripts_patcher.stop()
        self.chapters_patcher.stop()

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('illustrator.web.routes.chapters.EmotionalAnalyzer')
    @patch('illustrator.web.routes.chapters.LiterarySceneDetector')
    @patch('illustrator.web.routes.chapters.NarrativeAnalyzer')
    @patch('illustrator.web.routes.chapters.ComprehensiveSceneAnalyzer')
    @patch('illustrator.web.routes.chapters.init_chat_model')
    def test_full_chapter_analysis_workflow(
        self,
        mock_init_chat_model,
        mock_comprehensive,
        mock_narrative,
        mock_scene_detector,
        mock_emotional
    ):
        """Test complete chapter analysis workflow from API call to saved results."""
        # Set up API key
        os.environ['ANTHROPIC_API_KEY'] = 'test_key_12345'

        try:
            # Mock all dependencies
            mock_llm = MagicMock()
            mock_init_chat_model.return_value = mock_llm

            # Mock emotional analyzer
            mock_emotional_instance = MagicMock()
            mock_emotional.return_value = mock_emotional_instance

            mock_emotional_moments = [
                MagicMock(
                    text_excerpt="Joy filled their heart",
                    emotional_tones=[EmotionalTone.JOY],
                    intensity_score=0.85,
                    visual_potential=0.9,
                    context="Hero discovering prophecy",
                    start_position=15,
                    end_position=35
                ),
                MagicMock(
                    text_excerpt="dark shadows lurked",
                    emotional_tones=[EmotionalTone.FEAR, EmotionalTone.MYSTERY],
                    intensity_score=0.75,
                    visual_potential=0.85,
                    context="Ominous presence in room",
                    start_position=60,
                    end_position=80
                )
            ]
            mock_emotional_instance.analyze_chapter_with_scenes = AsyncMock(
                return_value=mock_emotional_moments
            )

            # Mock scene detector
            mock_scene_instance = MagicMock()
            mock_scene_detector.return_value = mock_scene_instance

            mock_scenes = [
                MagicMock(
                    scene_type="awakening",
                    primary_characters=["hero"],
                    location="ancient chamber",
                    time_context="dawn",
                    emotional_tone="hopeful_with_foreboding",
                    text="The hero awoke to sunlight...",
                    start_position=0,
                    end_position=100
                ),
                MagicMock(
                    scene_type="discovery",
                    primary_characters=["hero"],
                    location="ancient chamber",
                    time_context="morning",
                    emotional_tone="mysterious",
                    text="The mystical artifact glowed...",
                    start_position=100,
                    end_position=200
                )
            ]
            mock_scene_instance.extract_scenes = AsyncMock(return_value=mock_scenes)

            # Mock narrative analyzer
            mock_narrative_instance = MagicMock()
            mock_narrative.return_value = mock_narrative_instance

            mock_tension_points = [
                MagicMock(
                    position=150,
                    intensity=0.8,
                    description="Footsteps approaching",
                    tension_type="suspense"
                )
            ]

            mock_character_arcs = [
                MagicMock(
                    character_name="hero",
                    arc_type="hero_journey",
                    development_stage="call_to_adventure",
                    key_moments=["awakening", "discovery"]
                )
            ]

            mock_narrative_structure = MagicMock(
                structure_type="hero_journey",
                pacing="building",
                tension_points=mock_tension_points,
                character_arcs=mock_character_arcs,
                themes=["destiny", "courage", "mystery"],
                narrative_devices=["foreshadowing", "symbolism"]
            )
            mock_narrative_instance.analyze_structure = AsyncMock(
                return_value=mock_narrative_structure
            )

            # Mock comprehensive analyzer
            mock_comprehensive_instance = MagicMock()
            mock_comprehensive.return_value = mock_comprehensive_instance
            mock_comprehensive_instance.analyze_chapter_comprehensive = AsyncMock(
                return_value=mock_emotional_moments
            )

            # Mock file system operations for saving analysis
            with patch('illustrator.web.routes.chapters.Path') as mock_path:
                mock_analysis_dir = MagicMock()
                mock_analysis_file = MagicMock()
                mock_path.return_value = mock_analysis_dir
                mock_analysis_dir.mkdir = MagicMock()
                mock_analysis_dir.__truediv__ = MagicMock(return_value=mock_analysis_file)

                with patch('builtins.open', create=True) as mock_open:
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    mock_open.return_value.__exit__.return_value = None

                    # Make the API call
                    response = self.client.post(f"/api/chapters/{self.chapter_1_id}/analyze")

            # Verify response
            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["chapter_id"] == self.chapter_1_id
            assert data["chapter_title"] == "The Awakening"

            # Verify analysis structure
            analysis = data["analysis"]
            assert "emotional_analysis" in analysis
            assert "scene_analysis" in analysis
            assert "narrative_analysis" in analysis
            assert "illustration_potential" in analysis
            assert "statistics" in analysis
            assert "summary" in analysis

            # Verify emotional analysis details
            emotional = analysis["emotional_analysis"]
            assert emotional["total_moments"] == 2
            assert len(emotional["moments"]) == 2

            # Verify scene analysis details
            scenes = analysis["scene_analysis"]
            assert scenes["total_scenes"] == 2
            assert len(scenes["scenes"]) == 2

            # Verify narrative analysis details
            narrative = analysis["narrative_analysis"]
            assert narrative["structure_type"] == "hero_journey"
            assert len(narrative["themes"]) == 3

            # Verify statistics
            stats = analysis["statistics"]
            assert stats["word_count"] == 50
            assert "emotional_density" in stats

            # Verify analysis was saved
            mock_analysis_dir.mkdir.assert_called()
            mock_open.assert_called()

        finally:
            # Clean up environment variable
            if 'ANTHROPIC_API_KEY' in os.environ:
                del os.environ['ANTHROPIC_API_KEY']

    def test_manuscript_export_complete_workflow(self):
        """Test complete manuscript export workflow for all formats."""
        export_formats = ['json', 'html', 'pdf', 'docx']

        for export_format in export_formats:
            with self.subTest(format=export_format):
                with patch('illustrator.web.routes.manuscripts.Path') as mock_path:
                    # Mock exports directory
                    mock_exports_dir = MagicMock()
                    mock_exports_dir.mkdir = MagicMock()
                    mock_path.return_value = mock_exports_dir

                    # Mock file path
                    mock_file_path = MagicMock()
                    mock_file_path.stat.return_value.st_size = 2048
                    mock_exports_dir.__truediv__ = MagicMock(return_value=mock_file_path)

                    # Mock file operations based on format
                    if export_format == 'json':
                        with patch('builtins.open', create=True) as mock_open:
                            mock_file = MagicMock()
                            mock_open.return_value.__enter__.return_value = mock_file

                            response = self.client.post(
                                f"/api/manuscripts/{self.manuscript_id}/export?export_format={export_format}"
                            )

                    elif export_format == 'html':
                        with patch('builtins.open', create=True) as mock_open:
                            mock_file = MagicMock()
                            mock_open.return_value.__enter__.return_value = mock_file

                            response = self.client.post(
                                f"/api/manuscripts/{self.manuscript_id}/export?export_format={export_format}"
                            )

                    elif export_format == 'pdf':
                        with patch('illustrator.web.routes.manuscripts.SimpleDocTemplate') as mock_doc:
                            mock_doc_instance = MagicMock()
                            mock_doc.return_value = mock_doc_instance

                            with patch('illustrator.web.routes.manuscripts.getSampleStyleSheet') as mock_styles:
                                mock_styles.return_value = {
                                    'Heading1': MagicMock(),
                                    'Normal': MagicMock()
                                }

                                response = self.client.post(
                                    f"/api/manuscripts/{self.manuscript_id}/export?export_format={export_format}"
                                )

                    elif export_format == 'docx':
                        with patch('illustrator.web.routes.manuscripts.Document') as mock_doc_class:
                            mock_doc = MagicMock()
                            mock_doc_class.return_value = mock_doc
                            mock_para = MagicMock()
                            mock_doc.add_paragraph.return_value = mock_para
                            mock_run = MagicMock()
                            mock_para.add_run.return_value = mock_run

                            response = self.client.post(
                                f"/api/manuscripts/{self.manuscript_id}/export?export_format={export_format}"
                            )

                    # Verify response
                    assert response.status_code == 200, f"Failed for format {export_format}: {response.text}"
                    data = response.json()

                    assert data["success"] is True
                    assert data["export_format"] == export_format.upper()
                    assert data["manuscript_title"] == "Epic Fantasy Novel"
                    assert "filename" in data
                    assert "download_url" in data
                    assert "file_size" in data

    def test_chapter_analysis_error_scenarios(self):
        """Test chapter analysis error handling scenarios."""
        # Test missing API key
        response = self.client.post(f"/api/chapters/{self.chapter_1_id}/analyze")
        assert response.status_code == 503
        assert "Anthropic API key is required" in response.json()["detail"]

        # Test non-existent chapter
        fake_chapter_id = str(uuid.uuid4())
        os.environ['ANTHROPIC_API_KEY'] = 'test_key'
        try:
            response = self.client.post(f"/api/chapters/{fake_chapter_id}/analyze")
            assert response.status_code == 404
            assert "Chapter not found" in response.json()["detail"]
        finally:
            del os.environ['ANTHROPIC_API_KEY']

    def test_manuscript_export_error_scenarios(self):
        """Test manuscript export error handling scenarios."""
        # Test unsupported format
        response = self.client.post(
            f"/api/manuscripts/{self.manuscript_id}/export?export_format=unsupported"
        )
        assert response.status_code == 400
        assert "Unsupported export format" in response.json()["detail"]

        # Test non-existent manuscript
        fake_manuscript_id = str(uuid.uuid4())
        response = self.client.post(
            f"/api/manuscripts/{fake_manuscript_id}/export?export_format=json"
        )
        assert response.status_code == 404
        assert "Manuscript not found" in response.json()["detail"]

    def test_download_endpoint_functionality(self):
        """Test file download endpoint."""
        test_filename = "test_export.json"

        # Test successful download
        with patch('illustrator.web.app.Path') as mock_path:
            mock_file_path = MagicMock()
            mock_file_path.exists.return_value = True
            mock_path.return_value = mock_file_path

            with patch('illustrator.web.app.FileResponse') as mock_file_response:
                mock_response = MagicMock()
                mock_file_response.return_value = mock_response

                response = self.client.get(f"/api/manuscripts/{self.manuscript_id}/download/{test_filename}")

                mock_file_response.assert_called_once()

        # Test file not found
        with patch('illustrator.web.app.Path') as mock_path:
            mock_file_path = MagicMock()
            mock_file_path.exists.return_value = False
            mock_path.return_value = mock_file_path

            response = self.client.get(f"/api/manuscripts/{self.manuscript_id}/download/{test_filename}")
            assert response.status_code == 404

        # Test invalid filename (path traversal attack)
        malicious_filename = "../../../etc/passwd"
        response = self.client.get(f"/api/manuscripts/{self.manuscript_id}/download/{malicious_filename}")
        assert response.status_code == 400
        assert "Invalid filename" in response.json()["detail"]

    def test_analysis_persistence_and_loading(self):
        """Test that analysis results are saved and can be loaded."""
        # Mock saving analysis
        test_analysis = {
            "emotional_analysis": {"total_moments": 2},
            "summary": {"analysis_timestamp": datetime.now().isoformat()}
        }

        analysis_file = self.test_analysis_dir / f"chapter_{self.chapter_1_id}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(test_analysis, f)

        # Test loading the analysis
        from illustrator.web.routes.chapters import load_chapter_analysis

        with patch('illustrator.web.routes.chapters.Path') as mock_path:
            mock_path.return_value = self.test_analysis_dir
            mock_path.return_value.__truediv__ = lambda self, other: analysis_file

            loaded_analysis = load_chapter_analysis(self.chapter_1_id)

            assert loaded_analysis is not None
            assert loaded_analysis["emotional_analysis"]["total_moments"] == 2

    def test_image_counting_functionality(self):
        """Test image counting across different scenarios."""
        from illustrator.web.routes.chapters import count_chapter_images

        # Test with no images
        count = count_chapter_images(self.manuscript_id, 1)
        assert count == 0

        # Test with filesystem images (mocked)
        with patch('illustrator.web.routes.chapters.IllustrationService', side_effect=Exception("DB error")):
            with patch('illustrator.web.routes.chapters.Path') as mock_path:
                mock_images_dir = MagicMock()
                mock_images_dir.exists.return_value = True

                # Mock image files
                mock_files = [
                    MagicMock(name="chapter_1_scene_1.png", is_file=MagicMock(return_value=True)),
                    MagicMock(name="chapter_1_scene_2.png", is_file=MagicMock(return_value=True)),
                    MagicMock(name="chapter_2_scene_1.png", is_file=MagicMock(return_value=True)),
                ]
                for mock_file in mock_files:
                    mock_file.name = mock_file.name

                mock_images_dir.iterdir.return_value = mock_files
                mock_path.return_value = mock_images_dir

                count = count_chapter_images(self.manuscript_id, 1)
                assert count == 2  # Should find 2 chapter_1 images


if __name__ == "__main__":
    pytest.main([__file__])