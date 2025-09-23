"""Comprehensive unit tests for manuscript export routes."""

import json
import tempfile
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from illustrator.models import SavedManuscript, ManuscriptMetadata, Chapter
from illustrator.web.routes.manuscripts import (
    generate_html_export,
    generate_docx_export,
    generate_pdf_export,
    get_manuscript_processing_status,
    SAVED_MANUSCRIPTS_DIR
)
from illustrator.web.app import app


class TestManuscriptExportRoutes:
    """Test class for manuscript export functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_manuscripts_dir = self.temp_dir / "manuscripts"
        self.test_exports_dir = self.temp_dir / "exports"

        self.test_manuscripts_dir.mkdir(parents=True, exist_ok=True)
        self.test_exports_dir.mkdir(parents=True, exist_ok=True)

        # Create test manuscript
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
                    content="This is the first chapter.\n\nIt has multiple paragraphs.",
                    number=1,
                    word_count=8
                ),
                Chapter(
                    title="The Journey",
                    content="This is the second chapter.\n\nThe story continues here.",
                    number=2,
                    word_count=9
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

        # Test client
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_generate_html_export(self):
        """Test HTML export generation."""
        html_content = generate_html_export(self.test_manuscript)

        assert isinstance(html_content, str)
        assert "Test Novel" in html_content
        assert "Test Author" in html_content
        assert "Fantasy" in html_content
        assert "The Beginning" in html_content
        assert "The Journey" in html_content
        assert "This is the first chapter" in html_content
        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content
        assert "</html>" in html_content

        # Check for proper HTML structure
        assert "<title>Test Novel by Test Author</title>" in html_content
        assert 'class="manuscript-title"' in html_content
        assert 'class="chapter-title"' in html_content

    def test_generate_docx_export_success(self):
        """Test successful DOCX export generation."""
        test_file = self.test_exports_dir / "test_export.docx"

        with patch('illustrator.web.routes.manuscripts.Document') as mock_doc_class:
            mock_doc = MagicMock()
            mock_doc_class.return_value = mock_doc

            # Mock paragraph and run objects
            mock_para = MagicMock()
            mock_run = MagicMock()
            mock_para.add_run.return_value = mock_run
            mock_doc.add_paragraph.return_value = mock_para
            mock_doc.add_heading.return_value = MagicMock()

            generate_docx_export(self.test_manuscript, test_file)

            # Verify document creation calls
            mock_doc_class.assert_called_once()
            assert mock_doc.add_paragraph.call_count >= 3  # Title, author, metadata
            assert mock_doc.add_heading.call_count == 2   # Two chapters
            mock_doc.save.assert_called_once_with(str(test_file))

    def test_generate_docx_export_missing_dependency(self):
        """Test DOCX export with missing python-docx dependency."""
        test_file = self.test_exports_dir / "test_export.docx"

        with patch('illustrator.web.routes.manuscripts.Document', side_effect=ImportError("No module named 'docx'")):
            with pytest.raises(Exception) as exc_info:
                generate_docx_export(self.test_manuscript, test_file)

            assert "python-docx" in str(exc_info.value)

    @patch('illustrator.web.routes.manuscripts.SimpleDocTemplate')
    @patch('illustrator.web.routes.manuscripts.getSampleStyleSheet')
    def test_generate_pdf_export_success(self, mock_styles, mock_doc):
        """Test successful PDF export generation."""
        test_file = self.test_exports_dir / "test_export.pdf"

        # Mock reportlab components
        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance

        mock_styles_dict = {
            'Heading1': MagicMock(),
            'Heading2': MagicMock(),
            'Normal': MagicMock()
        }
        mock_styles.return_value = mock_styles_dict

        # Run the async function
        import asyncio
        asyncio.run(generate_pdf_export(self.test_manuscript, test_file))

        # Verify PDF generation calls
        mock_doc.assert_called_once()
        mock_doc_instance.build.assert_called_once()

    @patch('illustrator.web.routes.manuscripts.SimpleDocTemplate', side_effect=ImportError("No module named 'reportlab'"))
    @patch('illustrator.web.routes.manuscripts.weasyprint', side_effect=ImportError("No module named 'weasyprint'"))
    def test_generate_pdf_export_missing_dependencies(self, mock_weasy, mock_reportlab):
        """Test PDF export with missing dependencies."""
        test_file = self.test_exports_dir / "test_export.pdf"

        import asyncio
        with pytest.raises(Exception) as exc_info:
            asyncio.run(generate_pdf_export(self.test_manuscript, test_file))

        assert "reportlab" in str(exc_info.value) or "weasyprint" in str(exc_info.value)

    @patch('illustrator.web.routes.manuscripts.weasyprint')
    @patch('illustrator.web.routes.manuscripts.SimpleDocTemplate', side_effect=ImportError)
    def test_generate_pdf_export_weasyprint_fallback(self, mock_reportlab, mock_weasy):
        """Test PDF export fallback to weasyprint."""
        test_file = self.test_exports_dir / "test_export.pdf"

        # Mock weasyprint
        mock_html = MagicMock()
        mock_weasy.HTML.return_value = mock_html

        import asyncio
        asyncio.run(generate_pdf_export(self.test_manuscript, test_file))

        # Verify weasyprint was used as fallback
        mock_weasy.HTML.assert_called_once()
        mock_html.write_pdf.assert_called_once_with(str(test_file))

    def test_get_manuscript_processing_status_active(self):
        """Test getting processing status for active manuscript."""
        mock_session_data = MagicMock()
        mock_session_data.manuscript_id = self.manuscript_id
        mock_status = MagicMock()
        mock_status.status = "processing"
        mock_session_data.status = mock_status

        with patch('illustrator.web.routes.manuscripts.connection_manager') as mock_manager:
            mock_manager.sessions = {"session1": mock_session_data}

            status = get_manuscript_processing_status(self.manuscript_id)

            assert status == "processing"

    def test_get_manuscript_processing_status_draft(self):
        """Test getting processing status for draft manuscript."""
        with patch('illustrator.web.routes.manuscripts.connection_manager') as mock_manager:
            mock_manager.sessions = {}

            status = get_manuscript_processing_status(self.manuscript_id)

            assert status == "draft"

    def test_get_manuscript_processing_status_error(self):
        """Test getting processing status with error."""
        with patch('illustrator.web.routes.manuscripts.connection_manager', side_effect=Exception("Connection error")):
            status = get_manuscript_processing_status(self.manuscript_id)

            assert status == "draft"

    def test_export_manuscript_endpoint_json(self):
        """Test manuscript export endpoint with JSON format."""
        with patch('illustrator.web.routes.manuscripts.SAVED_MANUSCRIPTS_DIR', self.test_manuscripts_dir):
            with patch('illustrator.web.routes.manuscripts.Path') as mock_path:
                # Mock exports directory
                mock_exports_dir = MagicMock()
                mock_path.return_value = mock_exports_dir
                mock_exports_dir.mkdir = MagicMock()
                mock_exports_dir.__truediv__ = MagicMock(return_value=self.test_exports_dir / "test.json")

                # Mock file operations
                mock_file_path = MagicMock()
                mock_file_path.stat.return_value.st_size = 1024
                mock_exports_dir.__truediv__.return_value = mock_file_path

                with patch('builtins.open', create=True) as mock_open:
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_file

                    response = self.client.post(f"/api/manuscripts/{self.manuscript_id}/export?export_format=json")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["export_format"] == "JSON"
        assert data["manuscript_title"] == "Test Novel"
        assert "filename" in data
        assert "download_url" in data

    def test_export_manuscript_endpoint_unsupported_format(self):
        """Test manuscript export endpoint with unsupported format."""
        with patch('illustrator.web.routes.manuscripts.SAVED_MANUSCRIPTS_DIR', self.test_manuscripts_dir):
            response = self.client.post(f"/api/manuscripts/{self.manuscript_id}/export?export_format=xyz")

        assert response.status_code == 400
        assert "Unsupported export format" in response.json()["detail"]

    def test_export_manuscript_endpoint_not_found(self):
        """Test manuscript export endpoint with non-existent manuscript."""
        nonexistent_id = str(uuid.uuid4())

        with patch('illustrator.web.routes.manuscripts.SAVED_MANUSCRIPTS_DIR', self.test_manuscripts_dir):
            response = self.client.post(f"/api/manuscripts/{nonexistent_id}/export?export_format=json")

        assert response.status_code == 404
        assert "Manuscript not found" in response.json()["detail"]

    def test_download_exported_file_endpoint(self):
        """Test download exported file endpoint."""
        # Create test export file
        test_filename = "test_export.json"
        test_content = '{"test": "content"}'

        with patch('illustrator.web.routes.manuscripts.Path') as mock_path:
            mock_file_path = MagicMock()
            mock_file_path.exists.return_value = True
            mock_path.return_value = mock_file_path

            with patch('illustrator.web.app.FileResponse') as mock_response:
                mock_response.return_value = MagicMock()

                response = self.client.get(f"/api/manuscripts/{self.manuscript_id}/download/{test_filename}")

                mock_response.assert_called_once()

    def test_download_exported_file_endpoint_not_found(self):
        """Test download exported file endpoint with non-existent file."""
        test_filename = "nonexistent.json"

        with patch('illustrator.web.routes.manuscripts.Path') as mock_path:
            mock_file_path = MagicMock()
            mock_file_path.exists.return_value = False
            mock_path.return_value = mock_file_path

            response = self.client.get(f"/api/manuscripts/{self.manuscript_id}/download/{test_filename}")

        assert response.status_code == 404

    def test_download_exported_file_endpoint_invalid_filename(self):
        """Test download exported file endpoint with invalid filename."""
        test_filename = "../../../etc/passwd"

        response = self.client.get(f"/api/manuscripts/{self.manuscript_id}/download/{test_filename}")

        assert response.status_code == 400
        assert "Invalid filename" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__])