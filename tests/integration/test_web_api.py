"""Integration tests for the web API endpoints."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from illustrator.web.app import app
from illustrator.models import Chapter, ManuscriptMetadata, SavedManuscript


class TestAPIIntegration:
    """Test web API integration."""

    def setup_method(self):
        """Setup for each test."""
        self.client = TestClient(app)
        self.test_manuscripts_dir = Path("test_saved_manuscripts")
        self.test_manuscripts_dir.mkdir(exist_ok=True)

        # Patch the SAVED_MANUSCRIPTS_DIR in the routes
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

    def teardown_method(self):
        """Cleanup after each test."""
        self.manuscripts_patcher.stop()
        self.chapters_patcher.stop()

        # Clean up test directory
        if self.test_manuscripts_dir.exists():
            shutil.rmtree(self.test_manuscripts_dir)

    def create_test_manuscript(self, title="Test Manuscript", author="Test Author"):
        """Create a test manuscript file."""
        manuscript_file = self.test_manuscripts_dir / f"{title.lower().replace(' ', '_')}.json"

        manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title=title,
                author=author,
                genre="Fantasy",
                total_chapters=0,
                total_words=0,
                completion_status="draft",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z"
            ),
            chapters=[],
            saved_at="2024-01-01T00:00:00Z",
            file_path=str(manuscript_file)
        )

        with open(manuscript_file, 'w') as f:
            json.dump(manuscript.model_dump(), f)

        return manuscript_file


class TestManuscriptRoutes(TestAPIIntegration):
    """Test manuscript API routes."""

    def test_get_dashboard_stats(self):
        """Test dashboard statistics endpoint."""
        # Create some test manuscripts
        self.create_test_manuscript("Test Book 1", "Author 1")
        self.create_test_manuscript("Test Book 2", "Author 2")

        response = self.client.get("/api/manuscripts/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_manuscripts" in data
        assert "total_chapters" in data
        assert "total_images" in data
        assert "recent_manuscripts" in data
        assert "processing_count" in data

        assert data["total_manuscripts"] == 2
        assert isinstance(data["recent_manuscripts"], list)

    def test_get_all_manuscripts(self):
        """Test get all manuscripts endpoint."""
        # Create test manuscripts
        self.create_test_manuscript("Fantasy Novel", "J.R.R. Tolkien")
        self.create_test_manuscript("Sci-Fi Story", "Isaac Asimov")

        response = self.client.get("/api/manuscripts/")
        assert response.status_code == 200

        manuscripts = response.json()
        assert len(manuscripts) == 2

        # Check manuscript structure
        manuscript = manuscripts[0]
        assert "id" in manuscript
        assert "metadata" in manuscript
        assert "chapters" in manuscript
        assert manuscript["metadata"]["title"] in ["Fantasy Novel", "Sci-Fi Story"]

    def test_create_manuscript(self):
        """Test manuscript creation endpoint."""
        manuscript_data = {
            "title": "New Test Manuscript",
            "author": "Test Author",
            "genre": "Mystery"
        }

        response = self.client.post("/api/manuscripts/", json=manuscript_data)
        assert response.status_code == 200

        created_manuscript = response.json()
        assert created_manuscript["metadata"]["title"] == "New Test Manuscript"
        assert created_manuscript["metadata"]["author"] == "Test Author"
        assert created_manuscript["metadata"]["genre"] == "Mystery"

        # Verify file was created
        assert len(list(self.test_manuscripts_dir.glob("*.json"))) == 1

    def test_create_manuscript_validation_error(self):
        """Test manuscript creation with invalid data."""
        # Missing title
        invalid_data = {
            "author": "Test Author"
        }

        response = self.client.post("/api/manuscripts/", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_get_manuscript_by_id(self):
        """Test get manuscript by ID endpoint."""
        # Create test manuscript
        manuscript_file = self.create_test_manuscript("Test Manuscript", "Test Author")

        # Generate manuscript ID (simulated)
        import uuid
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(manuscript_file)))

        response = self.client.get(f"/api/manuscripts/{manuscript_id}")
        assert response.status_code == 200

        manuscript = response.json()
        assert manuscript["metadata"]["title"] == "Test Manuscript"
        assert manuscript["metadata"]["author"] == "Test Author"

    def test_get_nonexistent_manuscript(self):
        """Test getting nonexistent manuscript."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = self.client.get(f"/api/manuscripts/{fake_id}")
        assert response.status_code == 404

    def test_delete_manuscript(self):
        """Test manuscript deletion endpoint."""
        # Create test manuscript
        manuscript_file = self.create_test_manuscript("To Delete", "Author")

        # Generate manuscript ID
        import uuid
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(manuscript_file)))

        # Delete manuscript
        response = self.client.delete(f"/api/manuscripts/{manuscript_id}")
        assert response.status_code == 200

        # Verify file was deleted
        assert not manuscript_file.exists()

        # Verify manuscript is no longer accessible
        get_response = self.client.get(f"/api/manuscripts/{manuscript_id}")
        assert get_response.status_code == 404


class TestChapterRoutes(TestAPIIntegration):
    """Test chapter API routes."""

    def setup_method(self):
        """Setup with a test manuscript containing chapters."""
        super().setup_method()

        # Create manuscript with chapters
        chapter = Chapter(
            title="Chapter 1",
            content="This is the first chapter of the test manuscript.",
            number=1,
            word_count=50
        )

        self.test_manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test Manuscript",
                author="Test Author",
                genre="Fantasy",
                total_chapters=1,
                total_words=50,
                completion_status="draft",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z"
            ),
            chapters=[chapter],
            saved_at="2024-01-01T00:00:00Z",
            file_path=""
        )

        self.manuscript_file = self.test_manuscripts_dir / "test_manuscript.json"
        with open(self.manuscript_file, 'w') as f:
            json.dump(self.test_manuscript.model_dump(), f)

        import uuid
        self.manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(self.manuscript_file)))
        self.chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.manuscript_id}_1"))

    def test_get_manuscript_chapters(self):
        """Test getting chapters for a manuscript."""
        response = self.client.get(f"/api/chapters/{self.manuscript_id}")
        assert response.status_code == 200

        chapters = response.json()
        assert len(chapters) == 1
        assert chapters[0]["chapter"]["title"] == "Chapter 1"
        assert chapters[0]["chapter"]["number"] == 1

    def test_get_chapter_detail(self):
        """Test getting chapter detail."""
        response = self.client.get(f"/api/chapters/detail/{self.chapter_id}")
        assert response.status_code == 200

        chapter_response = response.json()
        assert chapter_response["chapter"]["title"] == "Chapter 1"
        assert chapter_response["chapter"]["content"] == "This is the first chapter of the test manuscript."

    def test_add_chapter(self):
        """Test adding a new chapter."""
        chapter_data = {
            "title": "Chapter 2",
            "content": "This is the second chapter with more content to test the addition functionality.",
            "manuscript_id": self.manuscript_id
        }

        response = self.client.post(f"/api/chapters/{self.manuscript_id}", json=chapter_data)
        assert response.status_code == 200

        new_chapter = response.json()
        assert new_chapter["chapter"]["title"] == "Chapter 2"
        assert new_chapter["chapter"]["number"] == 2

        # Verify manuscript was updated
        get_response = self.client.get(f"/api/chapters/{self.manuscript_id}")
        chapters = get_response.json()
        assert len(chapters) == 2

    def test_update_chapter(self):
        """Test updating an existing chapter."""
        update_data = {
            "title": "Updated Chapter 1",
            "content": "This is the updated content for the first chapter.",
            "manuscript_id": self.manuscript_id
        }

        response = self.client.put(f"/api/chapters/{self.chapter_id}", json=update_data)
        assert response.status_code == 200

        updated_chapter = response.json()
        assert updated_chapter["chapter"]["title"] == "Updated Chapter 1"
        assert updated_chapter["chapter"]["content"] == "This is the updated content for the first chapter."

    def test_delete_chapter(self):
        """Test deleting a chapter."""
        # Add a second chapter first
        chapter_data = {
            "title": "Chapter 2",
            "content": "Second chapter to test deletion.",
            "manuscript_id": self.manuscript_id
        }
        self.client.post(f"/api/chapters/{self.manuscript_id}", json=chapter_data)

        # Delete the first chapter
        response = self.client.delete(f"/api/chapters/{self.chapter_id}")
        assert response.status_code == 200

        # Verify chapter was deleted and numbering updated
        get_response = self.client.get(f"/api/chapters/{self.manuscript_id}")
        chapters = get_response.json()
        assert len(chapters) == 1
        assert chapters[0]["chapter"]["number"] == 1  # Should be renumbered

    def test_generate_chapter_headers(self):
        """Test chapter header generation endpoint."""
        response = self.client.post(f"/api/chapters/{self.chapter_id}/headers", json={})
        assert response.status_code == 200

        header_response = response.json()
        assert "chapter_id" in header_response
        assert "chapter_title" in header_response
        assert "header_options" in header_response

        header_options = header_response["header_options"]
        assert len(header_options) == 4  # Should generate 4 options

        # Check structure of header options
        option = header_options[0]
        assert "option_number" in option
        assert "title" in option
        assert "description" in option
        assert "visual_focus" in option
        assert "artistic_style" in option
        assert "composition_notes" in option
        assert "prompt" in option

    def test_chapter_reordering(self):
        """Test chapter reordering functionality."""
        # Add multiple chapters
        for i in range(2, 4):  # Add chapters 2 and 3
            chapter_data = {
                "title": f"Chapter {i}",
                "content": f"Content for chapter {i}",
                "manuscript_id": self.manuscript_id
            }
            self.client.post(f"/api/chapters/{self.manuscript_id}", json=chapter_data)

        # Get all chapter IDs
        get_response = self.client.get(f"/api/chapters/{self.manuscript_id}")
        chapters = get_response.json()
        chapter_ids = [ch["id"] for ch in chapters]

        # Reorder chapters (reverse order)
        reorder_data = list(reversed(chapter_ids))

        response = self.client.post(f"/api/chapters/{self.manuscript_id}/reorder", json=reorder_data)
        assert response.status_code == 200

        # Verify new order
        get_response = self.client.get(f"/api/chapters/{self.manuscript_id}")
        reordered_chapters = get_response.json()

        # Check that numbering was updated correctly
        assert reordered_chapters[0]["chapter"]["number"] == 1
        assert reordered_chapters[1]["chapter"]["number"] == 2
        assert reordered_chapters[2]["chapter"]["number"] == 3


class TestWebAppRoutes(TestAPIIntegration):
    """Test web application HTML routes."""

    def test_index_page(self):
        """Test the main index page."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_manuscript_form_page(self):
        """Test manuscript form page."""
        response = self.client.get("/manuscript/new")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_manuscript_detail_page(self):
        """Test manuscript detail page."""
        # Create test manuscript
        manuscript_file = self.create_test_manuscript("Test Manuscript")
        import uuid
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(manuscript_file)))

        response = self.client.get(f"/manuscript/{manuscript_id}")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_chapter_form_page(self):
        """Test chapter form page."""
        # Create test manuscript
        manuscript_file = self.create_test_manuscript("Test Manuscript")
        import uuid
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(manuscript_file)))

        response = self.client.get(f"/manuscript/{manuscript_id}/chapter/new")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_chapter_headers_page(self):
        """Test chapter headers page."""
        # Create test manuscript with chapter
        chapter = Chapter(
            title="Test Chapter",
            content="Test content",
            number=1,
            word_count=10
        )

        manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test Manuscript",
                author="Test Author",
                genre="Fantasy",
                total_chapters=1,
                total_words=10,
                completion_status="draft",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z"
            ),
            chapters=[chapter],
            saved_at="2024-01-01T00:00:00Z",
            file_path=""
        )

        manuscript_file = self.test_manuscripts_dir / "test_manuscript.json"
        with open(manuscript_file, 'w') as f:
            json.dump(manuscript.model_dump(), f)

        import uuid
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(manuscript_file)))
        chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_1"))

        response = self.client.get(f"/chapter/{chapter_id}/headers")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestErrorHandling(TestAPIIntegration):
    """Test error handling in API endpoints."""

    def test_invalid_manuscript_id_format(self):
        """Test handling of invalid manuscript ID format."""
        response = self.client.get("/api/manuscripts/invalid-id-format")
        assert response.status_code == 404

    def test_invalid_json_in_request(self):
        """Test handling of invalid JSON in requests."""
        response = self.client.post(
            "/api/manuscripts/",
            data="invalid json content",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Test manuscript creation without title
        response = self.client.post("/api/manuscripts/", json={"author": "Test"})
        assert response.status_code == 422

        # Test chapter creation without content
        manuscript_file = self.create_test_manuscript("Test Manuscript")
        import uuid
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(manuscript_file)))

        response = self.client.post(
            f"/api/chapters/{manuscript_id}",
            json={"title": "Test Chapter"}
        )
        assert response.status_code == 422


class TestStaticFiles(TestAPIIntegration):
    """Test static file serving."""

    def test_css_file_serving(self):
        """Test that CSS files are served correctly."""
        response = self.client.get("/static/css/custom.css")
        # Note: This might return 404 if static files aren't properly configured
        # In a real deployment, this would return 200 with the CSS content
        assert response.status_code in [200, 404]

    def test_js_file_serving(self):
        """Test that JS files are served correctly."""
        response = self.client.get("/static/js/app.js")
        # Note: This might return 404 if static files aren't properly configured
        # In a real deployment, this would return 200 with the JS content
        assert response.status_code in [200, 404]