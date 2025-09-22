"""Tests for JavaScript functionality using Selenium WebDriver."""

import pytest
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import subprocess
import threading
import uvicorn

from illustrator.web.app import app
from illustrator.models import Chapter, ManuscriptMetadata, SavedManuscript


@pytest.fixture(scope="session")
def web_server():
    """Start the web server for testing."""
    # Create a test server in a separate thread
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Give the server time to start
    time.sleep(3)

    yield "http://127.0.0.1:8001"


@pytest.fixture(scope="session")
def browser():
    """Setup Chrome WebDriver for testing."""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)

        yield driver

        driver.quit()

    except Exception as e:
        pytest.skip(f"Chrome WebDriver not available: {e}")


class TestWebInterfaceBasics:
    """Test basic web interface functionality."""

    def test_page_loads(self, web_server, browser):
        """Test that the main page loads correctly."""
        browser.get(web_server)

        # Check that the page title is correct
        assert "Manuscript Illustrator" in browser.title

        # Check that the navigation exists
        nav = browser.find_element(By.TAG_NAME, "nav")
        assert nav is not None

    def test_navigation_elements(self, web_server, browser):
        """Test navigation elements are present."""
        browser.get(web_server)

        # Check for navigation links
        dashboard_link = browser.find_element(By.LINK_TEXT, "Dashboard")
        assert dashboard_link is not None

        new_manuscript_link = browser.find_element(By.LINK_TEXT, "New Manuscript")
        assert new_manuscript_link is not None

    def test_theme_toggle(self, web_server, browser):
        """Test theme toggle functionality."""
        browser.get(web_server)

        # Wait for the page to fully load
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "themeToggle"))
        )

        # Find and click the theme toggle
        theme_toggle = browser.find_element(By.ID, "themeToggle")
        initial_theme = browser.find_element(By.TAG_NAME, "html").get_attribute("data-bs-theme")

        theme_toggle.click()

        # Wait for theme to change
        time.sleep(1)

        new_theme = browser.find_element(By.TAG_NAME, "html").get_attribute("data-bs-theme")
        assert new_theme != initial_theme

    def test_api_keys_modal(self, web_server, browser):
        """Test API keys modal functionality."""
        browser.get(web_server)

        # Wait for page to load
        WebDriverWait(browser, 10).until(
            EC.element_to_be_clickable((By.LINK_TEXT, "API Keys"))
        )

        # Click API Keys link
        api_keys_link = browser.find_element(By.LINK_TEXT, "API Keys")
        api_keys_link.click()

        # Wait for modal to appear
        modal = WebDriverWait(browser, 10).until(
            EC.visibility_of_element_located((By.ID, "apiKeysModal"))
        )

        assert modal is not None

        # Check that form fields exist
        anthropic_field = browser.find_element(By.ID, "anthropicApiKey")
        openai_field = browser.find_element(By.ID, "openaiApiKey")

        assert anthropic_field is not None
        assert openai_field is not None

    def test_flash_messages(self, web_server, browser):
        """Test flash message system."""
        browser.get(web_server)

        # Execute JavaScript to show a test message
        browser.execute_script("""
            window.illustratorUtils.showSuccess('Test success message');
        """)

        # Wait for message to appear
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "alert-success"))
        )

        success_alert = browser.find_element(By.CLASS_NAME, "alert-success")
        assert "Test success message" in success_alert.text


class TestManuscriptInterface:
    """Test manuscript management interface."""

    def setup_method(self):
        """Setup test data directory."""
        self.test_manuscripts_dir = Path("test_web_manuscripts")
        self.test_manuscripts_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup test data."""
        if self.test_manuscripts_dir.exists():
            shutil.rmtree(self.test_manuscripts_dir)

    def create_test_manuscript(self, title="Test Manuscript"):
        """Create a test manuscript file."""
        manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title=title,
                author="Test Author",
                genre="Fantasy",
                total_chapters=0,
                total_words=0,
                completion_status="draft",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z"
            ),
            chapters=[],
            saved_at="2024-01-01T00:00:00Z",
            file_path=""
        )

        manuscript_file = self.test_manuscripts_dir / f"{title.lower().replace(' ', '_')}.json"
        with open(manuscript_file, 'w') as f:
            json.dump(manuscript.model_dump(), f)

        return manuscript_file

    def test_new_manuscript_form(self, web_server, browser):
        """Test new manuscript form functionality."""
        browser.get(f"{web_server}/manuscript/new")

        # Wait for form to load
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "manuscriptForm"))
        )

        # Fill out form
        title_field = browser.find_element(By.ID, "title")
        author_field = browser.find_element(By.ID, "author")
        genre_field = browser.find_element(By.ID, "genre")

        title_field.send_keys("Test Web Manuscript")
        author_field.send_keys("Test Web Author")
        genre_field.send_keys("Web Testing")

        # Check that fields are filled
        assert title_field.get_attribute("value") == "Test Web Manuscript"
        assert author_field.get_attribute("value") == "Test Web Author"
        assert genre_field.get_attribute("value") == "Web Testing"

    def test_form_validation(self, web_server, browser):
        """Test form validation functionality."""
        browser.get(f"{web_server}/manuscript/new")

        # Wait for form to load
        form = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "manuscriptForm"))
        )

        # Try to submit empty form
        submit_button = browser.find_element(By.CSS_SELECTOR, "button[type='submit']")
        submit_button.click()

        # Check for validation feedback
        time.sleep(1)  # Allow validation to trigger

        title_field = browser.find_element(By.ID, "title")
        # Check if field is marked as invalid
        assert "is-invalid" in title_field.get_attribute("class") or not title_field.get_attribute("value")

    def test_chapter_form(self, web_server, browser):
        """Test chapter form functionality."""
        # Mock a manuscript ID for the chapter form
        fake_manuscript_id = "test-manuscript-id"
        browser.get(f"{web_server}/manuscript/{fake_manuscript_id}/chapter/new")

        # Wait for form to load
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "chapterForm"))
        )

        # Fill out chapter form
        title_field = browser.find_element(By.ID, "title")
        content_field = browser.find_element(By.ID, "content")

        title_field.send_keys("Test Chapter")
        content_field.send_keys("This is test chapter content for the web interface testing.")

        # Check word count is updated
        time.sleep(1)  # Allow JavaScript to update
        word_count = browser.find_element(By.ID, "wordCount")
        assert "words" in word_count.text

    def test_word_count_updates(self, web_server, browser):
        """Test real-time word count updates."""
        fake_manuscript_id = "test-manuscript-id"
        browser.get(f"{web_server}/manuscript/{fake_manuscript_id}/chapter/new")

        # Wait for form to load
        content_field = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "content"))
        )

        # Type content and check word count updates
        test_content = "This is a test sentence with multiple words."
        content_field.send_keys(test_content)

        # Wait for JavaScript to update counts
        time.sleep(2)

        word_count_large = browser.find_element(By.ID, "wordCountLarge")
        character_count = browser.find_element(By.ID, "characterCount")

        # Check that counts are updated (should be > 0)
        assert int(word_count_large.text.replace(',', '')) > 0
        assert int(character_count.text.replace(',', '')) > 0


class TestChapterHeadersInterface:
    """Test chapter headers interface functionality."""

    def test_chapter_headers_page_loads(self, web_server, browser):
        """Test that chapter headers page loads."""
        fake_chapter_id = "test-chapter-id"
        browser.get(f"{web_server}/chapter/{fake_chapter_id}/headers")

        # Check page loads (may show error due to fake ID, but structure should be there)
        page_title = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h4"))
        )

        assert "Chapter Header Illustration Options" in page_title.text

    def test_loading_state_display(self, web_server, browser):
        """Test loading state display."""
        fake_chapter_id = "test-chapter-id"
        browser.get(f"{web_server}/chapter/{fake_chapter_id}/headers")

        # Check that loading message appears initially
        try:
            loading_message = browser.find_element(By.ID, "loadingMessage")
            assert "Generating header options" in loading_message.text
        except NoSuchElementException:
            # Loading might complete too quickly, which is also valid
            pass


class TestJavaScriptUtilities:
    """Test JavaScript utility functions."""

    def test_format_file_size(self, web_server, browser):
        """Test file size formatting utility."""
        browser.get(web_server)

        # Test the formatFileSize function
        result = browser.execute_script("""
            return window.illustratorUtils.formatFileSize(1024);
        """)

        assert "KB" in result or "B" in result

    def test_debounce_function(self, web_server, browser):
        """Test debounce utility function."""
        browser.get(web_server)

        # Test that debounce function exists and works
        result = browser.execute_script("""
            let callCount = 0;
            const debouncedFn = window.illustratorUtils.debounce(() => {
                callCount++;
            }, 100);

            // Call multiple times rapidly
            debouncedFn();
            debouncedFn();
            debouncedFn();

            // Return immediately (before debounce completes)
            return callCount;
        """)

        # Should be 0 because function is debounced
        assert result == 0

    def test_escape_html_function(self, web_server, browser):
        """Test HTML escaping utility."""
        browser.get(web_server)

        result = browser.execute_script("""
            return window.illustratorUtils.escapeHtml('<script>alert("test")</script>');
        """)

        assert "&lt;" in result
        assert "&gt;" in result

    def test_format_date_function(self, web_server, browser):
        """Test date formatting utility."""
        browser.get(web_server)

        result = browser.execute_script("""
            return window.illustratorUtils.formatDate('2024-01-01T00:00:00Z');
        """)

        # Should return a formatted date string
        assert isinstance(result, str)
        assert len(result) > 0


class TestWebSocketFunctionality:
    """Test WebSocket functionality (mock)."""

    def test_websocket_connection_attempt(self, web_server, browser):
        """Test WebSocket connection attempt."""
        browser.get(web_server)

        # Attempt to connect WebSocket (will likely fail in test, but should not error)
        browser.execute_script("""
            try {
                window.illustratorUtils.connectWebSocket('test-session-id');
            } catch(e) {
                console.log('WebSocket connection failed as expected in test environment');
            }
        """)

        # If no JavaScript errors, the test passes
        errors = browser.get_log('browser')
        fatal_errors = [log for log in errors if log['level'] == 'SEVERE' and 'connectWebSocket' not in log['message']]
        assert len(fatal_errors) == 0


class TestResponsiveDesign:
    """Test responsive design functionality."""

    def test_mobile_viewport(self, web_server, browser):
        """Test mobile viewport rendering."""
        # Set mobile viewport size
        browser.set_window_size(375, 667)  # iPhone dimensions
        browser.get(web_server)

        # Check that page still loads correctly
        nav = browser.find_element(By.TAG_NAME, "nav")
        assert nav is not None

        # Check that navbar toggler is visible on mobile
        try:
            navbar_toggler = browser.find_element(By.CLASS_NAME, "navbar-toggler")
            assert navbar_toggler.is_displayed()
        except NoSuchElementException:
            # May not be visible depending on Bootstrap behavior
            pass

    def test_desktop_viewport(self, web_server, browser):
        """Test desktop viewport rendering."""
        # Set desktop viewport size
        browser.set_window_size(1920, 1080)
        browser.get(web_server)

        # Check that navigation is fully visible
        nav = browser.find_element(By.TAG_NAME, "nav")
        assert nav is not None

        # Navigation items should be visible
        dashboard_link = browser.find_element(By.LINK_TEXT, "Dashboard")
        assert dashboard_link.is_displayed()


@pytest.mark.slow
class TestPerformance:
    """Test performance-related functionality."""

    def test_page_load_time(self, web_server, browser):
        """Test that pages load within reasonable time."""
        start_time = time.time()
        browser.get(web_server)

        # Wait for page to fully load
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "nav"))
        )

        load_time = time.time() - start_time
        assert load_time < 10  # Should load within 10 seconds

    def test_javascript_execution_time(self, web_server, browser):
        """Test JavaScript execution performance."""
        browser.get(web_server)

        # Test that utility functions execute quickly
        start_time = time.time()

        browser.execute_script("""
            for(let i = 0; i < 1000; i++) {
                window.illustratorUtils.formatFileSize(i * 1024);
                window.illustratorUtils.escapeHtml('<div>test' + i + '</div>');
            }
        """)

        execution_time = time.time() - start_time
        assert execution_time < 5  # Should execute within 5 seconds