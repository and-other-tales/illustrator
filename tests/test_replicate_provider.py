"""Tests for Replicate image provider utilities."""

from illustrator.providers import ReplicateImageProvider
from replicate.file import File


class _DummyReplicateProvider(ReplicateImageProvider):
    def get_provider_type(self):  # pragma: no cover - not used in test
        return None


def test_extract_image_urls_handles_replicate_file_objects():
    provider = object.__new__(_DummyReplicateProvider)
    file_obj = File(
        id="file-id",
        name="preview.png",
        content_type="image/png",
        size=1234,
        etag="etag",
        checksums={"md5": "deadbeef"},
        metadata={},
        created_at="2024-01-01T00:00:00Z",
        expires_at=None,
        urls={"get": "https://example.com/preview.png"},
    )

    urls = provider._extract_image_urls(file_obj)

    assert urls == ["https://example.com/preview.png"]
