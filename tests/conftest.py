"""
Common test fixtures and configuration.
"""
import os
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_openai():
    """Mock OpenAI API responses."""
    mock = MagicMock()
    mock.images.generate.return_value = MagicMock(
        data=[MagicMock(url="/Users/mani/cloud/cloud-gen-project/car1.webp")]
    )
    return mock

@pytest.fixture
def mock_milvus():
    """Mock Milvus connection and operations."""
    mock = MagicMock()
    mock.search.return_value = []
    mock.insert.return_value = [1]  # Mock inserted ID
    return mock

@pytest.fixture
def test_image():
    """Create a test image for testing."""
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    return img

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    os.environ['OPENAI_API_KEY'] = 'test-key'
    os.environ['MILVUS_HOST'] = 'localhost'
    os.environ['MILVUS_PORT'] = '19530'
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'test-project'
    os.environ['GCS_BUCKET_NAME'] = 'test-bucket' 