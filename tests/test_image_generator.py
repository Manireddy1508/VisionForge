"""
Tests for the image generator module.
"""
import pytest
from src.image_generator import ImageGenerator
from unittest.mock import patch

def test_image_generator_initialization():
    """Test ImageGenerator initialization."""
    generator = ImageGenerator()
    assert generator.model_name == "dall-e-3"
    assert generator.temp_dir is not None

@patch('src.image_generator.openai')
def test_generate_images(mock_openai, mock_openai_response):
    """Test image generation."""
    generator = ImageGenerator()
    result = generator.generate_images(
        prompt="test prompt",
        num_images=1
    )
    assert isinstance(result, list)
    assert len(result) > 0
    assert "image" in result[0]
    assert "prompt" in result[0]

@patch('src.image_generator.openai')
def test_edit_images(mock_openai, test_image):
    """Test image editing."""
    generator = ImageGenerator()
    result = generator.edit_images(
        images=[test_image],
        prompt="test edit prompt"
    )
    assert isinstance(result, list)
    assert len(result) > 0
    assert "image" in result[0]
    assert "prompt" in result[0]

def test_cleanup_temp_files():
    """Test temporary file cleanup."""
    generator = ImageGenerator()
    generator.cleanup_temp_files()  # Should not raise any errors 