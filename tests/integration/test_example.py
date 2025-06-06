"""
Example integration test to verify directory structure.
"""
import pytest


@pytest.mark.integration
def test_integration_directory_works():
    """Verify integration test directory is properly set up."""
    assert True


@pytest.mark.integration
def test_integration_example(sample_image, sample_masks):
    """Example integration test using fixtures."""
    # This would normally test integration between components
    assert sample_image.shape[0] == 3  # RGB channels
    assert len(sample_masks.shape) == 3  # batch, height, width