"""
Validation tests to verify testing infrastructure is working correctly.
"""
import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.mark.unit
def test_pytest_works():
    """Basic test to verify pytest is functioning."""
    assert True


@pytest.mark.unit
def test_fixtures_available(temp_dir, sample_image, sample_batch, mock_config):
    """Test that all fixtures are available and working."""
    # Test temp_dir fixture
    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()
    
    # Test sample_image fixture
    assert isinstance(sample_image, torch.Tensor)
    assert sample_image.shape == (3, 224, 224)
    
    # Test sample_batch fixture
    assert isinstance(sample_batch, torch.Tensor)
    assert sample_batch.shape == (2, 3, 224, 224)
    
    # Test mock_config fixture
    assert hasattr(mock_config, 'IMAGE_MIN_DIM')
    assert mock_config.IMAGE_MIN_DIM == 800


@pytest.mark.unit
def test_torch_functionality():
    """Test that PyTorch is working correctly."""
    x = torch.randn(5, 5)
    y = torch.randn(5, 5)
    z = x + y
    assert z.shape == (5, 5)


@pytest.mark.unit
def test_numpy_functionality():
    """Test that NumPy is working correctly."""
    x = np.random.randn(5, 5)
    y = np.random.randn(5, 5)
    z = x + y
    assert z.shape == (5, 5)


@pytest.mark.unit
def test_device_fixture(device, cuda_available):
    """Test device-related fixtures."""
    assert isinstance(device, torch.device)
    assert isinstance(cuda_available, bool)


@pytest.mark.integration
def test_sample_dataset_item(sample_dataset_item):
    """Test sample dataset item fixture."""
    assert 'image' in sample_dataset_item
    assert 'masks' in sample_dataset_item
    assert 'class_ids' in sample_dataset_item
    assert 'bbox' in sample_dataset_item
    
    # Check shapes
    assert sample_dataset_item['image'].shape == (3, 224, 224)
    assert sample_dataset_item['masks'].shape == (5, 224, 224)
    assert len(sample_dataset_item['class_ids']) == 5
    assert sample_dataset_item['bbox'].shape == (5, 4)


@pytest.mark.unit
def test_mock_model_fixture(mock_model):
    """Test mock model fixture functionality."""
    assert hasattr(mock_model, 'eval')
    assert hasattr(mock_model, 'train')
    assert hasattr(mock_model, 'forward')
    
    # Test mock model call
    result = mock_model.forward()
    assert 'pred_masks' in result


@pytest.mark.slow
def test_slow_operation():
    """Test that slow marker works - this is just for testing markers."""
    import time
    time.sleep(0.1)  # Small delay to simulate slow test
    assert True


def test_temp_directory_cleanup(temp_dir):
    """Test that temporary directories are properly cleaned up."""
    test_file = temp_dir / "test_file.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    # Directory will be cleaned up automatically by fixture