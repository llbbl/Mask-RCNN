"""
Shared pytest fixtures and configuration for the Mask R-CNN project.
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def sample_batch():
    """Create a sample batch of images."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def sample_masks():
    """Create sample mask data."""
    return torch.randint(0, 2, (5, 224, 224)).float()


@pytest.fixture
def sample_bboxes():
    """Create sample bounding boxes."""
    return torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60], [30, 30, 70, 70]], dtype=torch.float32)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.eval = Mock()
    model.train = Mock()
    model.forward = Mock(return_value={'pred_masks': torch.randn(1, 5, 224, 224)})
    return model


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.IMAGE_MIN_DIM = 800
    config.IMAGE_MAX_DIM = 1024
    config.BATCH_SIZE = 1
    config.NUM_CLASSES = 81
    config.DETECTION_MIN_CONFIDENCE = 0.7
    config.DETECTION_NMS_THRESHOLD = 0.3
    return config


@pytest.fixture
def sample_numpy_image():
    """Create a sample numpy image for testing."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_dataset_item():
    """Create a sample dataset item."""
    return {
        'image': torch.randn(3, 224, 224),
        'masks': torch.randint(0, 2, (5, 224, 224)).float(),
        'class_ids': torch.tensor([1, 2, 3, 4, 5]),
        'bbox': torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60], [30, 30, 70, 70], [40, 40, 80, 80], [50, 50, 90, 90]], dtype=torch.float32)
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables and cleanup."""
    # Set test-specific environment variables
    original_env = os.environ.copy()
    os.environ['TESTING'] = '1'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def cuda_available():
    """Check if CUDA is available for testing."""
    return torch.cuda.is_available()