"""
Shared pytest fixtures for the RealisHuman test suite.
"""
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = MagicMock()
    config.model_name = "test_model"
    config.batch_size = 1
    config.image_size = 224
    config.learning_rate = 1e-4
    config.device = "cpu"
    return config


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = MagicMock()
    model.eval.return_value = model
    model.train.return_value = model
    return model


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__.return_value = 10
    dataset.__getitem__.return_value = {
        'image': "mock_image_tensor",
        'label': "mock_label"
    }
    return dataset


@pytest.fixture
def mock_wandb():
    """Mock wandb for testing."""
    wandb_mock = MagicMock()
    wandb_mock.init.return_value = wandb_mock
    wandb_mock.log.return_value = None
    wandb_mock.finish.return_value = None
    return wandb_mock


@pytest.fixture
def sample_pose_data():
    """Sample pose data for testing."""
    return {
        'keypoints': [[10, 20, 0.9], [15, 25, 0.8]],  # Simple keypoint data
        'bbox': [10, 10, 100, 100],
        'person_id': 1
    }


@pytest.fixture
def mock_pipeline():
    """Mock diffusion pipeline for testing."""
    pipeline = MagicMock()
    pipeline.device = "cpu"
    pipeline.__call__.return_value = MagicMock(images=["mock_image"])
    return pipeline


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files."""
    return Path(__file__).parent / "data"


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )