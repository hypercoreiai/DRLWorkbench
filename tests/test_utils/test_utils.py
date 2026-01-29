"""Tests for utils module."""

import pytest
from pathlib import Path
import logging

from drlworkbench.utils import (
    setup_logger,
    get_logger,
    Checkpoint,
    DRLWorkbenchError,
    BacktestError,
)


def test_setup_logger():
    """Test logger setup."""
    logger = setup_logger("test_logger", level=logging.DEBUG)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG


def test_get_logger():
    """Test getting existing logger."""
    logger = get_logger("test_logger_2")
    assert logger is not None
    assert isinstance(logger, logging.Logger)


def test_checkpoint_save_load(tmp_path):
    """Test checkpoint save and load."""
    checkpoint = Checkpoint(checkpoint_dir=tmp_path)
    
    # Save data
    data = {"key": "value", "number": 42}
    filepath = checkpoint.save(data, "test_checkpoint")
    
    assert filepath.exists()
    
    # Load data
    loaded_data = checkpoint.load("test_checkpoint")
    assert loaded_data == data


def test_checkpoint_exists(tmp_path):
    """Test checkpoint existence check."""
    checkpoint = Checkpoint(checkpoint_dir=tmp_path)
    
    assert not checkpoint.exists("nonexistent")
    
    data = {"test": "data"}
    checkpoint.save(data, "test")
    
    assert checkpoint.exists("test")


def test_checkpoint_delete(tmp_path):
    """Test checkpoint deletion."""
    checkpoint = Checkpoint(checkpoint_dir=tmp_path)
    
    data = {"test": "data"}
    checkpoint.save(data, "test")
    
    assert checkpoint.exists("test")
    
    deleted = checkpoint.delete("test")
    assert deleted
    assert not checkpoint.exists("test")


def test_checkpoint_list(tmp_path):
    """Test listing checkpoints."""
    checkpoint = Checkpoint(checkpoint_dir=tmp_path)
    
    checkpoint.save({"a": 1}, "checkpoint1")
    checkpoint.save({"b": 2}, "checkpoint2")
    checkpoint.save({"c": 3}, "checkpoint3", use_json=True)
    
    checkpoints = checkpoint.list_checkpoints()
    
    assert "checkpoint1" in checkpoints["pickle"]
    assert "checkpoint2" in checkpoints["pickle"]
    assert "checkpoint3" in checkpoints["json"]


def test_exceptions():
    """Test custom exceptions."""
    with pytest.raises(DRLWorkbenchError):
        raise DRLWorkbenchError("Test error")
    
    with pytest.raises(BacktestError):
        raise BacktestError("Test backtest error")
