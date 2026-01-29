#!/usr/bin/env python3
"""
Test script for the data pipeline implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.api import get_data
from src.utils.logging import setup_logging
import yaml

def test_pipeline():
    """Test the data pipeline with a simple config."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting pipeline test...")
    
    # Load config
    config_path = "configs/test_pipeline.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run pipeline
    try:
        bundle = get_data(config, validate=True)
        
        # Check results
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"X_train shape: {bundle.X_train.shape if bundle.X_train is not None else 'None'}")
        logger.info(f"y_train shape: {bundle.y_train.shape if bundle.y_train is not None else 'None'}")
        logger.info(f"X_test shape: {bundle.X_test.shape if bundle.X_test is not None else 'None'}")
        logger.info(f"y_test shape: {bundle.y_test.shape if bundle.y_test is not None else 'None'}")
        logger.info(f"Validation report: {bundle.validation_report}")
        logger.info(f"Metadata keys: {list(bundle.metadata.keys())}")
        
        if bundle.error_log:
            logger.warning(f"Errors encountered: {bundle.error_log}")
        
        return True
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
