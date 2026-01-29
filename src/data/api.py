# Data API with logging & metrics (V3 — PROJECT_OUTLINE Section 2.4)

from typing import Any, Dict, Optional

from .pipeline import DataBundle, DataPipeline
from src.utils.errors import APIError, DataValidationError
from src.utils.logging import setup_logging


def get_data(
    config: Dict[str, Any],
    validate: bool = True,
    log_path: Optional[str] = None,
) -> DataBundle:
    """
    Input: Config with all parameters.
    Output: DataBundle with X_train, y_train, X_test, y_test, metadata, validation_report.
    Logs all steps to log_path if provided.

    Args:
        config: Pipeline configuration.
        validate: Whether to run data validation steps.
        log_path: Optional directory for log files.

    Returns:
        DataBundle with train/test data and metadata.

    Raises:
        DataValidationError: If validation fails.
        APIError: On unrecoverable API failure (after fallbacks).
    """
    run_id = config.get("display", {}).get("run_id", "run")
    logger = setup_logging(log_path, run_id) if log_path else setup_logging()

    try:
        pipeline = DataPipeline(config)
        bundle = pipeline.run()
        if validate and bundle.validation_report:
            logger.info("Validation report: %s", bundle.validation_report)
        return bundle
    except DataValidationError as e:
        logger.error("Validation failed: %s", e)
        raise
    except APIError as e:
        logger.warning("API error, using cached data if available: %s", e)
        # Fallback logic can be implemented here
        raise
