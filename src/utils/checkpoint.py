# Save/load checkpoints for pipeline resumption (V3 — PROJECT_OUTLINE Section 9.4)

from pathlib import Path
from typing import Any, Dict


def save_checkpoint(state_dict: Dict[str, Any], path: str) -> None:
    """
    Save pipeline state (data, models, backtest results) for resumption.

    Args:
        state_dict: Dictionary of state to persist.
        path: File path to write (e.g. .pkl or .joblib).
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    suffix = path_obj.suffix.lower()
    if suffix == ".pkl":
        import pickle
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)
    else:
        try:
            import joblib
            joblib.dump(state_dict, path)
        except ImportError:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(state_dict, f)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load pipeline state from checkpoint.

    Args:
        path: File path to checkpoint.

    Returns:
        State dictionary for resuming pipeline.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    suffix = path_obj.suffix.lower()
    if suffix == ".pkl":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    try:
        import joblib
        return joblib.load(path)
    except ImportError:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
