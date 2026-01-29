"""Checkpointing utilities for resuming long-running operations."""

import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Checkpoint:
    """
    Checkpoint manager for saving and loading state.
    
    Supports both pickle (for Python objects) and JSON (for serializable data).
    """
    
    def __init__(self, checkpoint_dir: Path = Path("checkpoints")):
        """
        Initialize checkpoint manager.
        
        Parameters
        ----------
        checkpoint_dir : Path, default Path("checkpoints")
            Directory to store checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        data: Any,
        name: str,
        use_json: bool = False
    ) -> Path:
        """
        Save checkpoint data.
        
        Parameters
        ----------
        data : Any
            Data to save (must be picklable or JSON-serializable).
        name : str
            Name of the checkpoint (without extension).
        use_json : bool, default False
            If True, save as JSON instead of pickle.
            
        Returns
        -------
        Path
            Path to the saved checkpoint file.
        """
        if use_json:
            filepath = self.checkpoint_dir / f"{name}.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        else:
            filepath = self.checkpoint_dir / f"{name}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        
        logger.info(f"Checkpoint saved: {filepath}")
        return filepath
    
    def load(
        self,
        name: str,
        use_json: bool = False
    ) -> Optional[Any]:
        """
        Load checkpoint data.
        
        Parameters
        ----------
        name : str
            Name of the checkpoint (without extension).
        use_json : bool, default False
            If True, load from JSON instead of pickle.
            
        Returns
        -------
        Any or None
            Loaded data, or None if checkpoint doesn't exist.
        """
        if use_json:
            filepath = self.checkpoint_dir / f"{name}.json"
            if not filepath.exists():
                logger.warning(f"Checkpoint not found: {filepath}")
                return None
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            filepath = self.checkpoint_dir / f"{name}.pkl"
            if not filepath.exists():
                logger.warning(f"Checkpoint not found: {filepath}")
                return None
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        
        logger.info(f"Checkpoint loaded: {filepath}")
        return data
    
    def exists(self, name: str, use_json: bool = False) -> bool:
        """
        Check if a checkpoint exists.
        
        Parameters
        ----------
        name : str
            Name of the checkpoint (without extension).
        use_json : bool, default False
            If True, check for JSON file instead of pickle.
            
        Returns
        -------
        bool
            True if checkpoint exists, False otherwise.
        """
        ext = ".json" if use_json else ".pkl"
        filepath = self.checkpoint_dir / f"{name}{ext}"
        return filepath.exists()
    
    def delete(self, name: str, use_json: bool = False) -> bool:
        """
        Delete a checkpoint.
        
        Parameters
        ----------
        name : str
            Name of the checkpoint (without extension).
        use_json : bool, default False
            If True, delete JSON file instead of pickle.
            
        Returns
        -------
        bool
            True if checkpoint was deleted, False if it didn't exist.
        """
        ext = ".json" if use_json else ".pkl"
        filepath = self.checkpoint_dir / f"{name}{ext}"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Checkpoint deleted: {filepath}")
            return True
        return False
    
    def list_checkpoints(self) -> Dict[str, list]:
        """
        List all available checkpoints.
        
        Returns
        -------
        Dict[str, list]
            Dictionary with 'pickle' and 'json' keys containing lists of checkpoint names.
        """
        pickle_files = [f.stem for f in self.checkpoint_dir.glob("*.pkl")]
        json_files = [f.stem for f in self.checkpoint_dir.glob("*.json")]
        return {
            "pickle": sorted(pickle_files),
            "json": sorted(json_files)
        }
