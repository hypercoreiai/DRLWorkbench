import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
