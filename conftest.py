import sys
from pathlib import Path

# Get the project's root directory
root_dir = Path(__file__)

# Append the project's root directory to sys.path
sys.path.append(str(root_dir))