import sys
from pathlib import Path

# make the package importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
