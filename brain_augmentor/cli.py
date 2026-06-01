"""Entry point for the brain-generate command installed by pyproject.toml."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate import main

if __name__ == "__main__":
    main()
