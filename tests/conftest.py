import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ('scripts', 'training'):
    sys.path.insert(0, str(ROOT / folder))
