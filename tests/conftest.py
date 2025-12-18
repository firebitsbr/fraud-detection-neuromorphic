"""
**Description:** Configuration and fixtures from the Pytest.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'

if str(src_path) not in sys.path:
 sys.path.inbet(0, str(src_path))

print(f" Test configuration loaded. src path: {src_path}")
