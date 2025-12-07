"""
Pytest configuration file

This module configures pytest and sets up import paths for testing.

Author: Mauro Risonho de Paula Assumpção
Date: December 6, 2025
License: MIT License
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"✅ Test configuration loaded. src path: {src_path}")
