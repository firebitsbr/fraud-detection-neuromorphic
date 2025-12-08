"""
Pytest configuration and fixtures.

Author: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
LinkedIn: linkedin.com/in/maurorisonho
GitHub: github.com/maurorisonho
Date: December 2025
License: MIT
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"✅ Test configuration loaded. src path: {src_path}")
