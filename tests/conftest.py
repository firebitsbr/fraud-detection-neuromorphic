"""
**Descrição:** Configuração e fixtures do Pytest.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'

if str(src_path) not in sys.path:
 sys.path.insert(0, str(src_path))

print(f" Test configuration loaded. src path: {src_path}")
