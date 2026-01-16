#!/usr/bin/env python3
"""
Final comprehensive translation using word boundaries to avoid partial matches.
"""

import re
from pathlib import Path

# Complete word translations (using regex word boundaries)
WORD_TRANSLATIONS = {
    # Portuguese words that should be translated as complete words only
    r'\btodos\b': 'all',
    r'\bTodos\b': 'All',
    r'\btodas\b': 'all',
    r'\bTodas\b': 'All',
    r'\bmais\b': 'more',
    r'\bMais\b': 'More',
    r'\bmenos\b': 'less',
    r'\bMenos\b': 'Less',
    r'\bcomo\b': 'as',
    r'\bComo\b': 'As',
    r'\bpara\b': 'for',
    r'\bPara\b': 'For',
    r'\bsobre\b': 'about',
    r'\bSobre\b': 'About',
    r'\bentre\b': 'between',
    r'\bEntre\b': 'Between',
    r'\bantes\b': 'before',
    r'\bAntes\b': 'Before',
    r'\bdepois\b': 'after',
    r'\bDepois\b': 'After',
    r'\bmesmo\b': 'same',
    r'\bMesmo\b': 'Same',
    r'\bmétodos\b': 'methods',
    r'\bmétodo\b': 'method',
    r'\bserviços\b': 'services',
    r'\bserviço\b': 'service',
    r'\bcomandos\b': 'commands',
    r'\bcomando\b': 'command',
    r'\bpacotes\b': 'packages',
    r'\bpacote\b': 'package',
    r'\bversões\b': 'versions',
    r'\bversão\b': 'version',
    r'\bfixas\b': 'fixed',
    r'\bfixa\b': 'fixed',
    r'\bLista\b': 'List',
    r'\blista\b': 'list',
    r'\bInicie\b': 'Start',
    r'\binicie\b': 'start',
    r'\bIniciar\b': 'Start',
    r'\biniciar\b': 'start',
    r'\bParar\b': 'Stop',
    r'\bparar\b': 'stop',
    r'\bListar\b': 'List',
    r'\blistar\b': 'list',
    r'\bReprodutível\b': 'Reproducible',
    r'\breprodutível\b': 'reproducible',
    r'\bPesquisar\b': 'Research',
    r'\bpesquisar\b': 'research',
}

# Phrase translations (these don't need word boundaries)
PHRASE_TRANSLATIONS = {
    "Comparação:": "Comparison:",
    "Comparação of": "Comparison of",
    "Comparação with": "Comparison with",
    "Codistaysção": "Codificação",  # Fix corrupted word
    "Inhaveface": "Interface",  # Fix corrupted word
    "beviços": "services",  # Fix corrupted word
    "withandos": "commands",  # Fix corrupted word
    "withtor": "comparator",  # Fix corrupted word
    
    # Common patterns
    "of todos os": "of all",
    "for todos": "for all",
    "with todos os": "with all",
    "Lista todos": "List all",
    "Inicie todos": "Start all",
    "Parar todos": "Stop all",
    "Health check of todos os": "Health check of all",
    "Mesmo environment for": "Same environment for",
    
    # Specific fixes
    "Versões fixas of": "Fixed versions of",
    "métodos of": "methods of",
    "Pesquisar métodos of explicabilidade for": "Research explainability methods for",
}

def translate_text(text: str) -> str:
    """Translate text using word boundaries for accuracy."""
    result = text
    
    # First apply phrase translations (more specific)
    for pt, en in PHRASE_TRANSLATIONS.items():
        result = result.replace(pt, en)
    
    # Then apply word-boundary translations
    for pattern, replacement in WORD_TRANSLATIONS.items():
        result = re.sub(pattern, replacement, result)
    
    return result

def process_file(file_path: Path) -> bool:
    """Process a file with intelligent translation."""
    # Skip translation scripts
    if 'translate' in file_path.name:
        return False
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        translated = translate_text(content)
        
        if translated != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated)
            print(f"✓ Translated: {file_path.relative_to(Path.cwd())}")
            return True
        return False
            
    except Exception as e:
        print(f"✗ Error: {file_path.name} - {e}")
        return False

def main():
    """Main function."""
    project_root = Path.cwd()
    
    print("=" * 80)
    print("Final Translation - Word Boundary Based")
    print("=" * 80)
    print()
    
    patterns = ["**/*.md", "**/*.py"]
    skip_dirs = {".venv", "venv", ".git", "__pycache__", "node_modules", ".conda", "data"}
    
    fixed_count = 0
    
    for pattern in patterns:
        for file_path in project_root.glob(pattern):
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
                
            if process_file(file_path):
                fixed_count += 1
    
    print()
    print("=" * 80)
    print(f"Translation complete! Translated {fixed_count} files.")
    print("=" * 80)

if __name__ == "__main__":
    main()
