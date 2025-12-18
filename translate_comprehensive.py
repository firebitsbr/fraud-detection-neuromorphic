#!/usr/bin/env python3
"""
Comprehensive final translation - handles all remaining Portuguese.
"""

import re
from pathlib import Path

# Complete translations including all variations
ALL_TRANSLATIONS = {
    # Headers and titles
    "Por Que": "Why",
    "Este Projeto é Importante": "This Project is Important",
    "O Problem": "The Problem",
    "As Funcionam": "How They Work",
    
    # Table headers
    "Diferenciais": "Differentials",
    "Valor": "Value",
    "Comparação": "Comparison",
    "Latência": "Latency",
    "Consumo Energético": "Energy Consumption",
    "Acurácia": "Accuracy",
    "Potência": "Power",
    
    # Words in sentences
    "Este Projeto": "This Project",
    "é Importante": "is Important",
    "Detecção": "Detection",
    "Eficiência": "Efficiency",
    "Aprendizado": "Learning",
    "instant": "instantaneous",
    "extreme": "extremely efficient",
    "continuous": "continuous",
    "Processamento": "Processing",
    "asynchronous": "asynchronous",
    "ultra-low": "ultra-low",
    "temporal": "temporal",
    "native": "native",
    "traditional": "traditional",
    "Neurônio": "Neuron",
    "Processa TEMPO": "Processes TIME",
    "alta": "high",
    "energia": "energy",
    "exploit": "exploit",
    "temporality": "temporality",
    
    # Fix corrupted translations
    "withputação": "Computing",
    "neuromórstays": "neuromorphic",
    "sistema": "system",
    "Sistema": "System",
    "pulifs": "pulses",
    "As ": "How ",
    
    # Anchor links in Portuguese
    "#-instalação-rápida-docker": "#-quick-start-docker",
    "#-instalação-manual-passo-a-passo": "#-manual-installation-step-by-step",
    "#-executando-os-notebooks": "#-running-the-notebooks",
    "#-using-a-api-rest": "#-using-the-api-rest",
    "#-testes-e-validação": "#-tests-and-validation",
    "#-resultados-e-benchmarks": "#-results-and-benchmarks",
    "#-documentação-detalhada": "#-detailed-documentation",
    "#-estrutura-do-projeto": "#-structure-of-the-project",
    "#-tecnologias": "#-technologies",
    "#-contribuindo": "#-contributing",
    "#-referências": "#-references",
    
    # Common Portuguese words with word boundaries
    r'\bestá\b': 'is',
    r'\bsão\b': 'are',
    r'\bnão\b': 'not',
    r'\bNão\b': 'Not',
    r'\bcomo\b': 'as',
    r'\bComo\b': 'How',
    r'\bpara\b': 'for',
    r'\bPara\b': 'For',
    r'\bsobre\b': 'about',
    r'\bSobre\b': 'About',
    r'\bentre\b': 'between',
    r'\bEntre\b': 'Between',
    r'\btodos\b': 'all',
    r'\bTodos\b': 'All',
    r'\btodas\b': 'all',
    r'\bTodas\b': 'All',
    r'\bmais\b': 'more',
    r'\bMais\b': 'More',
    r'\bmenos\b': 'less',
    r'\bMenos\b': 'Less',
}

def translate_comprehensive(text: str) -> str:
    """Comprehensive translation."""
    result = text
    
    # First do direct string replacements (order matters - longer first)
    sorted_direct = sorted(
        [(k, v) for k, v in ALL_TRANSLATIONS.items() if not k.startswith(r'\b')],
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    for pt, en in sorted_direct:
        result = result.replace(pt, en)
    
    # Then do regex replacements for word boundaries
    regex_patterns = [(k, v) for k, v in ALL_TRANSLATIONS.items() if k.startswith(r'\b')]
    
    for pattern, replacement in regex_patterns:
        result = re.sub(pattern, replacement, result)
    
    return result

def process_markdown_file(file_path: Path) -> bool:
    """Process markdown files with comprehensive translation."""
    if 'translate' in file_path.name:
        return False
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        translated = translate_comprehensive(content)
        
        if translated != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated)
            print(f"✓ {file_path.relative_to(Path.cwd())}")
            return True
        return False
            
    except Exception as e:
        print(f"✗ Error: {file_path.name} - {e}")
        return False

def main():
    """Main function."""
    project_root = Path.cwd()
    
    print("=" * 80)
    print("Comprehensive Final Translation")
    print("=" * 80)
    print()
    
    fixed_count = 0
    
    # Process all markdown files
    for file_path in project_root.glob("**/*.md"):
        skip_dirs = {".venv", "venv", ".git", "__pycache__", "node_modules", ".conda", "data"}
        if any(skip in file_path.parts for skip in skip_dirs):
            continue
            
        if process_markdown_file(file_path):
            fixed_count += 1
    
    print()
    print("=" * 80)
    print(f"Translated {fixed_count} files.")
    print("=" * 80)

if __name__ == "__main__":
    main()
