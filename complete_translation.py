#!/usr/bin/env python3
"""
Complete and final translation - removes ALL Portuguese from the project.
"""

import re
from pathlib import Path

# Comprehensive translation dictionary
COMPLETE_TRANSLATIONS = {
    # Common Portuguese words - using word boundaries
    r'\bneurônios\b': 'neurons',
    r'\bneurônio\b': 'neuron',
    r'\btransação\b': 'transaction',
    r'\btransações\b': 'transactions',
    r'\bcodificação\b': 'encoding',
    r'\busuário\b': 'user',
    r'\barquivo\b': 'file',
    r'\blegítima\b': 'legitimate',
    r'\blegítimos\b': 'legitimate',
    r'\bfraudulenta\b': 'fraudulent',
    r'\bfraudulentas\b': 'fraudulent',
    r'\bsimulação\b': 'simulation',
    r'\bconectados\b': 'connected',
    r'\bconectadas\b': 'connected',
    r'\bdispara\b': 'fires',
    r'\bdisparando\b': 'firing',
    r'\bdistom\b': 'fire',
    r'\bdisto\b': 'fired',
    r'\bsaturação\b': 'saturation',
    r'\bútima\b': 'last',
    r'\bdesde\b': 'since',
    r'\banômalo\b': 'anomalous',
    r'\banômalos\b': 'anomalous',
    r'\bpadrão\b': 'pattern',
    r'\bpadrões\b': 'patterns',
    r'\bentre\b': 'between',
    r'\bpara\b': 'for',
    r'\bsobre\b': 'about',
    r'\bcomo\b': 'as',
    r'\btodos\b': 'all',
    r'\btodas\b': 'all',
    r'\bde\b': 'of',
    r'\bda\b': 'of the',
    r'\bdo\b': 'of the',
    r'\bdos\b': 'of the',
    r'\bdas\b': 'of the',
    r'\bem\b': 'in',
    r'\bna\b': 'in the',
    r'\bno\b': 'in the',
    r'\bnos\b': 'in the',
    r'\bnas\b': 'in the',
    r'\bcom\b': 'with',
    r'\bsem\b': 'without',
    r'\bpor\b': 'by',
    r'\bpela\b': 'by the',
    r'\bpelo\b': 'by the',
    r'\bpelos\b': 'by the',
    r'\bpelas\b': 'by the',
    r'\bantes\b': 'before',
    r'\bdepois\b': 'after',
    r'\bdesde\b': 'since',
    r'\baté\b': 'until',
    r'\benquanto\b': 'while',
    r'\bquando\b': 'when',
    r'\bonde\b': 'where',
    r'\bporque\b': 'because',
    r'\bse\b': 'if',
    r'\bmas\b': 'but',
    r'\btambém\b': 'also',
    r'\bainda\b': 'still',
    r'\bsempre\b': 'always',
    r'\bnunca\b': 'never',
    r'\bmais\b': 'more',
    r'\bmenos\b': 'less',
    r'\bmuito\b': 'very',
    r'\bpouco\b': 'little',
    r'\bmétodos\b': 'methods',
    r'\bmétodo\b': 'method',
    r'\bsaída\b': 'output',
    r'\bentrada\b': 'input',
    r'\bnível\b': 'level',
    r'\batual\b': 'current',
    r'\batualização\b': 'update',
    r'\benvia\b': 'sends',
    r'\brecebe\b': 'receives',
    r'\brecewell\b': 'receive',
    r'\bprocessa\b': 'processes',
    r'\bavaliação\b': 'evaluation',
    r'\bútima\b': 'last',
    r'\bprimeiro\b': 'first',
    r'\bsegundo\b': 'second',
    r'\bpróximo\b': 'next',
    r'\bforça\b': 'strength',
    r'\benfraquece\b': 'weakens',
    r'\bintensidades\b': 'intensities',
    r'\bvelocidade\b': 'speed',
    r'\btempo\b': 'time',
    r'\bmáximo\b': 'maximum',
    r'\bmínimo\b': 'minimum',
    r'\bmúltiplos\b': 'multiple',
    r'\bjuntos\b': 'together',
    
    # Verbs
    r'\btem\b': 'has',
    r'\btêm\b': 'have',
    r'\bsão\b': 'are',
    r'\bestá\b': 'is',
    r'\bestão\b': 'are',
    r'\bfoi\b': 'was',
    r'\bforam\b': 'were',
    r'\bserá\b': 'will be',
    r'\bserão\b': 'will be',
    r'\bseria\b': 'would be',
    r'\bseriam\b': 'would be',
    r'\btinha\b': 'had',
    r'\btinham\b': 'had',
    r'\bnão\b': 'not',
    r'\bsim\b': 'yes',
    
    # Phrases and expressions
    "simulation of": "Simulation of",
    "Comparison of": "Comparison of",
    "comparison with": "comparison with",
    "Ler transaction": "Read transaction",
    "Create transaction": "Create transaction",
    "Status of the": "Status of the",
    "Value of the": "Value of the",
    "file:": "File:",
    "output:": "Output:",
    "input:": "Input:",
    "Objective:": "Objective:",
    "Problem:": "Problem:",
    "Responsibility:": "Responsibility:",
    "Seconds since": "Seconds since",
    "Latency/transaction": "Latency/transaction",
    "simulator": "simulator",
    "similar": "similar",
    "similar": "similar",
    "Block": "Block",
    "ModelComparator": "ModelComparator",
    "comparator": "comparator",
    "time": "time",
    "caused": "caused",
    "Weakens": "Weakens",
    "Strengthens": "Strengthens",
    "have limitations": "Have Limitations",
    "Why methods Tradicionais": "Why Traditional Methods",
    "Average of": "Average of",
    "History of the": "History of the",
    "Última transaction": "Last transaction",
    "hours ago": "hours ago",
    "Ativa neurons": "Activates neurons",
    "Overlap between": "Overlap between",
    "neurons vizinhos": "neighboring neurons",
    "multiple neurons": "Multiple neurons",
    "pattern anomalous": "Anomalous pattern",
    "spike train": "spike train",
    "more abstract": "more abstract",
    "encoded": "encoded",
    "compose": "compose",
    "self": "self",
}

def translate_content(text: str) -> str:
    """Apply comprehensive translation."""
    result = text
    
    # Apply regex word boundary replacements
    for pattern, replacement in COMPLETE_TRANSLATIONS.items():
        if pattern.startswith(r'\b'):
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        else:
            # Direct string replacement for phrases
            result = result.replace(pattern, replacement)
    
    return result

def process_file(file_path: Path) -> bool:
    """Process a single file."""
    if 'translate' in file_path.name or 'TRANSLATION_REPORT' in file_path.name:
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        translated = translate_content(content)
        
        if translated != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated)
            return True
        return False
    except Exception as e:
        print(f"Error: {file_path} - {e}")
        return False

def main():
    """Main translation function."""
    root = Path.cwd()
    skip_dirs = {".venv", "venv", ".git", "__pycache__", "node_modules", ".conda", "data", ".ipynb_checkpoints"}
    
    print("=" * 80)
    print("COMPLETE FINAL TRANSLATION")
    print("=" * 80)
    print()
    
    total_fixed = 0
    
    # Process .md and .py files
    for pattern in ["**/*.md", "**/*.py"]:
        for file_path in root.glob(pattern):
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            if process_file(file_path):
                print(f"✓ {file_path.relative_to(root)}")
                total_fixed += 1
    
    print()
    print("=" * 80)
    print(f"COMPLETE! Translated {total_fixed} files.")
    print("=" * 80)

if __name__ == "__main__":
    main()
