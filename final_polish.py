#!/usr/bin/env python3
"""
Final polish - fix remaining issues and improve quality.
"""

import re
from pathlib import Path

# Final fixes
FINAL_FIXES = {
    # Fix redundancies
    "Efficiency extremely efficient": "Extremely efficient",
    "Detection instantaneous": "Instantaneous detection",
    "Learning continuous": "Continuous learning",
    
    # Fix wrong order
    "IA Traditional": "Traditional AI",
    "The Problem with IA Traditional": "The Problem with Traditional AI",
    "How They Work SNNs": "How SNNs Work",
    "Neuron traditional": "Traditional neuron",
    
    # Fix remaining Spanish/Portuguese
    "Ultra-eficientes": "Ultra-efficient",
    "Processing asynchronous": "Asynchronous processing",
    "Processing native temporal": "Native temporal processing",
    "Latency high": "High latency",
    
    # Fix grammar
    "work as o human brain": "work like the human brain",
    "through of temporal": "through temporal",
    
    # Fix throughput typo
    "Throrghput": "Throughput",
    
    # Remaining Portuguese words
    r'\btodos\b': 'all',
    r'\bTodos\b': 'All',
    r'\bestá\b': 'is',
    r'\bEstá\b': 'Is',
    r'\bnão\b': 'not',
    r'\bNão\b': 'Not',
    r'\bsão\b': 'are',
    r'\bSão\b': 'Are',
}

def final_polish(text: str) -> str:
    """Apply final polish to translations."""
    result = text
    
    # Direct replacements (longer first)
    sorted_fixes = sorted(
        [(k, v) for k, v in FINAL_FIXES.items() if not k.startswith(r'\b')],
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    for wrong, correct in sorted_fixes:
        result = result.replace(wrong, correct)
    
    # Regex replacements
    regex_fixes = [(k, v) for k, v in FINAL_FIXES.items() if k.startswith(r'\b')]
    for pattern, replacement in regex_fixes:
        result = re.sub(pattern, replacement, result)
    
    return result

def process_file(file_path: Path) -> bool:
    """Process a file."""
    if 'translate' in file_path.name:
        return False
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        polished = final_polish(content)
        
        if polished != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(polished)
            print(f"✓ {file_path.name}")
            return True
        return False
            
    except Exception as e:
        print(f"✗ {file_path.name}: {e}")
        return False

def main():
    """Main function."""
    project_root = Path.cwd()
    
    print("=" * 80)
    print("Final Polish")
    print("=" * 80)
    print()
    
    fixed_count = 0
    skip_dirs = {".venv", "venv", ".git", "__pycache__", "node_modules", ".conda", "data"}
    
    for file_path in project_root.glob("**/*.md"):
        if any(skip in file_path.parts for skip in skip_dirs):
            continue
        if process_file(file_path):
            fixed_count += 1
    
    print()
    print("=" * 80)
    print(f"Polished {fixed_count} files.")
    print("=" * 80)

if __name__ == "__main__":
    main()
