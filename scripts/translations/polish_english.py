#!/usr/bin/env python3
"""
Final polish for natural English
"""
import re
from pathlib import Path

# Natural English improvements
NATURAL_ENGLISH = {
    # Fix awkward phrasing
    r'\bLoading and preprocessing of datasets\b': 'Dataset Loading and Preprocessing',
    r'\bLoading and Preprocessing of Datasets\b': 'Dataset Loading and Preprocessing',
    r'\bfor Loading and preprocessing\b': 'for loading and preprocessing',
    r'\bfor training of SNNs\b': 'for SNN training',
    r'\bfor Training of SNNs\b': 'for SNN Training',
    r'\bPreprocessing of\b': 'Preprocessing',
    r'\bpreprocessing of\b': 'preprocessing',
    r'\bLoading of\b': 'Loading',
    r'\bloading of\b': 'loading',
    r'\bpreparation of data\b': 'data preparation',
    r'\bPreparation of Data\b': 'Data Preparation',
    r'\bof data for\b': 'of data for',
    
    # Fix date format
    r'\b(\d+)\s+December\s+of\s+(\d{4})\b': r'December \1, \2',
    r'\b(\d+)\s+January\s+of\s+(\d{4})\b': r'January \1, \2',
    r'\b(\d+)\s+February\s+of\s+(\d{4})\b': r'February \1, \2',
    r'\b(\d+)\s+March\s+of\s+(\d{4})\b': r'March \1, \2',
    r'\b(\d+)\s+April\s+of\s+(\d{4})\b': r'April \1, \2',
    r'\b(\d+)\s+May\s+of\s+(\d{4})\b': r'May \1, \2',
    r'\b(\d+)\s+June\s+of\s+(\d{4})\b': r'June \1, \2',
    r'\b(\d+)\s+July\s+of\s+(\d{4})\b': r'July \1, \2',
    r'\b(\d+)\s+August\s+of\s+(\d{4})\b': r'August \1, \2',
    r'\b(\d+)\s+September\s+of\s+(\d{4})\b': r'September \1, \2',
    r'\b(\d+)\s+October\s+of\s+(\d{4})\b': r'October \1, \2',
    r'\b(\d+)\s+November\s+of\s+(\d{4})\b': r'November \1, \2',
    
    # Fix "of" constructions
    r'\bUtilities for\b': 'Utilities for',
    r'\bof banking transactions\b': 'for banking transactions',
    r'\bof SNNs\b': 'for SNNs',
    
    # Fix development credits
    r'\bDevelopment by AI Assistida\b': 'AI-Assisted Development',
    r'\bby AI Assistida\b': 'AI-Assisted',
    r'\bAssisted by AI\b': 'AI-Assisted',
    
    # Fix structure phrases
    r'\bStructure of the Project\b': 'Project Structure',
    r'\bArchitecture of the System\b': 'System Architecture',
    r'\bof the system\b': 'of the system',
    r'\bof the project\b': 'of the project',
}

def fix_file(file_path):
    """Fix a single file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        
        for pattern, replacement in NATURAL_ENGLISH.items():
            content = re.sub(pattern, replacement, content)
        
        if content != original:
            file_path.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    print("=" * 80)
    print("NATURAL ENGLISH POLISH")
    print("=" * 80)
    print()
    
    patterns = ['**/*.md', '**/*.py']
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
    skip_files = {'fix_', 'translate_', 'cleanup_', 'complete_', 'final_', 'polish_'}
    
    count = 0
    base = Path('.')
    
    for pattern in patterns:
        for file_path in base.glob(pattern):
            # Skip directories
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            # Skip translation/fix scripts
            if any(skip in file_path.name for skip in skip_files):
                continue
            
            if file_path.is_file():
                if fix_file(file_path):
                    print(f"✓ {file_path}")
                    count += 1
    
    print()
    print("=" * 80)
    print(f"COMPLETE! Polished {count} files.")
    print("=" * 80)

if __name__ == '__main__':
    main()
