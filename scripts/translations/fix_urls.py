#!/usr/bin/env python3
"""
Fix over-translations in URLs and specific terms
"""
import re
from pathlib import Path

# Fix over-translated URLs and specific terms
URL_FIXES = {
    # Fix email and URL domains
    r'@gmail\.com\b': '@gmail.com',
    r'linkedin\.com': 'linkedin.com',
    r'github\.com': 'github.com',
    r'\.com/': '.com/',
    r'\.com\b': '.com',
    
    # Fix other over-translations
    r'\bwith/': 'com/',
    r'\bwhypatibility\b': 'compatibility',
    r'\bwithtent\b': 'content',
    r'\bwithmon\b': 'common',
    r'\bwithmand\b': 'command',
    r'\bwithment\b': 'comment',
    r'\bwithpose\b': 'compose',
    r'\bwithpare\b': 'compare',
    r'\bwithplete\b': 'complete',
    r'\bwithplex\b': 'complex',
    r'\bwithpute\b': 'compute',
    r'\bwithpatible\b': 'compatible',
    r'\bwithponents\b': 'components',
    r'\bwithpression\b': 'compression',
    r'\bwithpiled\b': 'compiled',
    r'\bwithpiler\b': 'compiler',
    r'\bwithpatibility\b': 'compatibility',
    r'\bwithprehensive\b': 'comprehensive',
    
    # Fix specific broken words
    r'\btoward\b': 'to',
    r'\bToward\b': 'To',
    r'\busing to\b': 'using the',
    r'\bUsing to\b': 'Using the',
    r'\bof to\b': 'of the',
    r'\bOf to\b': 'Of the',
    r'\bfor to\b': 'for the',
    r'\bFor to\b': 'For the',
    r'\bin to\b': 'in the',
    r'\bIn to\b': 'In the',
}

def fix_file(file_path):
    """Fix a single file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        
        for pattern, replacement in URL_FIXES.items():
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
    print("URL AND OVER-TRANSLATION FIXES")
    print("=" * 80)
    print()
    
    patterns = ['**/*.md', '**/*.py']
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
    
    count = 0
    base = Path('.')
    
    for pattern in patterns:
        for file_path in base.glob(pattern):
            # Skip directories
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            if file_path.is_file():
                if fix_file(file_path):
                    print(f"✓ {file_path}")
                    count += 1
    
    print()
    print("=" * 80)
    print(f"COMPLETE! Fixed {count} files.")
    print("=" * 80)

if __name__ == '__main__':
    main()
