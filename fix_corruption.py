#!/usr/bin/env python3
"""
Fix all remaining corrupted translations
"""
import re
from pathlib import Path

# Comprehensive fixes for all remaining corruption
FINAL_FIXES = {
    # Fix corrupted words from over-translation
    r'\bDataift\b': 'Dataset',
    r'\bdataift\b': 'dataset',
    r'\bdataifts\b': 'datasets',
    r'\bDeifnvolvimento\b': 'Development',
    r'\bdeifnvolvimento\b': 'development',
    r'\byesilar\b': 'similar',
    r'\bYesilar\b': 'Similar',
    r'\bSephasber\b': 'September',
    r'\bsephasber\b': 'september',
    r'\bDecembro\b': 'December',
    r'\bdecembro\b': 'december',
    r'\brethatst\b': 'request',
    r'\bRethatst\b': 'Request',
    r'\biflection\b': 'selection',
    r'\bIflection\b': 'Selection',
    r'\bmodel_iflection\b': 'model_selection',
    
    # Fix dates and months
    r'\bjaneiro\b': 'January',
    r'\bfevereiro\b': 'February',
    r'\bmarço\b': 'March',
    r'\babril\b': 'April',
    r'\bmaio\b': 'May',
    r'\bjunho\b': 'June',
    r'\bjulho\b': 'July',
    r'\bagosto\b': 'August',
    r'\bsetembro\b': 'September',
    r'\boutubro\b': 'October',
    r'\bnovembro\b': 'November',
    r'\bdezembro\b': 'December',
    
    # Fix "of" over-translations
    r'\b(\d+)\s+of\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+of\s+(\d{4})\b': r'\1 \2 \3',
    r'\bof Paula\b': 'de Paula',
    r'\bof Dezembro\b': 'December',
    r'\bof Janeiro\b': 'January',
    
    # Fix prepositions
    r'\bfor AI\b': 'by AI',
    r'\bfor IA\b': 'by AI',
    r'\bassisted for\b': 'assisted by',
    r'\bAssisted for\b': 'Assisted by',
    
    # Fix "to" over-translations
    r'\bto API\b': 'the API',
    r'\bto SNN\b': 'the SNN',
    r'\bto model\b': 'the model',
    r'\bto system\b': 'the system',
    r'\bto project\b': 'the project',
    r'\bto data\b': 'the data',
    r'\bto file\b': 'the file',
    r'\bto code\b': 'the code',
    r'\bto dataset\b': 'the dataset',
    r'\busing to\b': 'using the',
    r'\bUsing to\b': 'Using the',
}

def fix_file(file_path):
    """Fix a single file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        
        for pattern, replacement in FINAL_FIXES.items():
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
    print("FINAL CORRUPTION FIXES")
    print("=" * 80)
    print()
    
    patterns = ['**/*.md', '**/*.py']
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
    skip_files = {'fix_', 'translate_', 'cleanup_', 'complete_', 'final_'}
    
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
    print(f"COMPLETE! Fixed {count} files.")
    print("=" * 80)

if __name__ == '__main__':
    main()
