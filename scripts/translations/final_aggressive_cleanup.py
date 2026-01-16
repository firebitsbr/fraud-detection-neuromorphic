#!/usr/bin/env python3
"""
Final aggressive cleanup - Remove ALL remaining Portuguese
"""
import re
from pathlib import Path

# Comprehensive Portuguese to English dictionary
COMPLETE_DICTIONARY = {
    # Common Portuguese words that were missed
    r'\babort\b': 'about',
    r'\bAbort\b': 'About',
    r'\bmecanismo\b': 'mechanism',
    r'\bMecanismo\b': 'Mechanism',
    r'\baprendizado\b': 'learning',
    r'\bAprendizado\b': 'Learning',
    r'\bbiológico\b': 'biological',
    r'\bBiológico\b': 'Biological',
    r'\butilizado\b': 'used',
    r'\bUtilizado\b': 'Used',
    r'\bredes\b': 'networks',
    r'\bRedes\b': 'Networks',
    r'\bneurais\b': 'neural',
    r'\bNeurais\b': 'Neural',
    r'\bneuromórstayss\b': 'neuromorphic',
    r'\bneuromórficas\b': 'neuromorphic',
    r'\bDemonstra\b': 'Demonstrates',
    r'\bdemonstra\b': 'demonstrates',
    r'\bas neurons\b': 'the neurons',
    r'\bAs neurons\b': 'The neurons',
    r'\baprendem\b': 'learn',
    r'\bAprendem\b': 'Learn',
    r'\bcorrelações\b': 'correlations',
    r'\bCorrelações\b': 'Correlations',
    r'\bhasforais\b': 'temporal',
    r'\btemporais\b': 'temporal',
    r'\bautomaticamente\b': 'automatically',
    r'\bAutomaticamente\b': 'Automatically',
    r'\binhaveativo\b': 'interactive',
    r'\bInhaveativo\b': 'Interactive',
    
    # Fix corrupted words from previous passes
    r'\bCompoif\b': 'Compose',
    r'\bcompoif\b': 'compose',
    r'\bponta to ponta\b': 'end-to-end',
    r'\bPonta to Ponta\b': 'End-to-End',
    r'\bdetecção\b': 'detection',
    r'\bDetecção\b': 'Detection',
    r'\bfraudulent\b': 'fraudulent',
    r'\bFraudulent\b': 'Fraudulent',
    r'\bcodistaysção\b': 'encoding',
    r'\bCodistaysção\b': 'Encoding',
    r'\binferência\b': 'inference',
    r'\bInferência\b': 'Inference',
    r'\bmotor\b': 'engine',
    r'\bMotor\b': 'Engine',
    r'\bdeciare\b': 'decision',
    r'\bDeciare\b': 'Decision',
    r'\bwithort\b': 'without',
    r'\bWithort\b': 'Without',
    
    # More Portuguese words
    r'\bpara\b': 'for',
    r'\bPara\b': 'For',
    r'\bcom\b': 'with',
    r'\bCom\b': 'With',
    r'\bque\b': 'that',
    r'\bQue\b': 'That',
    r'\bisso\b': 'this',
    r'\bIsso\b': 'This',
    r'\beste\b': 'this',
    r'\bEste\b': 'This',
    r'\bessa\b': 'this',
    r'\bEssa\b': 'This',
    r'\bfazer\b': 'make',
    r'\bFazer\b': 'Make',
    r'\bapenas\b': 'only',
    r'\bApenas\b': 'Only',
    r'\bExecution local\b': 'Local execution',
    r'\bexecution local\b': 'local execution',
    
    # Fix "de" in proper names (should remain)
    r'\bMauro Risonho for Paula\b': 'Mauro Risonho de Paula',
}

def fix_file(file_path):
    """Fix a single file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        
        for pattern, replacement in COMPLETE_DICTIONARY.items():
            content = re.sub(pattern, replacement, content)
        
        # Fix specific patterns
        # Fix "for" that should be "de" in Portuguese names
        content = re.sub(r'Mauro Risonho for Paula', 'Mauro Risonho de Paula', content)
        content = re.sub(r'Mauro Risonho of Paula', 'Mauro Risonho de Paula', content)
        
        if content != original:
            file_path.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    print("=" * 80)
    print("FINAL AGGRESSIVE CLEANUP - Removing ALL Portuguese")
    print("=" * 80)
    print()
    
    patterns = ['**/*.md', '**/*.py']
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
    skip_files = {'final_aggressive', 'translate', 'cleanup', 'polish', 'fix_'}
    
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
