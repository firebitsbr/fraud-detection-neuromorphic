#!/usr/bin/env python3
"""
Final cleanup to fix corrupted translations and remaining Portuguese
"""
import re
from pathlib import Path

# Fix corrupted translations from previous passes
CORRUPTION_FIXES = {
    # Fix broken words from previous translations
    r'\bVeristaysção\b': 'Verification',
    r'\bveristaysção\b': 'verification',
    r'\bbinários\b': 'binaries',
    r'\bpré-withpilados\b': 'pre-compiled',
    r'\bwithpilados\b': 'compiled',
    r'\bCarregamento\b': 'Loading',
    r'\bcarregamento\b': 'loading',
    r'\bdataifts\b': 'datasets',
    r'\bDataifts\b': 'Datasets',
    r'\bUtilitários\b': 'Utilities',
    r'\butilitários\b': 'utilities',
    r'\benagainstdo\b': 'enabled',
    r'\bCarregando\b': 'Loading',
    r'\bcarregando\b': 'loading',
    r'\bincluindo\b': 'including',
    r'\bnormalização\b': 'normalization',
    r'\bbalanceamento\b': 'balancing',
    r'\bpretoção\b': 'preparation',
    r'\btraing\b': 'training',
    r'\bRewithendação\b': 'Recommendation',
    r'\brewithendação\b': 'recommendation',
    r'\bReduzir\b': 'Reduce',
    r'\breduzir\b': 'reduce',
    r'\badicionar\b': 'add',
    r'\bindicadores\b': 'indicators',
    r'\bprogresso\b': 'progress',
    r'\bTempos\b': 'Times',
    r'\btempos\b': 'times',
    r'\bintegração\b': 'integration',
    
    # Remaining Portuguese words
    r'\bpré-requisitos\b': 'prerequisites',
    r'\bpré-requisito\b': 'prerequisite',
    r'\bpré-sináptica\b': 'pre-synaptic',
    r'\bpré-sináptico\b': 'pre-synaptic',
    r'\bpré-sinápticos\b': 'pre-synaptic',
    r'\bpré-processing\b': 'preprocessing',
    r'\bpré-processados\b': 'preprocessed',
    r'\bConstante\b': 'Constant',
    r'\bconstante\b': 'constant',
    r'\bEstima\b': 'Estimates',
    r'\bestima\b': 'estimates',
    r'\bnúmero\b': 'number',
    r'\bNúmero\b': 'Number',
    r'\btotal\b': 'total',
    r'\bTotal\b': 'Total',
    r'\bbaseado\b': 'based',
    r'\bBaseado\b': 'Based',
    r'\barquitetura\b': 'architecture',
    r'\bArquitetura\b': 'Architecture',
    r'\bsimulação\b': 'simulation',
    r'\bSimulação\b': 'Simulation',
    r'\bestimado\b': 'estimated',
    r'\bEstimado\b': 'Estimated',
    r'\binferências\b': 'inferences',
    r'\bInferências\b': 'Inferences',
    r'\bexecutar\b': 'execute',
    r'\bExecutar\b': 'Execute',
    r'\batravés\b': 'through',
    r'\bAtravés\b': 'Through',
    r'\bPré\b': 'Pre',
    r'\bpré\b': 'pre',
}

def fix_file(file_path):
    """Fix a single file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        
        for pattern, replacement in CORRUPTION_FIXES.items():
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        if content != original:
            file_path.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    print("=" * 80)
    print("FINAL CLEANUP - Fixing corruptions and remaining Portuguese")
    print("=" * 80)
    print()
    
    patterns = ['**/*.md', '**/*.py']
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
    skip_files = {'final_cleanup.py', 'translate', 'translation'}
    
    count = 0
    base = Path('.')
    
    for pattern in patterns:
        for file_path in base.glob(pattern):
            # Skip directories
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            # Skip translation scripts
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
