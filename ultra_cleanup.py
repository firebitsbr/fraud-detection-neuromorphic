#!/usr/bin/env python3
"""
Ultra comprehensive final cleanup - Remove EVERY remaining Portuguese word
"""
import re
from pathlib import Path

# Complete dictionary of all remaining Portuguese
ALL_PORTUGUESE = {
    # Nouns
    r'\bprodução\b': 'production',
    r'\bProdução\b': 'Production',
    r'\bDescrição\b': 'Description',
    r'\bdescrição\b': 'description',
    r'\bGuia\b': 'Guide',
    r'\bguia\b': 'guide',
    r'\bFaif\b': 'Phase',
    r'\bfaif\b': 'phase',
    r'\bdescreve\b': 'describes',
    r'\bDescreve\b': 'Describes',
    r'\bimplementadas\b': 'implemented',
    r'\bImplementadas\b': 'Implemented',
    r'\botimizações\b': 'optimizations',
    r'\bOtimizações\b': 'Optimizations',
    r'\bCarregamento\b': 'Loading',
    r'\bcarregamento\b': 'loading',
    r'\bmaximizar\b': 'maximize',
    r'\bMaximizar\b': 'Maximize',
    r'\bperformance\b': 'performance',
    r'\bPerformance\b': 'Performance',
    
    # Verbs and adjectives
    r'\bdireto\b': 'direct',
    r'\bDireto\b': 'Direct',
    r'\bpública\b': 'public',
    r'\bpúblico\b': 'public',
    r'\bPúblico\b': 'Public',
    r'\bgenérico\b': 'generic',
    r'\bGenérico\b': 'Generic',
    r'\bespecífico\b': 'specific',
    r'\bEspecífico\b': 'Specific',
    r'\bgrande\b': 'large',
    r'\bGrande\b': 'Large',
    r'\bpethatno\b': 'small',
    r'\bPethatno\b': 'Small',
    r'\bimediatamente\b': 'immediately',
    r'\bImediatamente\b': 'Immediately',
    r'\bimaturas\b': 'immature',
    r'\bImaturas\b': 'Immature',
    r'\bmadura\b': 'mature',
    r'\bMadura\b': 'Mature',
    r'\bcrítico\b': 'critical',
    r'\bCrítico\b': 'Critical',
    r'\bbalanceado\b': 'balanced',
    r'\bBalanceado\b': 'Balanced',
    r'\bavançado\b': 'advanced',
    r'\bAvançado\b': 'Advanced',
    r'\bdetalhadas\b': 'detailed',
    r'\bDetalhadas\b': 'Detailed',
    r'\bautomatizado\b': 'automated',
    r'\bAutomatizado\b': 'Automated',
    r'\binhaveativa\b': 'interactive',
    r'\bInhaveativa\b': 'Interactive',
    r'\binhaveativo\b': 'interactive',
    r'\bInhaveativo\b': 'Interactive',
    
    # Expressions and phrases
    r'\bObhave\b': 'Obtain',
    r'\bobhave\b': 'obtain',
    r'\bmeifs\b': 'months',
    r'\bmês\b': 'month',
    r'\bMês\b': 'Month',
    r'\batravés\b': 'through',
    r'\bAtravés\b': 'Through',
    r'\bferramentas\b': 'tools',
    r'\bFerramentas\b': 'Tools',
    r'\bpesquisa\b': 'research',
    r'\bPesquisa\b': 'Research',
    r'\bfuncionar\b': 'work',
    r'\bFuncionar\b': 'Work',
    r'\bSishasa\b': 'System',
    r'\bsishasa\b': 'system',
    r'\btráfego\b': 'traffic',
    r'\bTráfego\b': 'Traffic',
    r'\blimitado\b': 'limited',
    r'\bLimitado\b': 'Limited',
    r'\bconsumir\b': 'consume',
    r'\bConsumir\b': 'Consume',
    r'\bfeedback\b': 'feedback',
    r'\bFeedback\b': 'Feedback',
    
    # Verbs
    r'\bBaixa\b': 'Download',
    r'\bbaixa\b': 'download',
    r'\bbaixar\b': 'download',
    r'\bBaixar\b': 'Download',
    r'\bIniciando\b': 'Starting',
    r'\biniciando\b': 'starting',
    r'\blevar\b': 'take',
    r'\bLevar\b': 'Take',
    r'\balguns\b': 'some',
    r'\bAlguns\b': 'Some',
    r'\bminutes\b': 'minutes',
    r'\bMinutes\b': 'Minutes',
    r'\bAviso\b': 'Warning',
    r'\baviso\b': 'warning',
    r'\bguiar\b': 'guide',
    r'\bGuiar\b': 'Guide',
    r'\bmanual\b': 'manual',
    r'\bManual\b': 'Manual',
    r'\bespestay\b': 'specify',
    r'\bEspestay\b': 'Specify',
    r'\bespecistay\b': 'specify',
    r'\bEspecistay\b': 'Specify',
    r'\borigins\b': 'origins',
    r'\bOrigins\b': 'Origins',
    r'\binicial\b': 'initial',
    r'\bInicial\b': 'Initial',
    
    # More words
    r'\bCarregador\b': 'Loader',
    r'\bcarregador\b': 'loader',
    r'\bPreprocessa\b': 'Preprocess',
    r'\bpreprocessa\b': 'preprocess',
    r'\bpreprocessamento\b': 'preprocessing',
    r'\bPreprocessamento\b': 'Preprocessing',
    r'\bPróxima\b': 'Next',
    r'\bpróxima\b': 'next',
    r'\bcriação\b': 'creation',
    r'\bCriação\b': 'Creation',
    r'\bsolução\b': 'solution',
    r'\bSolução\b': 'Solution',
    r'\bprojeto\b': 'project',
    r'\bProjeto\b': 'Project',
    r'\bimplementa\b': 'implements',
    r'\bImplementa\b': 'Implements',
    r'\bfocado\b': 'focused',
    r'\bFocado\b': 'Focused',
    r'\bfocor\b': 'focused',
    r'\bFocor\b': 'Focused',
    r'\bobjetivos\b': 'objectives',
    r'\bObjetivos\b': 'Objectives',
    r'\balcançados\b': 'achieved',
    r'\bAlcançados\b': 'Achieved',
    r'\bimplementações\b': 'implementations',
    r'\bImplementações\b': 'Implementations',
    r'\bcomplete\b': 'complete',
    r'\bComplete\b': 'Complete',
    r'\btestadas\b': 'tested',
    r'\bTestadas\b': 'Tested',
    r'\bEstatísticas\b': 'Statistics',
    r'\bestatísticas\b': 'statistics',
    r'\bSuforte\b': 'Support',
    r'\bsuforte\b': 'support',
    r'\bTesta\b': 'Tests',
    r'\btesta\b': 'tests',
    r'\bdiferentes\b': 'different',
    r'\bDiferentes\b': 'Different',
    r'\btamanhos\b': 'sizes',
    r'\bTamanhos\b': 'Sizes',
    r'\binstruções\b': 'instructions',
    r'\bInstruções\b': 'Instructions',
    r'\bveja\b': 'see',
    r'\bVeja\b': 'See',
    
    # Corrupted words
    r'\bdataift\b': 'dataset',
    r'\bDataift\b': 'Dataset',
    r'\bDataiftLoader\b': 'DatasetLoader',
    r'\bCreditCardDataiftLoader\b': 'CreditCardDatasetLoader',
    r'\bTensorDataift\b': 'TensorDataset',
    r'\bDataAugmenhave\b': 'DataAugment',
    r'\breverif\b': 'reverse',
    r'\bReverif\b': 'Reverse',
    r'\braiif\b': 'raise',
    r'\bRaiif\b': 'Raise',
    r'\bwithmit\b': 'commit',
    r'\bWithmit\b': 'Commit',
    r'\bCompoif\b': 'Compose',
    r'\bcompoif\b': 'compose',
}

def fix_file(file_path):
    """Fix a single file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        
        for pattern, replacement in ALL_PORTUGUESE.items():
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
    print("ULTRA COMPREHENSIVE CLEANUP - Removing ALL Portuguese")
    print("=" * 80)
    print()
    
    patterns = ['**/*.md', '**/*.py']
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
    skip_files = {'ultra_', 'final_', 'translate', 'cleanup', 'polish', 'fix_', 'aggressive'}
    
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
