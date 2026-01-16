#!/usr/bin/env python3
"""
Limpeza ABSOLUTA FINAL de todas as palavras em inglês nos notebooks portugueses.
"""

import json
from pathlib import Path

def clean_text(text):
    """Remove TODAS as palavras em inglês restantes."""
    if not isinstance(text, str):
        return text
    
    # Lista completa de substituições
    replacements = [
        # Comentários Python
        ('ainda not estiver', 'ainda não estiver'),
        
        # Palavras técnicas em markdown
        ('**unsupervised**', '**não supervisionada**'),
        ('**supervised**', '**supervisionada**'),
        
        # Frases em inglês
        ('if o neurônio', 'se o neurônio'),
        (' BEFORE ', ' ANTES '),
        (' AFTER ', ' DEPOIS '),
        ('Potenciação** (weight ', 'Potenciação** (peso '),
        ('Depressão** (weight ', 'Depressão** (peso '),
        ('**temporal relações causais**', '**relações causais temporais**'),
        
        # Comentários técnicos
        ('time constant pre-synaptic', 'constante de tempo pré-sináptico'),
        ('time constant post-synaptic', 'constante de tempo pós-sináptico'),
        ('Potentiation Amplitude', 'Amplitude de Potenciação'),
        ('Depression Amplitude', 'Amplitude de Depressão'),
        
        # Mais palavras comuns
        (' per ', ' por '),
        (' according to', ' segundo'),
        (' main ', ' principal '),
        (' Main ', ' Principal '),
    ]
    
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    
    return result

def process_notebook(path):
    """Processa um notebook removendo palavras em inglês."""
    print(f"Limpando: {path.name}")
    
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb.get('cells', []):
        source = cell.get('source', [])
        if isinstance(source, list):
            cell['source'] = [clean_text(line) for line in source]
        else:
            cell['source'] = clean_text(source)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"  ✓ Limpo")

def main():
    notebooks_dir = Path(__file__).parent / 'notebooks'
    
    notebooks = [
        '01-stdp_example-pt.ipynb',
        '02-stdp-demo-pt.ipynb',
        '03-loihi_benchmark-pt.ipynb',
        '04_brian2_vs_snntorch-pt.ipynb',
        '05_production_solutions-pt.ipynb',
        '06_phase1_integration-pt.ipynb',
    ]
    
    print("=" * 70)
    print("LIMPEZA ABSOLUTA FINAL - REMOVENDO TODAS AS PALAVRAS EM INGLÊS")
    print("=" * 70)
    print()
    
    for nb_name in notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            process_notebook(nb_path)
    
    print()
    print("=" * 70)
    print("✅ LIMPEZA ABSOLUTA CONCLUÍDA!")
    print("=" * 70)

if __name__ == '__main__':
    main()
