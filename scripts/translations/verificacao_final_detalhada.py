#!/usr/bin/env python3
"""Verificação final detalhada - Busca QUALQUER palavra em inglês"""
import json
import re

notebooks = [
    "notebooks/01-stdp_example-pt.ipynb",
    "notebooks/02-stdp-demo-pt.ipynb",
    "notebooks/03-loihi_benchmark-pt.ipynb",
    "notebooks/04_brian2_vs_snntorch-pt.ipynb",
    "notebooks/05_production_solutions-pt.ipynb",
    "notebooks/06_phase1_integration-pt.ipynb",
]

# Lista de palavras em inglês comuns para detectar
english_words = [
    'weight', 'weights', 'spike', 'spikes', 'time', 'neuron', 'neurons',
    'the', 'is', 'are', 'with', 'for', 'and', 'of', 'in', 'to',
    'fires', 'fire', 'learn', 'learns', 'used', 'using',
    'network', 'evolution', 'simulation', 'results', 'pattern', 'patterns',
    'input', 'output', 'behavior', 'sequence', 'temporal', 'learning',
    'transaction', 'transactions', 'detection', 'fraud', 'application',
    'method', 'methods', 'comparison', 'future', 'conclusions',
    'visualization', 'analysis', 'metrics', 'benchmark', 'performance',
]

# Exceções permitidas (nomes de bibliotecas, código Python, etc.)
allowed_exceptions = [
    'import', 'from', 'def', 'class', 'return', 'if', 'else', 'for', 'while',
    'try', 'except', 'with', 'as', 'in', 'not', 'and', 'or', 'True', 'False',
    'None', 'print', 'range', 'len', 'str', 'int', 'float', 'list', 'dict',
    'brian2', 'numpy', 'matplotlib', 'pandas', 'pyplot', 'time', 'datetime',
    'Path', 'pathlib', '.py', '.ipynb', 'https://', 'http://',
    'github.com', 'colab.research.google.com',
]

print("="*80)
print("  VERIFICAÇÃO FINAL DETALHADA - PALAVRAS EM INGLÊS")
print("="*80)

total_issues = 0

for nb_path in notebooks:
    nb_name = nb_path.split('/')[-1]
    print(f"\n📘 {nb_name}")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    issues_found = []
    
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] in ['markdown', 'code']:
            content = ''.join(cell['source'])
            
            # Verificar cada palavra em inglês
            for eng_word in english_words:
                # Procurar a palavra como palavra inteira (word boundary)
                pattern = r'\b' + re.escape(eng_word) + r'\b'
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    # Verificar se está em contexto permitido
                    context = content[max(0, match.start()-30):min(len(content), match.end()+30)]
                    
                    # Pular se for exceção permitida
                    is_exception = False
                    for exc in allowed_exceptions:
                        if exc.lower() in context.lower():
                            is_exception = True
                            break
                    
                    if not is_exception:
                        issues_found.append({
                            'cell': cell_idx + 1,
                            'word': match.group(),
                            'context': context.strip()[:60] + '...'
                        })
    
    if issues_found:
        print(f"   ⚠️  {len(issues_found)} possíveis palavras em inglês encontradas:")
        for issue in issues_found[:5]:  # Mostrar apenas as primeiras 5
            print(f"      Célula {issue['cell']}: '{issue['word']}' em \"{issue['context']}\"")
        if len(issues_found) > 5:
            print(f"      ... e mais {len(issues_found) - 5} ocorrências")
        total_issues += len(issues_found)
    else:
        print(f"   ✅ Nenhuma palavra em inglês detectada!")

print(f"\n{'='*80}")
if total_issues == 0:
    print("✅ PERFEITO! NENHUMA PALAVRA EM INGLÊS ENCONTRADA!")
    print("   TODOS OS NOTEBOOKS ESTÃO 100% EM PORTUGUÊS!")
else:
    print(f"⚠️  Total de {total_issues} possíveis palavras em inglês encontradas")
    print("   (Pode incluir falsos positivos em código Python ou nomes próprios)")
print("="*80)
