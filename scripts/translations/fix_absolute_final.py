#!/usr/bin/env python3
"""Correção ABSOLUTA FINAL - Remove TODO inglês, inclusive em código Python"""
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

# Traduções para comentários Python e strings
code_translations = [
    # Comentários Python
    ('# ms - constante de tempo pré-sináptico', '# ms - constante de tempo pré-sináptico'),
    ('# ms - constante de tempo pós-sináptico', '# ms - constante de tempo pós-sináptico'),
    ('# Amplitude de Potenciação', '# Amplitude de Potenciação'),
    ('# Amplitude de Depressão', '# Amplitude de Depressão'),
    ('# Delta t (temporal difference)', '# Delta t (diferença temporal)'),
    ('# Calculate mudança de weight', '# Calcular mudança de peso'),
    ('# Post after Pre → Potentiation', '# Post depois Pre → Potenciação'),
    ('# Post before Pre → Depression', '# Post antes Pre → Depressão'),
    ('# Plot curve STDP', '# Plotar curva STDP'),
    
    # Docstrings
    ('Calculates mudança de weight according para STDP', 'Calcula a mudança de peso de acordo com STDP'),
    ('dt = t_post - t_pre', 'dt = t_post - t_pre'),
    
    # Variáveis e funções - MANTER EM INGLÊS (Python syntax)
    # Mas traduzir strings literais
    ("'seaborn-v0_8-whitegrid'", "'seaborn-v0_8-whitegrid'"),
    ('print(" Importações concluídas!")', 'print("✓ Importações concluídas!")'),
    
    # Texto em markdown dentro de código
    ('**Throughput**', '**Vazão**'),
    ('CPU Traditional', 'CPU Tradicional'),
    
    # Correções de código quebrado
    ('if .* not in str(notebook_dir):', 'if notebook_dir.name not in str(notebook_dir):'),
]

# Traduções apenas para markdown
markdown_translations = [
    ('**latency**', '**latência**'),
    ('**Throughput**', '**Vazão**'),
    ('**energy**', '**energia**'),
    ('**power**', '**potência**'),
    ('**efficiency**', '**eficiência**'),
    ('CPU Traditional', 'CPU Tradicional'),
    ('ms por inferência', 'ms por inferência'),
    ('transações por segundo', 'transações por segundo'),
    ('aceleração e eficiência de potência', 'aceleração e eficiência de potência'),
]

for nb_path in notebooks:
    print(f"Corrigindo ABSOLUTAMENTE: {nb_path}")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells_fixed = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # Para células de código: traduzir APENAS comentários e strings
            source = ''.join(cell['source'])
            original = source
            
            for old, new in code_translations:
                source = source.replace(old, new)
            
            if source != original:
                cell['source'] = source.split('\n')
                # Garantir quebra de linha no final
                if cell['source'] and not cell['source'][-1].endswith('\n'):
                    cell['source'][-1] += '\n'
                cells_fixed += 1
                
        elif cell['cell_type'] == 'markdown':
            # Para markdown: traduzir tudo
            source = ''.join(cell['source'])
            original = source
            
            for old, new in markdown_translations:
                source = source.replace(old, new)
            
            if source != original:
                cell['source'] = source.split('\n')
                if cell['source'] and not cell['source'][-1].endswith('\n'):
                    cell['source'][-1] += '\n'
                cells_fixed += 1
    
    # Salvar
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"  ✓ {cells_fixed} células corrigidas")

print("\n✅ CORREÇÃO ABSOLUTA FINAL CONCLUÍDA!")
print("\nValidando JSON...")
import subprocess
for nb_path in notebooks:
    result = subprocess.run(['python3', '-m', 'json.tool', nb_path], 
                          capture_output=True)
    if result.returncode == 0:
        print(f"✓ {nb_path}")
    else:
        print(f"✗ {nb_path} - ERRO JSON!")
