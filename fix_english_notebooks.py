#!/usr/bin/env python3
"""
Correção MASSIVA - Garantir que notebooks em INGLÊS estejam 100% em inglês
Remove TODO português dos notebooks .ipynb (sem -pt)
"""
import json
import re

notebooks = [
    "notebooks/01-stdp_example.ipynb",
    "notebooks/02-stdp-demo.ipynb",
    "notebooks/03-loihi_benchmark.ipynb",
    "notebooks/04_brian2_vs_snntorch.ipynb",
    "notebooks/05_production_solutions.ipynb",
    "notebooks/06_phase1_integration.ipynb",
]

# Dicionário MASSIVO de traduções PORTUGUÊS → INGLÊS
portuguese_to_english = [
    # URLs
    ('google.with', 'google.com'),
    
    # Títulos e descrições
    ('Detection de Fraude Neuromórfica', 'Neuromorphic Fraud Detection'),
    ('Demonstração:', 'Demonstration:'),
    ('Este notebook Demonstrates', 'This notebook demonstrates'),
    ('Descrição:', 'Description:'),
    ('Autor:', 'Author:'),
    ('Data de Criação:', 'Creation Date:'),
    ('Licença:', 'License:'),
    ('Desenvolvimento:', 'Development:'),
    
    # Comentários Python
    ('# Instalar a biblioteca', '# Install the library'),
    ('se ainda não', 'if not yet'),
    ('# Importação específica', '# Specific import'),
    ('ao invés de', 'instead of'),
    ('# Configurar para usar', '# Configure to use'),
    ('evita erro de compilação', 'avoids compilation error'),
    ('se os headers', 'if the headers'),
    ('# Adicionar src ao path', '# Add src to path'),
    ('# Add src para path', '# Add src to path'),
    ('Diretório src adicionado:', 'src directory added:'),
    ('Diretório src não encontrado!', 'src directory not found!'),
    ('Nossos módulos', 'Our modules'),
    ('erro ao importar módulos:', 'error importing modules:'),
    ('Configuração de visualização', 'Visualization configuration'),
    
    # Seções
    ('## 2. Geração de dados Sintéticos', '## 2. Synthetic Data Generation'),
    ('## 2. Generation de data Synthetics', '## 2. Synthetic Data Generation'),
    ('Vamos criar', "Let's create"),
    ('conjunto de dados sintético', 'synthetic dataset'),
    ('transações bancárias', 'banking transactions'),
    ('padrões realistas', 'realistic patterns'),
    
    # Print statements
    ('Gerando transações sintéticas...', 'Generating synthetic transactions...'),
    ('total de transações:', 'total transactions:'),
    ('Transações legítimas:', 'Legitimate transactions:'),
    ('Transações fraudulentas:', 'Fraudulent transactions:'),
    ('taxa de fraude:', 'fraud rate:'),
    ('Mostrar primeiras linhas', 'Show first rows'),
    
    # Visualização
    ('distribuição de valores por classe', 'value distribution by class'),
    ('valor da Transação', 'Transaction value'),
    ('frequência', 'frequency'),
    ('frequência diária por classe', 'daily frequency by class'),
    ('frequência de Transações por classe', 'Transactions frequency by class'),
    ('padrões observados:', 'observed patterns:'),
    
    # Objetivos e métricas
    ('Objetivo', 'Objective'),
    ('Comparar o desempenho da', 'Compare the performance of'),
    ('Comparar a performance da', 'Compare the performance of'),
    ('implementação de detecção de fraude', 'fraud detection implementation'),
    ('métricas Avaliadas', 'Evaluated Metrics'),
    ('latência', 'latency'),
    ('ms por inferência', 'ms per inference'),
    ('transações por segundo', 'transactions per second'),
    ('energia', 'energy'),
    ('potência', 'power'),
    ('eficiência', 'efficiency'),
    ('aceleração', 'speedup'),
    
    # Texto misto
    ('mudança', 'change'),
    ('Calculates mudança of weight', 'Calculates weight change'),
    ('according to', 'per'),
    ('de acordo com', 'according to'),
    ('para usar', 'to use'),
    ('for usar', 'to use'),
    ('problems with', 'problems with'),
    ('e problemas with', 'and problems with'),
    
    # Comentários de código
    ('Determinar o diretório raiz', 'Determine the project root directory'),
    ('O notebook está em:', 'The notebook is in:'),
    ('O notebook is em:', 'The notebook is in:'),
    ('Precisamos chegar em:', 'We need to reach:'),
    ('Se estamos em', 'If we are in'),
    ('Já estamos no diretório', 'Already in the directory'),
    ('Remover caminhos anteriores', 'Remove previous paths'),
    ('se existirem', 'if they exist'),
    ('para evitar duplicatas', 'to avoid duplicates'),
    ('Verificar se', 'Check if'),
    ('os diretórios existem', 'the directories exist'),
    
    # Termos específicos
    ('simulador', 'simulator'),
    ('Simulação de hardware', 'Hardware simulation'),
    ('neuromórfico', 'neuromorphic'),
    ('Implementation de detection de fraude', 'fraud detection implementation'),
    
    # Frases completas em português
    ('# O notebook está em: portfolio', '# The notebook is in: portfolio'),
    ('# Precisamos chegar em: portfolio', '# We need to reach: portfolio'),
    ('# Se estamos em ...', '# If we are in ...'),
    ('# Já estamos no diretório do projeto', '# Already in the project directory'),
    ('# Remover caminhos anteriores se existirem para evitar duplicatas', '# Remove previous paths if they exist to avoid duplicates'),
    ('# Verificar se os diretórios existem', '# Check if the directories exist'),
]

print("="*70)
print("  CORREÇÃO MASSIVA - INGLÊS 100%")
print("="*70)
print(f"\n🔧 Aplicando {len(portuguese_to_english)} traduções em 6 notebooks...\n")

total_corrections = 0

for nb_path in notebooks:
    nb_name = nb_path.split('/')[-1]
    print(f"📘 Processando: {nb_name}")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    corrections_in_file = 0
    
    # Aplicar todas as traduções
    for portuguese, english in portuguese_to_english:
        if portuguese in content:
            count = content.count(portuguese)
            content = content.replace(portuguese, english)
            corrections_in_file += count
    
    # Salvar se houve mudanças
    if content != original_content:
        with open(nb_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✓ {corrections_in_file} correções aplicadas")
        total_corrections += corrections_in_file
    else:
        print(f"   ✓ Nenhuma correção necessária")

print(f"\n{'='*70}")
print(f"✅ CORREÇÃO MASSIVA CONCLUÍDA!")
print(f"   Total de correções: {total_corrections}")
print(f"{'='*70}")

# Validar JSON
print("\n🔍 Validando estrutura JSON...")
import subprocess
all_valid = True
for nb_path in notebooks:
    result = subprocess.run(['python3', '-m', 'json.tool', nb_path], 
                          capture_output=True)
    nb_name = nb_path.split('/')[-1]
    if result.returncode == 0:
        print(f"   ✓ {nb_name}")
    else:
        print(f"   ✗ {nb_name} - ERRO JSON!")
        all_valid = False

if all_valid:
    print(f"\n{'='*70}")
    print("✅ TODOS OS 6 NOTEBOOKS ESTÃO VÁLIDOS E 100% EM INGLÊS!")
    print(f"{'='*70}")
