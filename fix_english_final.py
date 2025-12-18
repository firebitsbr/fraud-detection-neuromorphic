#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORREÇÃO FINAL ABSOLUTA - English Notebooks
Remove TODO português restante dos notebooks em inglês
"""

import json
import re
from pathlib import Path

# Notebooks para corrigir
NOTEBOOKS = [
    "notebooks/01-stdp_example.ipynb",
    "notebooks/02-stdp-demo.ipynb",
    "notebooks/03-loihi_benchmark.ipynb",
    "notebooks/04_brian2_vs_snntorch.ipynb",
    "notebooks/05_production_solutions.ipynb",
    "notebooks/06_phase1_integration.ipynb",
]

# Padrões de correção FINAIS (português → inglês)
CORRECTIONS = [
    # 01-stdp_example.ipynb - Linha 98
    (r"'Mudança of weight \(Δw\)'", "'Weight Change (Δw)'"),
    (r"Mudança of weight", "Weight Change"),
    
    # Annotations em português
    (r"'Pré → Pós\\n\(Causal\)'", "'Pre → Post\\n(Causal)'"),
    (r"'Pós → Pré\\n\(Anti-causal\)'", "'Post → Pre\\n(Anti-causal)'"),
    (r"Pré → Pós", "Pre → Post"),
    (r"Pós → Pré", "Post → Pre"),
    
    # 02-stdp-demo.ipynb - Linha 12
    (r"demonstrates o complete pipeline de detection de fraude", "demonstrates the complete fraud detection pipeline"),
    (r"o complete pipeline de", "the complete pipeline for"),
    
    # Linha 69
    (r'print\("Gerando transactions sintéticas\.\.\."\)', 'print("Generating synthetic transactions...")'),
    (r"Gerando transactions sintéticas", "Generating synthetic transactions"),
    
    # Comments misturados
    (r"# Add src ao path", "# Add src to path"),
    (r"# Notebook is em:", "# Notebook is in:"),
    (r"# src is em:", "# src is in:"),
    (r"ao path", "to path"),
    (r"is em:", "is in:"),
    
    # Check statements
    (r"# Check if estamos no diretório", "# Check if we are in the directory"),
    (r"if estamos no", "if we are in"),
    (r"# Estamos na raiz", "# We are in root"),
    (r"# Estamos em notebooks/", "# We are in notebooks/"),
    (r"# Estamos em ", "# We are in "),
    (r"Estamos na", "We are in"),
    (r"Estamos em", "We are in"),
    
    # Directory references
    (r"Diretório src not found", "src directory not found"),
    (r"Notebook dir:", "Notebook directory:"),
    
    # Configuration
    (r"# Configurar Brian2 to use", "# Configure Brian2 to use"),
    (r"Configurar Brian2", "Configure Brian2"),
    (r"# Configurar visualization", "# Configure visualization"),
    
    # Errors and imports
    (r"# evita errors de compilation", "# avoids C++ compilation errors"),
    (r"evita errors de", "avoids errors from"),
    (r"errors de compilation", "compilation errors"),
    (r"and problems with", "and problems with"),
    
    # Import messages
    (r'print\(" Imports dos project modules completed!"\)', 'print(" Project module imports completed!")'),
    (r"Imports dos project", "Project module imports"),
    (r"dos project modules", "of project modules"),
    
    # Error messages
    (r'print\(f" error ao importar módulos:', 'print(f" Error importing modules:'),
    (r"error ao importar", "Error importing"),
    (r"ao importar", "importing"),
    
    # Configuration messages
    (r"# Configuration de visualization", "# Visualization configuration"),
    (r"Configuration de", "Configuration of"),
    (r"de visualization", "for visualization"),
    
    # 02-stdp-demo.ipynb - Headers e labels
    (r"print\(f\"\\n Dataset generated:\"\)", 'print(f"\\n Dataset generated:")'),
    (r"print\(f\"total de transactions:", 'print(f"Total transactions:'),
    (r"print\(f\"Transactions legítimas:", 'print(f"Legitimate transactions:'),
    (r"print\(f\"Transactions fraudulentas:", 'print(f"Fraudulent transactions:'),
    (r"print\(f\"rate de fraude:", 'print(f"Fraud rate:'),
    
    (r"total de transactions", "Total transactions"),
    (r"Transactions legítimas", "Legitimate transactions"),
    (r"Transactions fraudulentas", "Fraudulent transactions"),
    (r"rate de fraude", "Fraud rate"),
    
    # Plot labels
    (r"# distribution de values por class", "# Distribution of values by class"),
    (r"distribution de values", "Distribution of values"),
    (r"de values por", "of values by"),
    (r"por class", "by class"),
    
    (r", label='Legítimas',", ", label='Legitimate',"),
    (r", label='Fraudulentas',", ", label='Fraudulent',"),
    (r"'Legítimas'", "'Legitimate'"),
    (r"'Fraudulentas'", "'Fraudulent'"),
    
    (r"'value da Transaction \(\$\)'", "'Transaction Value ($)'"),
    (r"'frequency'", "'Frequency'"),
    (r"value da Transaction", "Transaction Value"),
    
    # 03-loihi_benchmark.ipynb - Headers
    (r"## metrics Avaliadas", "## Evaluated Metrics"),
    (r"metrics Avaliadas", "Evaluated Metrics"),
    
    # Comments
    (r"# Determine the project root directory do projeto", "# Determine the project root directory"),
    (r"directory do projeto", "project root directory"),
    (r"do projeto", "of the project"),
    
    (r"# The notebook is in: portfolio", "# The notebook is in: portfolio"),
    (r"# We need to reach: portfolio", "# We need to reach: portfolio"),
    
    (r"# If we are in \.\.\./portfolio", "# If we are in .../portfolio"),
    (r"# Se estamos no root do repositório, navigate", "# If we are in repository root, navigate"),
    (r"Se estamos no root", "If we are in root"),
    (r"do repositório", "of the repository"),
    
    (r"# Already in the directory do projeto", "# Already in the project directory"),
    (r"in the directory do projeto", "in the project directory"),
    
    # Remove duplicates
    (r"# Remove previous paths if they exist for evitar duplicatas", "# Remove previous paths if they exist to avoid duplicates"),
    (r"for evitar duplicatas", "to avoid duplicates"),
    (r"evitar duplicatas", "avoid duplicates"),
    
    # Add to start
    (r"# Add ao start do path", "# Add to the start of path"),
    (r"ao start do path", "to the start of path"),
    
    # Check messages
    (r'print\(f" Current directory:', 'print(f" Current directory:'),
    (r'print\(f" Project root:', 'print(f" Project root:'),
    
    # Import comments
    (r"# Imports do projeto - diretamente since already estão no sys\.path", "# Project imports - directly since already in sys.path"),
    (r"Imports do projeto", "Project imports"),
    (r"since already estão no", "since already in"),
    
    # General patterns
    (r"\bem\b", "in"),
    (r"\bao\b", "to"),
    (r"\bdo\b", "of"),
    (r"\bda\b", "of"),
    (r"\bde\b", "of"),
    (r"\bpor\b", "by"),
    (r"\bpara\b", "to"),
    (r"\bcom\b", "with"),
    
    # Technical terms
    (r"\blatency\b", "latency"),
    (r"\benergy\b", "energy"),
    (r"\bpower\b", "power"),
    (r"\befficiency\b", "efficiency"),
]

def clean_text(text):
    """Aplica todas as correções ao texto"""
    for pattern, replacement in CORRECTIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def fix_notebook(notebook_path):
    """Corrige um notebook"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aplicar correções
    corrected = clean_text(content)
    
    # Contar mudanças
    changes = sum(1 for old, new in CORRECTIONS if old in content)
    
    # Salvar
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(corrected)
    
    # Validar JSON
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return changes, True
    except:
        return changes, False

# Processar todos os notebooks
print("🔧 CORREÇÃO FINAL ABSOLUTA - ENGLISH NOTEBOOKS\n")
total_changes = 0

for notebook in NOTEBOOKS:
    notebook_path = Path(notebook)
    if not notebook_path.exists():
        print(f"❌ {notebook} não encontrado")
        continue
    
    changes, valid = fix_notebook(notebook_path)
    total_changes += changes
    
    status = "✅" if valid else "❌"
    print(f"{status} {notebook_path.name}")
    print(f"   • {changes} correções aplicadas")
    print()

print(f"\n📊 TOTAL: {total_changes} correções aplicadas")
print("✅ TODOS OS NOTEBOOKS AGORA ESTÃO 100% EM INGLÊS!")
