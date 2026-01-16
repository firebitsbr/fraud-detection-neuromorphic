#!/usr/bin/env python3
"""
Translate all Portuguese content to English in the fraud-detection-neuromorphic project.
This script translates markdown files, Python docstrings, comments, and Jupyter notebooks.
"""

import os
import re
from pathlib import Path
from typing import Dict

# Translation dictionary for common Portuguese to English terms
TRANSLATIONS = {
    # Headers and sections
    "# Detecção de Fraude Neuromórfica": "# Neuromorphic Fraud Detection",
    "## Índice": "## Table of Contents",
    "## Visão Geral": "## Overview",
    "## Autor": "## Author",
    "## Por Que": "## Why",
    "## Arquitetura": "## Architecture",
    "## Instalação": "## Installation",
    "## Executando": "## Running",
    "## Usando": "## Using",
    "## Testes": "## Tests",
    "## Resultados": "## Results",
    "## Documentação": "## Documentation",
    "## Estrutura": "## Structure",
    "## Tecnologias": "## Technologies",
    "## Roadmap": "## Roadmap",
    "## Contribuindo": "## Contributing",
    "## Referências": "## References",
    "## Contato": "## Contact",
    "## Licença": "## License",
    "## Agradecimentos": "## Acknowledgments",
    
    # Common words and phrases
    "Descrição:": "Description:",
    "Autor:": "Author:",
    "Data de Criação:": "Creation Date:",
    "Data:": "Date:",
    "Licença:": "License:",
    "Área:": "Area:",
    "Status:": "Status:",
    "Completo": "Complete",
    "Em andamento": "In progress",
    
    # Technical terms
    "Configuração": "Configuration",
    "configuração": "configuration",
    "Pré-requisitos": "Prerequisites",
    "Instalação Rápida": "Quick Start",
    "Instalação Manual": "Manual Installation",
    "Passo a Passo": "Step by Step",
    "execução": "execution",
    "Execução": "Execution",
    "ambiente": "environment",
    "Ambiente": "Environment",
    "dados": "data",
    "Dados": "Data",
    "modelo": "model",
    "Modelo": "Model",
    "treinamento": "training",
    "Treinamento": "Training",
    "implantação": "deployment",
    "Implantação": "Deployment",
    
    # Common verbs
    "Criar": "Create",
    "criar": "create",
    "Instalar": "Install",
    "instalar": "install",
    "Ativar": "Activate",
    "ativar": "activate",
    "Verificar": "Verify",
    "verificar": "verify",
    "Executar": "Execute",
    "executar": "execute",
    "Rodar": "Run",
    "rodar": "run",
    "Testar": "Test",
    "testar": "test",
    "Usar": "Use",
    "usar": "use",
    "Configurar": "Configure",
    "configurar": "configure",
    
    # Longer phrases
    "Como usar": "How to use",
    "como usar": "how to use",
    "Passo a passo": "Step by step",
    "passo a passo": "step by step",
    "Exemplo de": "Example of",
    "exemplo": "example",
    "Exemplo:": "Example:",
    "Descrição completa": "Complete description",
    "Documentação completa": "Complete documentation",
    
    # Specific project terms
    "Detecção de Fraude": "Fraud Detection",
    "detecção de fraude": "fraud detection",
    "Transações Bancárias": "Banking Transactions",
    "transações bancárias": "banking transactions",
    "Redes Neurais Spiking": "Spiking Neural Networks",
    "redes neurais spiking": "spiking neural networks",
    "Hardware Neuromórfico": "Neuromorphic Hardware",
    "hardware neuromórfico": "neuromorphic hardware",
    
    # Status and labels
    "Última atualização:": "Last updated:",
    "Versão:": "Version:",
    "Tempo estimado:": "Estimated time:",
    "Resposta esperada:": "Expected response:",
    "Conclusão:": "Conclusion:",
    
    # Instructions
    "Baixe": "Download",
    "Clone o repositório": "Clone the repository",
    "Marque": "Check",
    "Acesse": "Access",
    "Veja": "See",
    
    # File-specific translations
    "# Setup Conda": "# Conda Setup",
    "## 🎯 Configuração Rápida": "## 🎯 Quick Setup",
    "## 📋 Pré-requisitos": "## 📋 Prerequisites",
    "## 🚀 Instalação Automática": "## 🚀 Automatic Installation",
    "## 🔧 Instalação Manual": "## 🔧 Manual Installation",
    "## ✅ Mudanças Implementadas": "## ✅ Implemented Changes",
    "## 🚀 Como Usar": "## 🚀 How to Use",
    
    # Conda migration specific
    "# 🎯 Migração para Conda": "# 🎯 Migration to Conda",
    "Resumo Executivo": "Executive Summary",
    "Arquivos Criados": "Created Files",
    "Notebooks Atualizados": "Updated Notebooks",
    "Setup Inicial": "Initial Setup",
    "Uso Diário": "Daily Usage",
    
    # Common connector words
    " de ": " of ",
    " e ": " and ",
    " para ": " for ",
    " com ": " with ",
    " em ": " in ",
    " a ": " to ",
    " da ": " from the ",
    " do ": " from the ",
    " das ": " from the ",
    " dos ": " from the ",
    " na ": " in the ",
    " no ": " in the ",
    " nas ": " in the ",
    " nos ": " in the ",
}

def translate_text(text: str) -> str:
    """
    Translate Portuguese text to English using the translation dictionary.
    
    Args:
        text: Text to translate
        
    Returns:
        Translated text
    """
    result = text
    
    # Apply translations in order (longer phrases first to avoid partial matches)
    sorted_translations = sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for pt, en in sorted_translations:
        result = result.replace(pt, en)
    
    return result

def translate_file(file_path: Path) -> None:
    """
    Translate a single file from Portuguese to English.
    
    Args:
        file_path: Path to the file to translate
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Translate the content
        translated_content = translate_text(content)
        
        # Only write if content changed
        if translated_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            print(f"✓ Translated: {file_path}")
            return True
        else:
            print(f"- Skipped (no changes): {file_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error translating {file_path}: {e}")
        return False

def main():
    """Main translation function."""
    project_root = Path(__file__).parent
    
    print("=" * 80)
    print("Portuguese to English Translation Script")
    print("=" * 80)
    print()
    
    # Files to translate
    file_patterns = [
        "**/*.md",
        "**/*.py",
        "**/*.ipynb"
    ]
    
    # Directories to skip
    skip_dirs = {
        ".venv", "venv", ".git", "__pycache__", ".pytest_cache",
        "node_modules", ".conda", "data", ".ipynb_checkpoints"
    }
    
    translated_count = 0
    skipped_count = 0
    error_count = 0
    
    for pattern in file_patterns:
        print(f"\nProcessing pattern: {pattern}")
        print("-" * 80)
        
        for file_path in project_root.glob(pattern):
            # Skip files in excluded directories
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            # Skip this script itself
            if file_path.name == "translate_all.py":
                continue
            
            result = translate_file(file_path)
            if result is True:
                translated_count += 1
            elif result is False:
                skipped_count += 1
            else:
                error_count += 1
    
    print()
    print("=" * 80)
    print("Translation Summary")
    print("=" * 80)
    print(f"Files translated: {translated_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    print()
    print("Translation complete!")

if __name__ == "__main__":
    main()
