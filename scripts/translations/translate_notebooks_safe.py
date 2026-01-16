#!/usr/bin/env python3
"""
Safe notebook translation - only translates content, not JSON structure.
"""

import json
from pathlib import Path

# Translation dictionary - same as before but will only apply to cell content
TRANSLATIONS = {
    # Common Portuguese to English translations
    "Descrição:": "Description:",
    "Autor:": "Author:",
    "Data de Criação:": "Creation Date:",
    "Licença:": "License:",
    "Desenvolvimento:": "Development:",
    
    # Specific phrases - longer first
    "Humano + Desenvolvimento por AI Assistida": "Human + AI Assisted Development",
    "Desenvolvimento por AI Assistida": "AI Assisted Development",
    "Human + Development by AI Assisted": "Human + AI Assisted Development",
    "Development by AI Assisted": "AI Assisted Development",
    "Humano + Desenvolvimento": "Human + Development",
    "Assistida": "Assisted",
    
    # Full phrases
    "Tutorial interativo sobre o mecanismo de aprendizado biológico": "Interactive tutorial about the biological learning mechanism",
    "Aprendizado Biológico": "Biological Learning",
    "utilizado em redes neurais neuromórficas": "used in neuromorphic neural networks",
    "Demonstra como neurônios aprendem correlações temporais automaticamente": "Demonstrates how neurons learn temporal correlations automatically",
    
    # Dates
    "de Dezembro de": "of December",
    "Dezembro": "December",
    
    # Common words
    "sobre": "about",
    "como": "how",
    "automaticamente": "automatically",
    "interativo": "interactive",
    "Importações concluídas": "Imports completed",
    "Exemplo": "Example",
    "exemplo": "example",
    "Configuração": "Configuration",
    "configuração": "configuration",
    "Execução": "Execution",
    "execução": "execution",
    "Resultados": "Results",
    "resultados": "results",
    "Visualização": "Visualization",
    "visualização": "visualization",
    "Análise": "Analysis",
    "análise": "analysis",
    "Demonstração": "Demonstration",
    "demonstração": "demonstration",
    "Conclusão": "Conclusion",
    "conclusão": "conclusion",
}

def translate_text(text):
    """Translate text using the translation dictionary."""
    result = text
    # Sort by length to handle longer phrases first
    for pt, en in sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        result = result.replace(pt, en)
    return result

def translate_notebook(notebook_path):
    """Translate notebook content safely without corrupting JSON structure."""
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        changes_made = False
        
        # Process each cell
        for cell in notebook.get('cells', []):
            # Only translate markdown cells and code comments
            if cell.get('cell_type') == 'markdown':
                # Translate markdown source
                if 'source' in cell:
                    original_source = cell['source']
                    if isinstance(original_source, list):
                        translated_source = [translate_text(line) for line in original_source]
                        if translated_source != original_source:
                            cell['source'] = translated_source
                            changes_made = True
                    elif isinstance(original_source, str):
                        translated = translate_text(original_source)
                        if translated != original_source:
                            cell['source'] = translated
                            changes_made = True
            
            elif cell.get('cell_type') == 'code':
                # Only translate comments in code cells
                if 'source' in cell:
                    original_source = cell['source']
                    if isinstance(original_source, list):
                        translated_source = []
                        for line in original_source:
                            # Only translate lines that are comments
                            if line.strip().startswith('#'):
                                translated_source.append(translate_text(line))
                                if translate_text(line) != line:
                                    changes_made = True
                            else:
                                translated_source.append(line)
                        if changes_made:
                            cell['source'] = translated_source
                    elif isinstance(original_source, str):
                        # Handle single string source
                        lines = original_source.split('\n')
                        translated_lines = []
                        for line in lines:
                            if line.strip().startswith('#'):
                                translated_lines.append(translate_text(line))
                                if translate_text(line) != line:
                                    changes_made = True
                            else:
                                translated_lines.append(line)
                        if changes_made:
                            cell['source'] = '\n'.join(translated_lines)
        
        # Save only if changes were made
        if changes_made:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=1)
            print(f"✓ Translated: {notebook_path.name}")
            return True
        else:
            print(f"- No changes: {notebook_path.name}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {notebook_path.name} - {e}")
        return False

def main():
    """Main function."""
    project_root = Path(__file__).parent
    notebooks_dir = project_root / 'notebooks'
    
    print("=" * 80)
    print("Safe Notebook Translation")
    print("=" * 80)
    print()
    
    if not notebooks_dir.exists():
        print("Error: notebooks/ directory not found")
        return
    
    translated_count = 0
    
    for notebook_path in notebooks_dir.glob('*.ipynb'):
        if translate_notebook(notebook_path):
            translated_count += 1
    
    print()
    print("=" * 80)
    print(f"Translation complete! Translated {translated_count} notebooks.")
    print("=" * 80)

if __name__ == "__main__":
    main()
