#!/usr/bin/env python3
"""
Final comprehensive translation pass - handles complete sentences and remaining Portuguese.
"""

import re
from pathlib import Path

# Comprehensive sentence-level translations
SENTENCE_TRANSLATIONS = {
    # Complete sentences (must be first)
    "Este projeto agora usa": "This project now uses",
    "Este projeto é": "This project is",
    "garantir compatibilidade perfeita entre": "ensure perfect compatibility between",
    "se ainda não fez": "if you haven't done so yet",
    "Execute o script of setup": "Execute the setup script",
    "Ative o environment": "Activate the environment",
    "Verifique GPU": "Verify GPU",
    
    # Common remaining Portuguese words
    "garantir": "ensure",
    "entre:": "between:",
    "instalado": "installed",
    "atualizados": "updated",
    "compatível": "compatible",
    "superior": "higher",
    "minutos": "minutes",
    "ou": "or",
    "se": "if",
    "fez": "done",
    "ainda": "still",
    "não": "not",
    "são": "are",
    "está": "is",
    "foram": "were",
    "será": "will be",
    "foi": "was",
    "sim": "yes",
    "pelo": "by the",
    "pela": "by the",
    "pelos": "by the",
    "pelas": "by the",
    "com": "with",
    "sem": "without",
    "sob": "under",
    "sobre": "about",
    "para": "to",
    "por": "for",
    "até": "until",
    "desde": "since",
    "segundo": "according to",
    "conforme": "according to",
    "durante": "during",
    "mediante": "through",
    "perante": "before",
    "através": "through",
    "contra": "against",
    "entre": "between",
    "após": "after",
    "antes": "before",
    
    # Remaining verbs
    "usam": "use",
    "usa": "uses",
    "usar": "use",
    "fazer": "do",
    "faz": "does",
    "fazem": "do",
    "ter": "have",
    "tem": "has",
    "têm": "have",
    "haver": "there is",
    "há": "there is",
    "ser": "be",
    "sendo": "being",
    "sido": "been",
    "estar": "be",
    "estando": "being",
    "estado": "been",
    "ficar": "stay",
    "fica": "stays",
    "ficam": "stay",
    "poder": "can",
    "pode": "can",
    "podem": "can",
    "dever": "should",
    "deve": "should",
    "devem": "should",
    "saber": "know",
    "sabe": "knows",
    "sabem": "know",
    "querer": "want",
    "quer": "wants",
    "querem": "want",
    
    # Adjectives and adverbs
    "completa": "complete",
    "completo": "complete",
    "completos": "complete",
    "completas": "complete",
    "válido": "valid",
    "válida": "valid",
    "válidos": "valid",
    "válidas": "valid",
    "customizado": "custom",
    "customizados": "custom",
    "reconhecido": "recognized",
    "reconhecidos": "recognized",
    "esperado": "expected",
    "esperados": "expected",
    "detectado": "detected",
    "detectados": "detected",
    "comentado": "commented",
    "importado": "imported",
    "importados": "imported",
    "usadas": "used",
    "comum": "common",
    "menores": "minor",
    "principais": "main",
    "próximos": "next",
    "Próximos": "Next",
    "recomendados": "recommended",
    "Recomendados": "Recommended",
    "recomendadas": "recommended",
    "Recomendadas": "Recommended",
    
    # Nouns
    "Módulos": "Modules",
    "módulos": "modules",
    "Avisos": "Warnings",
    "avisos": "warnings",
    "Erros": "Errors",
    "erros": "errors",
    "Variáveis": "Variables",
    "variáveis": "variables",
    "Células": "Cells",
    "células": "cells",
    "Comentários": "Comments",
    "comentários": "comments",
    "Código": "Code",
    "código": "code",
    "Recomendações": "Recommendations",
    "recomendações": "recommendations",
    "Passos": "Steps",
    "passos": "steps",
    
    # Specific technical phrases
    "magic commands": "magic commands",
    "mas not": "but not",
    "but not": "but not",
    "canm be": "can be",
    "can be": "can be",
    "bem comentado": "well commented",
    "comportamento esperado": "expected behavior",
    "afetam execution": "affect execution",
    "not afetam": "do not affect",
    "não afetam": "do not affect",
    
    # Fix broken translations
    "syshas": "system",
    "Syshas": "System",
    "throrgh": "through",
    "fashave": "faster",
    "carachaveística": "characteristic",
    "Carachaveística": "Characteristic",
    "throrghput": "throughput",
    "Throrghput": "Throughput",
    "continuors": "continuous",
    "pathavens": "patterns",
    "are Erros": "are Errors",
    "are esperados": "are expected",
    "Não are": "Not are",
    "that are válidos": "that are valid",
    "in Jupyhave": "in Jupyter",
    "canm be": "can be",
    "sem erro": "without error",
    "Demonstração completa": "Complete demonstration",
    "Sintaxe válida": "Valid syntax",
    
    # Common patterns to fix
    "of the": "of the",
    "in the": "in the",
    "to the": "to the",
    "from the": "from the",
    "by the": "by the",
    "with the": "with the",
    "at the": "at the",
    "on the": "on the",
}

def final_pass_translate(text: str) -> str:
    """Apply final comprehensive translation."""
    result = text
    
    # Sort by length (longer first)
    sorted_trans = sorted(
        SENTENCE_TRANSLATIONS.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    for pt, en in sorted_trans:
        result = result.replace(pt, en)
    
    # Fix double translations
    result = result.replace("  ", " ")
    result = result.replace("ofof", "of")
    result = result.replace("toto", "to")
    result = result.replace("inin", "in")
    result = result.replace("fromfrom", "from")
    
    return result

def process_file(file_path: Path) -> bool:
    """Process a file with final translation pass."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        translated = final_pass_translate(content)
        
        if translated != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated)
            print(f"✓ Final pass: {file_path.name}")
            return True
        return False
            
    except Exception as e:
        print(f"✗ Error: {file_path.name} - {e}")
        return False

def main():
    """Main function."""
    project_root = Path(__file__).parent
    
    print("=" * 80)
    print("Final Comprehensive Translation Pass")
    print("=" * 80)
    print()
    
    patterns = ["**/*.md", "**/*.py", "**/*.ipynb"]
    skip_dirs = {".venv", "venv", ".git", "__pycache__", "node_modules", ".conda", "data"}
    
    fixed_count = 0
    
    for pattern in patterns:
        for file_path in project_root.glob(pattern):
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            if "translate" in file_path.name:
                continue
                
            if process_file(file_path):
                fixed_count += 1
    
    print()
    print("=" * 80)
    print(f"Final pass complete! Fixed {fixed_count} files.")
    print("=" * 80)

if __name__ == "__main__":
    main()
