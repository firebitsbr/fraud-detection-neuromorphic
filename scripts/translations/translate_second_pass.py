#!/usr/bin/env python3
"""
Second pass translation to fix remaining Portuguese content and improve quality.
"""

import re
from pathlib import Path

# Additional translations for second pass
ADDITIONAL_TRANSLATIONS = {
    # Author name fix
    "Mauro Risonho of Paula Assumpção": "Mauro Risonho de Paula Assumpção",
    
    # Date fixes
    "Date:** Dezembro 2025": "Date:** December 2025",
    "Dezembro 2025": "December 2025",
    "dezembro 2025": "December 2025",
    
    # Area fix
    "Area:** Computação Neuromórfica": "Area:** Neuromorphic Computing",
    
    # Common phrases that were partially translated
    "from the Sistema": "of the System",
    "from the Projeto": "of the Project",
    "from the model": "of the model",
    "in the model": "in the model",
    
    # Fix problematic prepositions
    "usando": "using",
    "entre": "between",
    "para garantir": "to ensure",
    "compatibilidade perfeita": "perfect compatibility",
    "ou": "or",
    "se ainda não fez": "if you haven't done so yet",
    "Execute o script": "Execute the script",
    "Ative o": "Activate the",
    "Verifique": "Verify",
    "minutos": "minutes",
    "Alternativa": "Alternative",
    "Recomendado": "Recommended",
    "compatível": "compatible",
    "atualizados": "updated",
    "instalado": "installed",
    "superior": "or higher",
    
    # Technical terms
    "Este projeto agora usa": "This project now uses",
    "agora": "now",
    "ainda": "still",
    "sempre": "always",
    "também": "also",
    "mais": "more",
    "menos": "less",
    "muito": "very",
    "bem": "well",
    "como": "as",
    "quando": "when",
    "onde": "where",
    "porque": "because",
    "então": "then",
    "assim": "thus",
    "após": "after",
    "antes": "before",
    "durante": "during",
    "depois": "after",
    
    # Project specific
    "Sistema completo": "Complete system",
    "fraud detection bancária": "bank fraud detection",
    "redes neurais que funcionam": "neural networks that work",
    "cérebro humano": "human brain",
    "processando informação através": "processing information through",
    "pulsos elétricos temporais": "temporal electrical pulses",
    "Bancos": "Banks",
    "fintechs": "fintechs",
    "processam": "process",
    "milhões": "millions",
    "transações por segundo": "transactions per second",
    "Sistemas tradicionais consomem muita energia": "Traditional systems consume a lot of energy",
    "têm latência alta": "have high latency",
    "rodando": "running",
    "oferecem": "offer",
    "instantânea": "instant",
    "Resposta": "Response",
    "milissegundos": "milliseconds",
    "extrema": "extreme",
    "energia que": "energy than",
    "Pode": "Can",
    "dispositivos móveis": "mobile devices",
    "contínuo": "continuous",
    "Adapta-se": "Adapts",
    "novos padrões": "new patterns",
    "fraude": "fraud",
    
    # More fixes
    "mais rápido que": "faster than",
    "mais eficiente": "more efficient",
    "menos que": "less than",
    "Equivalente": "Equivalent",
    "Problema": "Problem",
    "Tradicional": "Traditional",
    "Consomem": "Consume",
    "muita": "a lot of",
    "Processam": "Process",
    "lotes": "batches",
    "exploram": "exploit",
    "temporalidade nativa": "native temporality",
    "tradicional": "traditional",
    "assíncrono": "asynchronous",
    "ultra-baixa": "ultra-low",
    "temporal nativo": "native temporal",
    
    # Table of contents fixes
    "Visão Geral": "Overview",
    "Computação Neuromórfica": "Neuromorphic Computing",
    "Arquitetura": "Architecture",
    "Quick Start": "Quick Start",
    "Manual Installation": "Manual Installation",
    "Executando os Notebooks": "Running the Notebooks",
    "Usando": "Using",
    "Testes": "Tests",
    "Validação": "Validation",
    "Resultados": "Results",
    "Documentação Detalhada": "Detailed Documentation",
    "Estrutura": "Structure",
    "Tecnologias": "Technologies",
    "Contribuindo": "Contributing",
    "Referências": "References",
    
    # More common words
    "que": "that",
    "não": "not",
    "são": "are",
    "está": "is",
    "foram": "were",
    "será": "will be",
    "foi": "was",
    "ter": "have",
    "fazer": "do",
    "pode": "can",
    "deve": "should",
    "precisa": "needs",
    "quer": "wants",
    "sabe": "knows",
    "tem": "has",
}

def second_pass_translate(text: str) -> str:
    """Apply second pass translations."""
    result = text
    
    # Sort by length (longer first) to avoid partial replacements
    sorted_translations = sorted(
        ADDITIONAL_TRANSLATIONS.items(), 
        key=lambda x: len(x[0]), 
        reverse=True
    )
    
    for pt, en in sorted_translations:
        result = result.replace(pt, en)
    
    return result

def process_file(file_path: Path) -> bool:
    """Process a single file with second pass translation."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        translated = second_pass_translate(content)
        
        if translated != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated)
            print(f"✓ Fixed: {file_path.name}")
            return True
        return False
            
    except Exception as e:
        print(f"✗ Error: {file_path.name} - {e}")
        return False

def main():
    """Main function for second pass translation."""
    project_root = Path(__file__).parent
    
    print("=" * 80)
    print("Second Pass Translation - Quality Improvements")
    print("=" * 80)
    print()
    
    patterns = ["**/*.md", "**/*.py", "**/*.ipynb"]
    skip_dirs = {".venv", "venv", ".git", "__pycache__", "node_modules", ".conda", "data"}
    
    fixed_count = 0
    
    for pattern in patterns:
        for file_path in project_root.glob(pattern):
            if any(skip in file_path.parts for skip in skip_dirs):
                continue
            if file_path.name in ["translate_all.py", "translate_second_pass.py"]:
                continue
                
            if process_file(file_path):
                fixed_count += 1
    
    print()
    print("=" * 80)
    print(f"Second pass complete! Fixed {fixed_count} files.")
    print("=" * 80)

if __name__ == "__main__":
    main()
