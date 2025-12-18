#!/usr/bin/env python3
"""
Cleanup script to fix over-translation issues where short Portuguese words
were incorrectly replaced within English words.
"""

from pathlib import Path

# Fix over-translations
CLEANUP_FIXES = {
    # Fix "com" → "with" over-translations
    "compatibility": "compatibility",
    "compatible": "compatible",
    "compute": "compute",
    "completo": "completo",
    "completa": "completa",
    "completos": "completos",
    "completas": "completas",
    "comparison": "comparison",
    "comparação": "comparação",
    "compare": "compare",
    "comparing": "comparing",
    "comment": "comment",
    "commentado": "comentado",
    "comments": "comments",
    "command": "command",
    "commands": "commands",
    "component": "component",
    "components": "components",
    "common": "common",
    "complex": "complex",
    "complexity": "complexity",
    "comprehensive": "comprehensive",
    "comparação": "comparação",
    
    # Fix "se" → "if" over-translations  
    "License": "License",
    "license": "license",
    "Use": "Use",
    "use": "use",
    "uses": "uses",
    "base": "base",
    "baseed": "based",
    "case": "case",
    "cases": "cases",
    "increase": "increase",
    "decrease": "decrease",
    "research": "research",
    "response": "response",
    "response": "response",
    "processing": "processing",
    "processing": "processing",
    "processamento": "processamento",
    "Processamento": "Processamento",
    "second": "second",
    "seconds": "seconds",
    "milliseconds": "milliseconds",
    "Response": "Response",
    "response": "response",
    "sequence": "sequence",
    "sequences": "sequences",
    "sensitive": "sensitive",
    "Cybersecurity": "Cybersecurity",
    "setup": "setup",
    "Setup": "Setup",
    
    # Fix "com" → "with" in URLs and emails
    ".com/": ".com/",
    ".com": ".com",
    "github.com": "github.com",
    "linkedin.com": "linkedin.com",
    
    # Fix "para" → "to" over-translations  
    "separação": "separação",
    "prepare": "prepare",
    "comparação": "comparação",
    "Comparação": "Comparação",
    
    # Fix "por" → "for" over-translations
    "import": "import",
    "Import": "Import",
    "important": "important",
    "Important": "Important",
    "Importante": "Importante",
    "importante": "importante",
    "support": "support",
    "Support": "Support",
    "portifolio": "portifolio",
    "Portifolio": "Portifolio",
    "temporal": "temporal",
    "Temporal": "Temporal",
    "temporal": "temporal",
    "Temporal": "Temporal",
    "temporality": "temporality",
    
    # Fix other common over-translations
    "through": "through",
    "that": "that",
    "that are": "that are",
    "that sure": "that sure",
    "characteristic": "characteristic",
    "characteristic": "characteristic",
    "Characteristic": "Characteristic",
    "that or": "that or",
    "that ore": "that ore",
    "faster": "faster",
    "Faster": "Faster",
    "stayou": "stayou",
    "continuous": "continuous",
    "Continuous": "Continuous",
    "patterns": "patterns",
    "Patterns": "Patterns",
    "output": "output",
    "Output": "Output",
    "outputs": "outputs",
    "asynchronous": "asynchronous",
    "Asynchronous": "Asynchronous",
    
    # Fix system/system typos
    "system": "system",
    "System": "System",
    "sistema": "sistema",
    "System": "System",
    
    # Fix "are" issues
    "are Errors": "are Errors",
    "are expected": "are expected",
    "are valid": "are valid",
    "are Warnings": "are Warnings",
    
    # Fix "for" → "to" where it shouldn't be
    "to ensure": "to ensure",
    "to ensure": "to ensure",
    
    # Fix Portuguese that should stay
    "Mauro Risonho de Paula Assumpção": "Mauro Risonho de Paula Assumpção",
    
    # Fix broken anchors
    "#-overview": "#-overview",
    "#-why-": "#-why-",
    "#-architecture-of-the-": "#-architecture-of-the-",
    "#-sistema": "#-system",
    "#-using-the-api": "#-using-the-api",
    
    # Additional specific fixes
    "you": "you",
    "Alternative": "Alternative",
    "Recommended": "Recommended",
}

def cleanup_file(file_path: Path) -> bool:
    """Clean up over-translations in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply all fixes
        for wrong, correct in CLEANUP_FIXES.items():
            content = content.replace(wrong, correct)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Cleaned: {file_path.name}")
            return True
        return False
            
    except Exception as e:
        print(f"✗ Error: {file_path.name} - {e}")
        return False

def main():
    """Main cleanup function."""
    project_root = Path(__file__).parent
    
    print("=" * 80)
    print("Cleanup Over-translations")
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
                
            if cleanup_file(file_path):
                fixed_count += 1
    
    print()
    print("=" * 80)
    print(f"Cleanup complete! Fixed {fixed_count} files.")
    print("=" * 80)

if __name__ == "__main__":
    main()
