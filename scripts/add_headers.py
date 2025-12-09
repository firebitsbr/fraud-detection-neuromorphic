#!/usr/bin/env python3
"""
**Descrição:** Script para adicionar cabeçalhos padronizados aos arquivos Python.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import os
import re
from pathlib import Path

# Standard header template
HEADER_TEMPLATE = '''"""
{description}

Autor: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
Linkedin: https://www.linkedin.com/in/maurorisonho
github: https://github.com/maurorisonho
Data de criação: Dezembro 2025
LICENSE MIT
Desenvolvimento: Humano + Desenvolvimento por AI Assitida (Claude Sonnet 4.5, Gemini 3 Pro Preview).
"""
'''

# File descriptions
FILE_DESCRIPTIONS = {
    'src/main.py': 'Complete fraud detection pipeline using Spiking Neural Networks.',
    'src/models_snn.py': 'SNN implementation using Brian2 with LIF neurons and STDP learning.',
    'src/models_snn_snntorch.py': 'SNN implementation using snnTorch framework.',
    'src/models_snn_pytorch.py': 'PyTorch-based SNN implementation for fraud detection.',
    'src/encoders.py': 'Spike encoding schemes for transaction data.',
    'src/advanced_encoders.py': 'Advanced spike encoding techniques.',
    'src/dataset_loader.py': 'Dataset loading and preprocessing utilities.',
    'src/dataset_kaggle.py': 'Kaggle IEEE Fraud Detection dataset loader.',
    'src/api_server.py': 'FastAPI REST server for fraud detection inference.',
    'src/performance_profiler.py': 'Performance profiling and benchmarking tools.',
    'src/hyperparameter_optimizer.py': 'Hyperparameter optimization for SNNs.',
    'src/model_comparator.py': 'Model comparison and evaluation utilities.',
    'src/explainability.py': 'Model explainability and interpretability tools.',
    'src/cost_optimization.py': 'Cloud cost optimization and auto-scaling strategies.',
    'src/performance_optimization.py': 'Performance optimization techniques.',
    'src/overfitting_prevention.py': 'Overfitting prevention and regularization methods.',
    'src/security.py': 'Security hardening and authentication utilities.',
    'api/main.py': 'FastAPI REST API for neuromorphic fraud detection.',
    'api/models.py': 'Pydantic models for API request/response validation.',
    'api/monitoring.py': 'API monitoring, metrics collection, and alerting.',
    'api/kafka_integration.py': 'Kafka streaming integration for real-time detection.',
    'hardware/loihi_adapter.py': 'Intel Loihi 2 hardware adapter.',
    'hardware/loihi_simulator.py': 'Intel Loihi simulator implementation.',
    'hardware/loihi2_simulator.py': 'Intel Loihi 2 advanced simulator.',
    'hardware/brainscales2_simulator.py': 'BrainScaleS-2 neuromorphic hardware simulator.',
    'hardware/energy_benchmark.py': 'Energy efficiency benchmarking suite.',
    'hardware/deploy_model.py': 'Model deployment to neuromorphic hardware.',
    'scaling/distributed_cluster.py': 'Multi-chip distributed neuromorphic processing.',
    'tests/test_models_snn.py': 'Unit tests for SNN models.',
    'tests/test_encoders.py': 'Unit tests for spike encoders.',
    'tests/test_main.py': 'Integration tests for main pipeline.',
    'tests/test_integration.py': 'End-to-end integration tests.',
    'tests/test_scaling.py': 'Scalability and distributed processing tests.',
    'tests/run_tests.py': 'Test runner with reporting.',
    'tests/conftest.py': 'Pytest configuration and fixtures.',
    'scripts/download_kaggle_dataset.py': 'Automated Kaggle dataset download script.',
    'scripts/manual_download_helper.py': 'Interactive manual download helper.',
    'scripts/manual_kaggle_setup.py': 'Interactive Kaggle dataset setup with auto-detection.',
    'scripts/deploy.sh': 'Deployment script for production environments.',
    'examples/api_client.py': 'Example API client for fraud detection.',
    'examples/kafka_producer_example.py': 'Example Kafka producer for transactions.',
    'examples/load_test.py': 'Load testing script for API endpoints.',
}

def has_author_info(content):
    """Check if file already has author information."""
    return 'Mauro Risonho' in content or 'mauro.risonho@gmail.com' in content

def extract_existing_description(content):
    """Extract existing docstring description if present."""
    match = re.match(r'^"""(.*?)"""', content, re.DOTALL)
    if match:
        desc = match.group(1).strip()
        # Remove author info if present
        lines = [l for l in desc.split('\n') if not any(k in l.lower() for k in ['author', 'email', 'linkedin', 'github', 'date', 'license'])]
        return '\n'.join(lines).strip()
    return None

def add_header_to_file(filepath, description=None):
    """Add standardized header to a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already has complete author info
        if has_author_info(content) and 'linkedin.com/in/maurorisonho' in content:
            print(f"✓ {filepath} - Already has complete header")
            return False
        
        # Try to extract existing description
        existing_desc = extract_existing_description(content)
        
        # Use provided description or existing or filename
        if description:
            desc = description
        elif existing_desc:
            desc = existing_desc
        else:
            desc = f"Module: {Path(filepath).name}"
        
        # Remove old docstring if present
        content = re.sub(r'^""".*?"""[\n]*', '', content, count=1, flags=re.DOTALL)
        
        # Remove shebang temporarily
        shebang = ''
        if content.startswith('#!'):
            lines = content.split('\n', 1)
            shebang = lines[0] + '\n'
            content = lines[1] if len(lines) > 1 else ''
        
        # Create new header
        header = HEADER_TEMPLATE.format(description=desc)
        
        # Combine
        new_content = shebang + header + '\n' + content.lstrip()
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ {filepath} - Header added")
        return True
        
    except Exception as e:
        print(f"❌ {filepath} - Error: {e}")
        return False

def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    
    print("="*70)
    print("Adding standardized headers to Python files".center(70))
    print("="*70 + "\n")
    
    updated_count = 0
    skipped_count = 0
    
    # Process all files with known descriptions
    for rel_path, description in FILE_DESCRIPTIONS.items():
        filepath = project_root / rel_path
        if filepath.exists():
            if add_header_to_file(filepath, description):
                updated_count += 1
            else:
                skipped_count += 1
    
    # Find other Python files
    print(f"\n{'='*70}")
    print("Scanning for other Python files...".center(70))
    print(f"{'='*70}\n")
    
    for pattern in ['src/**/*.py', 'api/**/*.py', 'tests/**/*.py', 'scripts/**/*.py', 
                    'hardware/**/*.py', 'scaling/**/*.py', 'examples/**/*.py']:
        for filepath in project_root.glob(pattern):
            if filepath.name == '__init__.py':
                continue
            
            rel_path = str(filepath.relative_to(project_root))
            if rel_path not in FILE_DESCRIPTIONS:
                if add_header_to_file(filepath):
                    updated_count += 1
                else:
                    skipped_count += 1
    
    print(f"\n{'='*70}")
    print(f"✅ Updated: {updated_count} files")
    print(f"✓ Skipped: {skipped_count} files (already have headers)")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
