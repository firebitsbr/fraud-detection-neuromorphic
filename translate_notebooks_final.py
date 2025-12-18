#!/usr/bin/env python3
"""
Final comprehensive notebook translation script
Translates ALL remaining Portuguese content to English
"""

import json
import re
from pathlib import Path
from typing import Dict

# Comprehensive translation dictionary
TRANSLATIONS = {
    # Common Portuguese words and phrases
    "Autor": "Author",
    "Data de Criação": "Creation Date",
    "Licença": "License",
    "Desenvolvimento": "Development",
    "Descrição": "Description",
    
    # Months
    "dezembro": "December",
    "Janeiro": "January",
    
    # Cell content translations
    "Imports completados": "Imports completed",
    "Configurando Simulação": "Configuring Simulation",
    "Duração": "Duration",
    "Executando Simulação": "Executing Simulation",
    "Simulação concluída": "Simulation completed",
    "Resultados": "Results",
    "Spikes pré-sinápticos": "Pre-synaptic spikes",
    "Spikes pós-sinápticos": "Post-synaptic spikes",
    "Peso inicial": "Initial weight",
    "Peso final": "Final weight",
    "Mudança": "Change",
    "neurônio": "neuron",
    "Gráfico de Rastros": "Raster Plot",
    "Evolução do Peso Sináptico": "Evolution of Synaptic Weight",
    "Potencial de Membrana": "Membrane Potential",
    "tempo": "time",
    "Peso Sináptico": "Synaptic Weight",
    "Voltagem": "Voltage",
    "Limiar": "Threshold",
    "Repouso": "Resting",
    
    # Configuration
    "Configurando Simulação com múltiplos neurônios": "Configuring Simulation with multiple neurons",
    "neurônios pré-sinápticos": "pre-synaptic neurons",
    "Gerar padrão temporal": "Generate temporal pattern",
    "neurônio": "neuron",
    "spikes regulares": "regular spikes",
    "levemente atrasado": "slightly delayed",
    "mais atrasado": "more delayed",
    "spikes esparsos": "sparse spikes",
    "fase diferente": "different phase",
    "padrões de spikes": "spike patterns",
    
    # Analysis
    "Análise dos pesos Sinápticos": "Analysis of Synaptic Weights",
    "melhor": "best",
    "Comparação antes/depois": "Comparison before/after",
    "Inicial": "Initial",
    "Final": "Final",
    "Interpretação": "Interpretation",
    "neurônios que disparam consistentemente ANTES do pós-sináptico são reforçados": 
        "neurons that fire consistently BEFORE the post-synaptic are reinforced",
    "neurônios com timing inconsistente têm pesos reduzidos":
        "neurons with inconsistent timing have reduced weights",
    "A rede aprende a correlação temporal automaticamente":
        "The network learns temporal correlation automatically",
        
    # Fraud detection context
    "Como STDP ajuda na detecção de fraude": "How STDP helps in fraud detection",
    "Cenário": "Scenario",
    "Sequência Temporal Normal": "Normal Temporal Sequence",
    "Transação Legítima": "Legitimate Transaction",
    "Login no app": "Login to app",
    "Navegação no saldo": "Balance navigation",
    "Seleção de beneficiário conhecido": "Selection of known beneficiary",
    "Confirmação de pagamento": "Payment confirmation",
    "STDP aprende": "STDP learns",
    "Sequência causal esperada": "Expected causal sequence",
    "Intervalos temporais normais": "Normal time intervals",
    "Reforça conexões que representam comportamento legítimo":
        "Reinforces connections that represent legitimate behavior",
        
    "Sequência Anômala": "Anomalous Sequence",
    "Transação Fraudulenta": "Fraudulent Transaction",
    "Transferência imediata sem navegação": "Immediate transfer without navigation",
    "valor alto para novo beneficiário": "high value to new beneficiary",
    "Localização geográfica inconsistente": "Inconsistent geographic location",
    "STDP detecta": "STDP detects",
    "padrão temporal anômalo": "anomalous temporal pattern",
    "Sequência não reforçada durante treinamento": "Sequence not reinforced during training",
    "alta ativação de neurônios de fraude": "high activation of fraud neurons",
    
    # Advantages
    "Vantagens do STDP": "Advantages of STDP",
    "Aprendizado não supervisionado": "Unsupervised learning",
    "não precisa de labels explícitos inicialmente": "does not need explicit labels initially",
    "Adaptação contínua": "Continuous adaptation",
    "Aprende novos padrões de fraude automaticamente": "Learns new fraud patterns automatically",
    "Sensibilidade temporal": "Temporal sensitivity",
    "Detecta anomalias na sequência de eventos": "Detects anomalies in event sequence",
    "Eficiência": "Efficiency",
    "Atualização local de pesos": "Local weight update",
    "sem backpropagation": "without backpropagation",
    "Biologicamente plausível": "Biologically plausible",
    "Inspirado no cérebro humano": "Inspired by human brain",
    
    # Conclusions
    "Conclusões": "Conclusions",
    "mecanismo": "mechanism",
    "Aprende correlações temporais entre características de transação":
        "Learns temporal correlations between transaction features",
    "Reforça padrões legítimos frequentes": "Reinforces frequent legitimate patterns",
    "Detecta desvios na sequência temporal": "Detects deviations in temporal sequence",
    "Aplicações Práticas": "Practical Applications",
    "Análise de comportamento": "Behavioral analysis",
    "Sequência de ações no mobile banking": "Sequence of actions in mobile banking",
    "Detecção de velocidade": "Speed detection",
    "Transações impossíveis": "Impossible transactions",
    "compras em cidades diferentes em poucos minutos": "purchases in different cities within minutes",
    "padrões de uso": "usage patterns",
    "Horários, frequência, valores típicos": "Times, frequency, typical values",
    "Navegação suspeita": "Suspicious navigation",
    "Sequências de páginas atípicas": "Atypical page sequences",
    
    # Comparison
    "Comparação com métodos Tradicionais": "Comparison with Traditional methods",
    "característica": "feature",
    "Processamento temporal": "Temporal processing",
    "Nativo": "Native",
    "Emulado": "Emulated",
    "Supervisão": "Supervision",
    "sim": "yes",
    "não": "no",
    "latência": "latency",
    "Ultra-baixa": "Ultra-low",
    "alta": "high",
    "Consumo energético": "Energy consumption",
    "muito baixo": "very low",
    "Adaptação online": "Online adaptation",
    "Difícil": "Difficult",
    "Hardware especializado": "Specialized hardware",
    
    # Future
    "Futuro": "Future",
    "Chips neuromórficos dedicados": "Dedicated neuromorphic chips",
    "modulação de recompensa": "reward modulation",
    "dopamina artificial": "artificial dopamine",
    "Aprendizado federado": "Federated learning",
    "Explicabilidade": "Explainability",
    "Visualizar pesos aprendidos": "Visualize learned weights",
    
    # Project
    "Projeto": "Project",
    "Computação Neuromórfica para Cibersegurança Bancária":
        "Neuromorphic Computing for Banking Cybersecurity",
}

def translate_text(text: str) -> str:
    """Translate Portuguese text to English"""
    result = text
    for pt, en in TRANSLATIONS.items():
        # Case-insensitive replacement
        result = re.sub(re.escape(pt), en, result, flags=re.IGNORECASE)
    return result

def translate_notebook(notebook_path: Path) -> None:
    """Translate all Portuguese content in a notebook"""
    print(f"Processing {notebook_path.name}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes = 0
    for cell in notebook.get('cells', []):
        if 'source' in cell:
            original = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            translated = translate_text(original)
            
            if original != translated:
                changes += 1
                if isinstance(cell['source'], list):
                    cell['source'] = [translated]
                else:
                    cell['source'] = translated
    
    if changes > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        print(f"  ✓ Translated {changes} cells")
    else:
        print(f"  ✓ No changes needed")

def main():
    """Main translation function"""
    notebooks_dir = Path(__file__).parent / 'notebooks'
    
    notebooks = [
        '01-stdp_example.ipynb',
        '02-stdp-demo.ipynb',
        '03-loihi_benchmark.ipynb',
        '04_brian2_vs_snntorch.ipynb',
        '05_production_solutions.ipynb',
        '06_phase1_integration.ipynb',
    ]
    
    print("="*70)
    print("FINAL COMPREHENSIVE NOTEBOOK TRANSLATION")
    print("="*70)
    print()
    
    for notebook_name in notebooks:
        notebook_path = notebooks_dir / notebook_name
        if notebook_path.exists():
            translate_notebook(notebook_path)
        else:
            print(f"  ⚠ Not found: {notebook_name}")
    
    print()
    print("="*70)
    print("✓ TRANSLATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
