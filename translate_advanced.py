#!/usr/bin/env python3
"""
Script avançado para tradução COMPLETA e CORRETA de notebooks para português.
"""

import json
import re
from pathlib import Path

def translate_comprehensive(text):
    """Traduz texto com mapeamento completo e contextual."""
    if not isinstance(text, str) or not text.strip():
        return text
    
    # Preservar código Python (linhas que começam com comandos Python)
    if re.match(r'^\s*(import|from|def|class|if|for|while|try|except|with|return|print|#!)', text):
        # Traduzir apenas comentários
        if text.strip().startswith('#') and not text.strip().startswith('#!/'):
            text = text.replace('# Install', '# Instalar')
            text = text.replace('# Specific', '# Específico')
            text = text.replace('if not yet installed', 'se ainda não instalado')
            text = text.replace('instead of wildcard', 'ao invés de wildcard')
            text = text.replace('Configure to use', 'Configurar para usar')
            text = text.replace('avoids error of compilation', 'evita erro de compilação')
            text = text.replace('if headers are missing', 'se os headers estiverem faltando')
            text = text.replace('Imports completed!', 'Importações concluídas!')
        return text
    
    # Preservar URLs
    if 'http://' in text or 'https://' in text:
        text = text.replace('Open In Colab', 'Abrir no Colab')
        return text
    
    # Dicionário de traduções contextuais (ordem importa!)
    translations = [
        # Títulos principais
        ('STDP Example: Biological Learning', 'Exemplo STDP: Aprendizado Biológico'),
        ('Demonstration: Neuromorphic Fraud Detection', 'Demonstração: Detecção de Fraude Neuromórfica'),
        ('Hardware Benchmark: Loihi vs CPU', 'Benchmark de Hardware: Loihi vs CPU'),
        ('Brian2 vs snnTorch vs BindsNET: Complete Comparison', 'Brian2 vs snnTorch vs BindsNET: Comparação Completa'),
        ('Production Solutions and Optimization', 'Soluções para Produção e Otimização'),
        ('Phase 1: Complete Integration', 'Fase 1: Integração Completa'),
        
        # Descrições longas
        ('Interactive Tutorial about the biological learning mechanism', 'Tutorial Interativo sobre o mecanismo biológico de aprendizado'),
        ('Spike-Timing-Dependent Plasticity', 'Plasticidade Dependente do Tempo de Spike'),
        ('used in neuromorphic neural networks', 'usado em redes neurais neuromórficas'),
        ('Demonstrates how neurons learn temporal correlations automatically', 'Demonstra como os neurônios aprendem correlações temporais automaticamente'),
        
        # Metadados
        ('Description:', 'Descrição:'),
        ('Author:', 'Autor:'),
        ('Creation Date:', 'Data de Criação:'),
        ('License:', 'Licença:'),
        ('Development:', 'Desenvolvimento:'),
        ('Human + AI Assisted Development', 'Desenvolvimento Humano + Assistido por IA'),
        ('AI-Assisted Development', 'Desenvolvimento Assistido por IA'),
        ('December', 'Dezembro'),
        ('January', 'Janeiro'),
        ('February', 'Fevereiro'),
        ('March', 'Março'),
        ('April', 'Abril'),
        ('May', 'Maio'),
        ('June', 'Junho'),
        ('July', 'Julho'),
        ('August', 'Agosto'),
        ('September', 'Setembro'),
        ('October', 'Outubro'),
        ('November', 'Novembro'),
        
        # Frases completas
        ('This notebook explores the biological learning mechanism', 'Este notebook explora o mecanismo biológico de aprendizado'),
        ('This notebook explores', 'Este notebook explora'),
        ('This notebook demonstrates', 'Este notebook demonstra'),
        ('What is STDP?', 'O que é STDP?'),
        ('STDP is an unsupervised learning rule inspired by biological neurons', 'STDP é uma regra de aprendizado não supervisionado inspirada por neurônios biológicos'),
        ('if the pre-synaptic neuron fires BEFORE the post-synaptic', 'se o neurônio pré-sináptico dispara ANTES do pós-sináptico'),
        ('if the pre-synaptic neuron fires AFTER the post-synaptic', 'se o neurônio pré-sináptico dispara DEPOIS do pós-sináptico'),
        ('This allows the network to learn temporal causal relationships without explicit labels', 'Isso permite que a rede aprenda relações causais temporais sem rótulos explícitos'),
        ('This allows the network to learn', 'Isso permite que a rede aprenda'),
        ('without explicit labels', 'sem rótulos explícitos'),
        
        # Títulos de seções
        ('Setup and Imports', 'Configuração e Importações'),
        ('Classic STDP Curve', 'Curva STDP Clássica'),
        ('STDP Simulation with Brian2', 'Simulação STDP com Brian2'),
        ('Temporal Pattern Learning', 'Aprendizado de Padrões Temporais'),
        ('Comparison with Traditional Methods', 'Comparação com Métodos Tradicionais'),
        ('Application in Fraud Detection', 'Aplicação na Detecção de Fraude'),
        ('Synthetic Data Generation', 'Geração de Dados Sintéticos'),
        ('Spike Encoding', 'Codificação de Spikes'),
        ('SNN Architecture', 'Arquitetura da SNN'),
        ('Complete Pipeline', 'Pipeline Completo'),
        ('Individual Prediction Examples', 'Exemplos de Predição Individual'),
        ('Performance Analysis', 'Análise de Desempenho'),
        ('Conclusions', 'Conclusões'),
        ('Recommendations', 'Recomendações'),
        ('Results', 'Resultados'),
        ('Benchmark on CPU', 'Benchmark em CPU'),
        ('Intel Loihi 2 Simulation', 'Simulação Intel Loihi 2'),
        ('Comparison and Analysis', 'Comparação e Análise'),
        ('Visualizations', 'Visualizações'),
        ('Scalability Analysis', 'Análise de Escalabilidade'),
        ('Main Results', 'Principais Resultados'),
        ('Environment Configuration', 'Configuração do Ambiente'),
        ('CUDA Diagnostics', 'Diagnóstico CUDA'),
        ('Device Configuration', 'Configuração de Dispositivo'),
        
        # Frases descritivas
        ('Visualize how the change in weight depends on the temporal difference between spikes', 'Visualizar como a mudança no peso depende da diferença temporal entre spikes'),
        ('Simulate two neurons connected with STDP and observe evolution of weights', 'Simular dois neurônios conectados com STDP e observar evolução dos pesos'),
        ('Demonstrate how STDP learns temporal correlations in repeated patterns', 'Demonstrar como STDP aprende correlações temporais em padrões repetidos'),
        ("Let's create a synthetic dataset of banking transactions with realistic patterns", 'Vamos criar um conjunto de dados sintético de transações bancárias com padrões realistas'),
        ('Demonstrate how transaction features are converted into temporal spikes', 'Demonstrar como as características de transações são convertidas em spikes temporais'),
        ('Visualize and understand the architecture of the Spiking Neural Network', 'Visualizar e entender a arquitetura da Rede Neural Spiking'),
        ('Run the end-to-end pipeline: train and evaluate', 'Executar o pipeline de ponta a ponta: treinar e avaliar'),
        ('Test with specific transactions', 'Testar com transações específicas'),
        ('Evaluate latency and throughput of the system', 'Avaliar latência e throughput do sistema'),
        
        # Vantagens/Desvantagens
        ('Advantages of the Neuromorphic Approach', 'Vantagens da Abordagem Neuromórfica'),
        ('Ultra-low latency', 'Latência ultra-baixa'),
        ('Minimal energy consumption', 'Consumo mínimo de energia'),
        ('Real-time processing', 'Processamento em tempo real'),
        ('Scalable', 'Escalável'),
        
        # Objetivo
        ('Objective', 'Objetivo'),
        ('Compare the performance of the fraud detection implementation with SNN in', 'Comparar o desempenho da implementação de detecção de fraude com SNN em'),
        ('Traditional CPU', 'CPU Tradicional'),
        ('simulator', 'simulador'),
        
        # Métricas
        ('Metrics Evaluated', 'Métricas Avaliadas'),
        ('latency (ms per inference)', 'latência (ms por inferência)'),
        ('Throughput (transactions per second)', 'Throughput (transações por segundo)'),
        ('energy (millijoules)', 'energia (milijoules)'),
        ('power (milliwatts)', 'potência (miliwatts)'),
        ('efficiency (speedup and power efficiency)', 'eficiência (aceleração e eficiência de potência)'),
        
        # Palavras comuns que ficaram em inglês
        (' the ', ' o '),
        (' The ', ' O '),
        (' is ', ' é '),
        (' are ', ' são '),
        (' and ', ' e '),
        (' or ', ' ou '),
        (' but ', ' mas '),
        (' with ', ' com '),
        (' without ', ' sem '),
        (' for ', ' para '),
        (' from ', ' de '),
        (' in ', ' em '),
        (' on ', ' em '),
        (' at ', ' em '),
        (' to ', ' para '),
        (' of ', ' de '),
        (' by ', ' por '),
        (' as ', ' como '),
        (' if ', ' se '),
        (' when ', ' quando '),
        (' where ', ' onde '),
        (' how ', ' como '),
        (' what ', ' o que '),
        (' that ', ' que '),
        (' this ', ' este '),
        (' can ', ' pode '),
        (' will ', ' irá '),
        (' should ', ' deveria '),
        (' must ', ' deve '),
        (' used ', ' usado '),
        (' using ', ' usando '),
        (' between ', ' entre '),
        (' through ', ' através '),
        (' also ', ' também '),
        (' about ', ' sobre '),
        (' all ', ' todos '),
        (' first ', ' primeiro '),
        (' now ', ' agora '),
        ('Now ', 'Agora '),
        ('First,', 'Primeiro,'),
        ("Let's", 'Vamos'),
        
        # Ações
        ('Visualize', 'Visualizar'),
        ('Simulate', 'Simular'),
        ('Demonstrate', 'Demonstrar'),
        ('Compare', 'Comparar'),
        ('Analyze', 'Analisar'),
        ('Evaluate', 'Avaliar'),
        ('Configure', 'Configurar'),
        ('Install', 'Instalar'),
        ('Create', 'Criar'),
        ('Generate', 'Gerar'),
        ('Execute', 'Executar'),
        ('Run', 'Executar'),
        ('Test', 'Testar'),
        ('Train', 'Treinar'),
        ('Validate', 'Validar'),
    ]
    
    result = text
    for eng, pt in translations:
        result = result.replace(eng, pt)
    
    return result

def process_notebook(input_path):
    """Processa e traduz um notebook."""
    print(f"Processando: {input_path.name}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells_processed = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = cell.get('source', [])
            if isinstance(source, list):
                cell['source'] = [translate_comprehensive(line) for line in source]
            else:
                cell['source'] = translate_comprehensive(source)
            cells_processed += 1
        elif cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                cell['source'] = [translate_comprehensive(line) for line in source]
            else:
                cell['source'] = translate_comprehensive(source)
            cells_processed += 1
    
    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"  ✓ {cells_processed} células processadas")

def main():
    notebooks_dir = Path(__file__).parent / 'notebooks'
    
    notebooks = [
        '01-stdp_example-pt.ipynb',
        '02-stdp-demo-pt.ipynb',
        '03-loihi_benchmark-pt.ipynb',
        '04_brian2_vs_snntorch-pt.ipynb',
        '05_production_solutions-pt.ipynb',
        '06_phase1_integration-pt.ipynb',
    ]
    
    print("=" * 70)
    print("TRADUÇÃO AVANÇADA PARA PORTUGUÊS")
    print("=" * 70)
    print()
    
    for nb_name in notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            process_notebook(nb_path)
        else:
            print(f"❌ Não encontrado: {nb_name}")
    
    print()
    print("=" * 70)
    print("✅ Tradução avançada concluída!")
    print("=" * 70)

if __name__ == '__main__':
    main()
