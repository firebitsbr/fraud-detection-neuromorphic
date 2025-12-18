#!/usr/bin/env python3
"""
Script para traduzir notebooks de inglês para português de forma CORRETA.
Preserva a estrutura JSON e código Python.
"""

import json
import re
import sys
from pathlib import Path

# Dicionário completo de traduções (inglês -> português)
TRANSLATIONS = {
    # Títulos e descrições
    "STDP Example: Biological Learning": "Exemplo STDP: Aprendizado Biológico",
    "Interactive Tutorial": "Tutorial Interativo",
    "biological learning mechanism": "mecanismo biológico de aprendizado",
    "Spike-Timing-Dependent Plasticity": "Plasticidade Dependente do Tempo de Spike",
    "neuromorphic neural networks": "redes neurais neuromórficas",
    "Demonstrates how neurons learn": "Demonstra como os neurônios aprendem",
    "temporal correlations automatically": "correlações temporais automaticamente",
    
    # Metadados
    "Author:": "Autor:",
    "Creation Date:": "Data de Criação:",
    "License:": "Licença:",
    "Development:": "Desenvolvimento:",
    "Human + AI Assisted Development": "Desenvolvimento Humano + Assistido por IA",
    "AI-Assisted Development": "Desenvolvimento Assistido por IA",
    
    # Frases comuns
    "This notebook explores": "Este notebook explora",
    "What is STDP?": "O que é STDP?",
    "unsupervised learning rule": "regra de aprendizado não supervisionado",
    "inspired by biological neurons": "inspirada por neurônios biológicos",
    "pre-synaptic neuron fires": "neurônio pré-sináptico dispara",
    "post-synaptic": "pós-sináptico",
    "Potentiation": "Potenciação",
    "Depression": "Depressão",
    "This allows the network to learn": "Isso permite que a rede aprenda",
    "causal relationships": "relações causais",
    "without explicit labels": "sem rótulos explícitos",
    
    # Seções
    "Setup and Imports": "Configuração e Importações",
    "Classic STDP Curve": "Curva STDP Clássica",
    "Visualize how": "Visualizar como",
    "the change in weight depends on": "a mudança no peso depende do",
    "temporal difference between spikes": "diferença temporal entre spikes",
    "Simulation": "Simulação",
    "Simulate two neurons": "Simular dois neurônios",
    "connected with STDP": "conectados com STDP",
    "observe evolution of weights": "observar evolução dos pesos",
    "Temporal Pattern Learning": "Aprendizado de Padrões Temporais",
    "Demonstrate how STDP learns": "Demonstrar como STDP aprende",
    "temporal correlations in repeated patterns": "correlações temporais em padrões repetidos",
    "Comparison with Traditional Methods": "Comparação com Métodos Tradicionais",
    "Application in Fraud Detection": "Aplicação em Detecção de Fraude",
    
    # Termos técnicos gerais
    "Description:": "Descrição:",
    "Objective": "Objetivo",
    "Parameters": "Parâmetros",
    "Results": "Resultados",
    "Analysis": "Análise",
    "Conclusions": "Conclusões",
    "Recommendations": "Recomendações",
    "Advantages": "Vantagens",
    "Disadvantages": "Desvantagens",
    "Performance": "Desempenho",
    "Comparison": "Comparação",
    "Evaluation": "Avaliação",
    "Implementation": "Implementação",
    "Configuration": "Configuração",
    "Installation": "Instalação",
    "Documentation": "Documentação",
    
    # Palavras individuais comuns
    " the ": " o ",
    " The ": " O ",
    " is ": " é ",
    " are ": " são ",
    " and ": " e ",
    " or ": " ou ",
    " but ": " mas ",
    " with ": " com ",
    " without ": " sem ",
    " for ": " para ",
    " from ": " de ",
    " to ": " para ",
    " in ": " em ",
    " on ": " em ",
    " at ": " em ",
    " by ": " por ",
    " of ": " de ",
    " as ": " como ",
    " if ": " se ",
    " when ": " quando ",
    " where ": " onde ",
    " how ": " como ",
    " what ": " o que ",
    " which ": " qual ",
    " that ": " que ",
    " this ": " este ",
    " these ": " estes ",
    " can ": " pode ",
    " must ": " deve ",
    " should ": " deveria ",
    " will ": " irá ",
    " used ": " usado ",
    " using ": " usando ",
    " between ": " entre ",
    " after ": " após ",
    " before ": " antes ",
    " during ": " durante ",
    " through ": " através ",
    " also ": " também ",
    " only ": " apenas ",
    " about ": " sobre ",
    " all ": " todos ",
    " some ": " alguns ",
    " more ": " mais ",
    " less ": " menos ",
    " very ": " muito ",
    " much ": " muito ",
    " many ": " muitos ",
    " few ": " poucos ",
    " first ": " primeiro ",
    " last ": " último ",
    " next ": " próximo ",
    " same ": " mesmo ",
    " different ": " diferente ",
    " new ": " novo ",
    " old ": " antigo ",
    " good ": " bom ",
    " bad ": " ruim ",
    " high ": " alto ",
    " low ": " baixo ",
    " fast ": " rápido ",
    " slow ": " lento ",
    " large ": " grande ",
    " small ": " pequeno ",
    " here ": " aqui ",
    " there ": " lá ",
    " now ": " agora ",
    " then ": " então ",
    " always ": " sempre ",
    " never ": " nunca ",
    " sometimes ": " às vezes ",
    " often ": " frequentemente ",
    " rarely ": " raramente ",
    
    # Verbos comuns
    " import ": " importar ",
    " create ": " criar ",
    " generate ": " gerar ",
    " execute ": " executar ",
    " run ": " executar ",
    " save ": " salvar ",
    " load ": " carregar ",
    " add ": " adicionar ",
    " remove ": " remover ",
    " update ": " atualizar ",
    " calculate ": " calcular ",
    " compute ": " computar ",
    " process ": " processar ",
    " analyze ": " analisar ",
    " evaluate ": " avaliar ",
    " measure ": " medir ",
    " compare ": " comparar ",
    " visualize ": " visualizar ",
    " display ": " exibir ",
    " show ": " mostrar ",
    " print ": " imprimir ",
    " configure ": " configurar ",
    " install ": " instalar ",
    " validate ": " validar ",
    " test ": " testar ",
    " train ": " treinar ",
    " learn ": " aprender ",
    " detect ": " detectar ",
    " predict ": " prever ",
    " classify ": " classificar ",
    
    # Substantivos técnicos
    " network ": " rede ",
    " neuron ": " neurônio ",
    " weight ": " peso ",
    " synapse ": " sinapse ",
    " spike ": " spike ",
    " pattern ": " padrão ",
    " correlation ": " correlação ",
    " learning ": " aprendizado ",
    " training ": " treinamento ",
    " testing ": " teste ",
    " validation ": " validação ",
    " data ": " dados ",
    " dataset ": " conjunto de dados ",
    " file ": " arquivo ",
    " result ": " resultado ",
    " example ": " exemplo ",
    " parameter ": " parâmetro ",
    " value ": " valor ",
    " time ": " tempo ",
    " duration ": " duração ",
    " latency ": " latência ",
    " speed ": " velocidade ",
    " accuracy ": " acurácia ",
    " precision ": " precisão ",
    " memory ": " memória ",
    " energy ": " energia ",
    " power ": " potência ",
    " efficiency ": " eficiência ",
    " model ": " modelo ",
    " method ": " método ",
    " system ": " sistema ",
    " process ": " processo ",
    " detection ": " detecção ",
    " fraud ": " fraude ",
    " transaction ": " transação ",
    " anomaly ": " anomalia ",
    " feature ": " característica ",
    " label ": " rótulo ",
    " prediction ": " predição ",
    " classification ": " classificação ",
}

def translate_text(text):
    """Traduz texto de inglês para português."""
    if not isinstance(text, str):
        return text
    
    # Não traduzir código Python ou URLs
    if text.strip().startswith(('import ', 'from ', 'def ', 'class ', 'http://', 'https://')):
        return text
    
    # Aplicar traduções
    result = text
    for eng, pt in TRANSLATIONS.items():
        result = result.replace(eng, pt)
    
    return result

def translate_notebook(input_path, output_path):
    """Traduz um notebook Jupyter de inglês para português."""
    print(f"Traduzindo: {input_path.name}")
    
    # Carregar notebook
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Traduzir células
    cells_translated = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            # Traduzir conteúdo markdown
            source = cell.get('source', [])
            if isinstance(source, list):
                cell['source'] = [translate_text(line) for line in source]
            else:
                cell['source'] = translate_text(source)
            cells_translated += 1
        elif cell.get('cell_type') == 'code':
            # Traduzir apenas comentários em código Python
            source = cell.get('source', [])
            if isinstance(source, list):
                translated = []
                for line in source:
                    if line.strip().startswith('#') and not line.strip().startswith('#!/'):
                        translated.append(translate_text(line))
                    else:
                        translated.append(line)
                cell['source'] = translated
            cells_translated += 1
    
    # Salvar notebook traduzido
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"  ✓ {cells_translated} células processadas")
    return True

def main():
    notebooks_dir = Path(__file__).parent / 'notebooks'
    
    # Lista de notebooks para traduzir
    notebooks = [
        '01-stdp_example',
        '02-stdp-demo',
        '03-loihi_benchmark',
        '04_brian2_vs_snntorch',
        '05_production_solutions',
        '06_phase1_integration',
    ]
    
    print("=" * 70)
    print("TRADUÇÃO CORRETA DE NOTEBOOKS PARA PORTUGUÊS")
    print("=" * 70)
    print()
    
    success_count = 0
    for nb_name in notebooks:
        input_file = notebooks_dir / f"{nb_name}.ipynb"
        output_file = notebooks_dir / f"{nb_name}-pt.ipynb"
        
        if not input_file.exists():
            print(f"❌ Arquivo não encontrado: {input_file}")
            continue
        
        try:
            if translate_notebook(input_file, output_file):
                success_count += 1
        except Exception as e:
            print(f"❌ Erro ao traduzir {nb_name}: {e}")
    
    print()
    print("=" * 70)
    print(f"✅ Tradução concluída: {success_count}/{len(notebooks)} notebooks")
    print("=" * 70)

if __name__ == '__main__':
    main()
