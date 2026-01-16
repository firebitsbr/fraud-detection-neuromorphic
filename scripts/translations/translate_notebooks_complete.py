#!/usr/bin/env python3
"""
Tradução COMPLETA de todos os notebooks Jupyter de português para inglês
"""
import json
import re
from pathlib import Path

# Dicionário MASSIVO de traduções português → inglês
TRANSLATIONS = {
    # Common words and phrases
    'Exemplo': 'Example',
    'Demonstração': 'Demonstration',
    'Tutorial': 'Tutorial',
    'Interativo': 'Interactive',
    'sobre': 'about',
    'mecanismo': 'mechanism',
    'aprendizado': 'learning',
    'biológico': 'biological',
    'usado': 'used',
    'usada': 'used',
    'redes': 'networks',
    'neurais': 'neural',
    'neuromórficas': 'neuromorphic',
    'Demonstra': 'Demonstrates',
    'demonstra': 'demonstrates',
    'como': 'how',
    'neurônios': 'neurons',
    'neurônio': 'neuron',
    'aprendem': 'learn',
    'correlações': 'correlations',
    'temporais': 'temporal',
    'automaticamente': 'automatically',
    'Descrição': 'Description',
    'Autor': 'Author',
    'Data de Criação': 'Creation Date',
    'Licença': 'License',
    'Desenvolvimento': 'Development',
    'Humano': 'Human',
    'Assistida': 'Assisted',
    'Importações concluídas': 'Imports completed',
    'Instala a biblioteca': 'Install the library',
    'ainda não estiver instalada': 'not yet installed',
    'Importação específica': 'Specific import',
    'ao invés de': 'instead of',
    'Configurar para usar': 'Configure to use',
    'evita erro': 'avoids error',
    'se headers faltarem': 'if headers are missing',
    'Este notebook explora': 'This notebook explores',
    'regra de aprendizado': 'learning rule',
    'não-supervisionado': 'unsupervised',
    'inspirada em': 'inspired by',
    'dispara ANTES': 'fires BEFORE',
    'pré-sináptico': 'pre-synaptic',
    'pós-sináptico': 'post-synaptic',
    'Potenciação': 'Potentiation',
    'potenciação': 'potentiation',
    'peso': 'weight',
    'pesos': 'weights',
    'dispara DEPOIS': 'fires AFTER',
    'Depressão': 'Depression',
    'depressão': 'depression',
    'Isso permite que': 'This allows',
    'rede': 'network',
    'aprenda': 'learns',
    'relações causais': 'causal relationships',
    'sem labels explícitos': 'without explicit labels',
    'Visualizar': 'Visualize',
    'mudança de peso': 'weight change',
    'depende da': 'depends on',
    'diferença temporal': 'temporal difference',
    'entre spikes': 'between spikes',
    'Parâmetros': 'Parameters',
    'constante de tempo': 'time constant',
    'pré-sináptica': 'pre-synaptic',
    'pós-sináptica': 'post-synaptic',
    'Amplitude': 'Amplitude',
    'Calcular': 'Calculate',
    'Calcula': 'Calculates',
    'segundo': 'according to',
    'segundo a': 'according to',
    'Plotar': 'Plot',
    'curva': 'curve',
    'O que é': 'What is',
    'Configuração': 'Configuration',
    'Preparação': 'Preparation',
    'Carregamento': 'Loading',
    'Processamento': 'Processing',
    'Treinamento': 'Training',
    'Avaliação': 'Evaluation',
    'Otimização': 'Optimization',
    'Implementação': 'Implementation',
    'Execução': 'Execution',
    'Simulação': 'Simulation',
    'Análise': 'Analysis',
    'Resultados': 'Results',
    'Conclusão': 'Conclusion',
    'Próximos passos': 'Next steps',
    'Referências': 'References',
    'Experimento': 'Experiment',
    'Teste': 'Test',
    'Validação': 'Validation',
    'Visualização': 'Visualization',
    'Comparação': 'Comparison',
    'Integração': 'Integration',
    'Soluções': 'Solutions',
    'Produção': 'Production',
    'Benchmark': 'Benchmark',
    'de December': 'December',
    'of December': 'December',
    
    # Common verbs
    'pode': 'can',
    'podem': 'can',
    'deve': 'should',
    'devem': 'should',
    'precisa': 'needs',
    'precisam': 'need',
    'permite': 'allows',
    'permitem': 'allow',
    'mostra': 'shows',
    'mostram': 'show',
    'indica': 'indicates',
    'indicam': 'indicate',
    'representa': 'represents',
    'representam': 'represent',
    'utiliza': 'uses',
    'utilizam': 'use',
    'gera': 'generates',
    'geram': 'generate',
    'cria': 'creates',
    'criam': 'create',
    'define': 'defines',
    'definem': 'define',
    'implementa': 'implements',
    'implementam': 'implement',
    'executa': 'executes',
    'executam': 'execute',
    'realiza': 'performs',
    'realizam': 'perform',
    'aplica': 'applies',
    'aplicam': 'apply',
    'calcula': 'calculates',
    'calculam': 'calculate',
    'simula': 'simulates',
    'simulam': 'simulate',
    'treina': 'trains',
    'treinam': 'train',
    'testa': 'tests',
    'testam': 'test',
    'valida': 'validates',
    'validam': 'validate',
    'verifica': 'verifies',
    'verificam': 'verify',
    'compara': 'compares',
    'comparam': 'compare',
    'visualiza': 'visualizes',
    'visualizam': 'visualize',
    'exibe': 'displays',
    'exibem': 'display',
    'apresenta': 'presents',
    'apresentam': 'present',
    
    # Common adjectives
    'necessário': 'necessary',
    'necessária': 'necessary',
    'necessários': 'necessary',
    'necessárias': 'necessary',
    'importante': 'important',
    'importantes': 'important',
    'principal': 'main',
    'principais': 'main',
    'básico': 'basic',
    'básica': 'basic',
    'básicos': 'basic',
    'básicas': 'basic',
    'simples': 'simple',
    'complexo': 'complex',
    'complexa': 'complex',
    'complexos': 'complex',
    'complexas': 'complex',
    'rápido': 'fast',
    'rápida': 'fast',
    'rápidos': 'fast',
    'rápidas': 'fast',
    'lento': 'slow',
    'lenta': 'slow',
    'lentos': 'slow',
    'lentas': 'slow',
    'alto': 'high',
    'alta': 'high',
    'altos': 'high',
    'altas': 'high',
    'baixo': 'low',
    'baixa': 'low',
    'baixos': 'low',
    'baixas': 'low',
    'grande': 'large',
    'grandes': 'large',
    'pequeno': 'small',
    'pequena': 'small',
    'pequenos': 'small',
    'pequenas': 'small',
    'novo': 'new',
    'nova': 'new',
    'novos': 'new',
    'novas': 'new',
    'antigo': 'old',
    'antiga': 'old',
    'antigos': 'old',
    'antigas': 'old',
    'melhor': 'better',
    'melhores': 'better',
    'pior': 'worse',
    'piores': 'worse',
    'maior': 'larger',
    'maiores': 'larger',
    'menor': 'smaller',
    'menores': 'smaller',
    
    # Common nouns
    'dados': 'data',
    'modelo': 'model',
    'modelos': 'models',
    'exemplo': 'example',
    'exemplos': 'examples',
    'resultado': 'result',
    'resultados': 'results',
    'valor': 'value',
    'valores': 'values',
    'tempo': 'time',
    'tempos': 'times',
    'taxa': 'rate',
    'taxas': 'rates',
    'número': 'number',
    'números': 'numbers',
    'tipo': 'type',
    'tipos': 'types',
    'forma': 'form',
    'formas': 'forms',
    'método': 'method',
    'métodos': 'methods',
    'função': 'function',
    'funções': 'functions',
    'classe': 'class',
    'classes': 'classes',
    'objeto': 'object',
    'objetos': 'objects',
    'variável': 'variable',
    'variáveis': 'variables',
    'parâmetro': 'parameter',
    'parâmetros': 'parameters',
    'entrada': 'input',
    'entradas': 'inputs',
    'saída': 'output',
    'saídas': 'outputs',
    'camada': 'layer',
    'camadas': 'layers',
    'época': 'epoch',
    'épocas': 'epochs',
    'iteração': 'iteration',
    'iterações': 'iterations',
    'erro': 'error',
    'erros': 'errors',
    'perda': 'loss',
    'perdas': 'losses',
    'acurácia': 'accuracy',
    'precisão': 'precision',
    'revocação': 'recall',
    'métrica': 'metric',
    'métricas': 'metrics',
    'desempenho': 'performance',
    'eficiência': 'efficiency',
    'velocidade': 'speed',
    'latência': 'latency',
    'consumo': 'consumption',
    'energia': 'energy',
    'potência': 'power',
    'memória': 'memory',
    'armazenamento': 'storage',
    'capacidade': 'capacity',
    'tamanho': 'size',
    'dimensão': 'dimension',
    'dimensões': 'dimensions',
    'quantidade': 'quantity',
    'porcentagem': 'percentage',
    'proporção': 'proportion',
    'fração': 'fraction',
    'parte': 'part',
    'partes': 'parts',
    'total': 'total',
    'soma': 'sum',
    'média': 'average',
    'mediana': 'median',
    'mínimo': 'minimum',
    'máximo': 'maximum',
    'variância': 'variance',
    'desvio': 'deviation',
    'distribuição': 'distribution',
    'probabilidade': 'probability',
    'frequência': 'frequency',
    'padrão': 'pattern',
    'padrões': 'patterns',
    'característica': 'feature',
    'características': 'features',
    'atributo': 'attribute',
    'atributos': 'attributes',
    'propriedade': 'property',
    'propriedades': 'properties',
}

def translate_text(text):
    """Translate text using the dictionary"""
    for pt, en in TRANSLATIONS.items():
        # Use word boundaries to avoid partial matches
        text = re.sub(r'\b' + re.escape(pt) + r'\b', en, text, flags=re.IGNORECASE)
    return text

def translate_notebook(notebook_path):
    """Translate a Jupyter notebook"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Translate all cell sources
        for cell in nb.get('cells', []):
            if 'source' in cell and cell['source']:
                # Translate each line
                translated_source = []
                for line in cell['source']:
                    translated_line = translate_text(line)
                    translated_source.append(translated_line)
                cell['source'] = translated_source
        
        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        
        return True
    except Exception as e:
        print(f"Error in {notebook_path}: {e}")
        return False

def main():
    print("=" * 80)
    print(" TRADUZINDO TODOS OS NOTEBOOKS")
    print("=" * 80)
    print()
    
    notebooks_dir = Path('notebooks')
    notebooks = list(notebooks_dir.glob('*.ipynb'))
    
    translated = 0
    for nb_path in notebooks:
        if translate_notebook(nb_path):
            print(f"✓ {nb_path.name}")
            translated += 1
        else:
            print(f"✗ {nb_path.name}")
    
    print()
    print("=" * 80)
    print(f" COMPLETO! Traduzidos {translated}/{len(notebooks)} notebooks")
    print("=" * 80)

if __name__ == '__main__':
    main()
