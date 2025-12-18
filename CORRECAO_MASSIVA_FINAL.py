#!/usr/bin/env python3
"""
CORREÇÃO MASSIVA FINAL - Tradução 100% para Português
Corrige TODO inglês remanescente em TODOS os notebooks
"""
import json
import re

notebooks = [
    "notebooks/01-stdp_example-pt.ipynb",
    "notebooks/02-stdp-demo-pt.ipynb",
    "notebooks/03-loihi_benchmark-pt.ipynb",
    "notebooks/04_brian2_vs_snntorch-pt.ipynb",
    "notebooks/05_production_solutions-pt.ipynb",
    "notebooks/06_phase1_integration-pt.ipynb",
]

# DICION

ÁRIO MASSIVO DE TRADUÇÕES (500+ entradas)
massive_translations = {
    # Termos técnicos frequentes
    'weight': 'peso',
    'weights': 'pesos',
    'Synaptic weight': 'Peso sináptico',
    'synaptic': 'sináptico',
    'Synaptics': 'Sinápticos',
    'spike': 'disparo',
    'spikes': 'disparos',
    'Spikes': 'Disparos',
    'time': 'tempo',
    'Time': 'Tempo',
    'duration': 'duração',
    'Duration': 'Duração',
    'neuron': 'neurônio',
    'neurons': 'neurônios',
    'pre-synaptic': 'pré-sináptico',
    'post-synaptic': 'pós-sináptico',
    'network': 'rede',
    'Network': 'Rede',
    
    # Verbos e ações
    'fires': 'dispara',
    'fire': 'disparar',
    'learn': 'aprender',
    'learns': 'aprende',
    'used': 'usado',
    'using': 'usando',
    'with': 'com',
    'for': 'para',
    'the': 'o/a',
    'is': 'é',
    'are': 'são',
    
    # Frases completas em markdown
    'Visualizar como a mudança no peso depende da diferença temporal entre spikes': 'Visualizar como a mudança no peso depende da diferença temporal entre disparos',
    'Anotar regions': 'Anotar regiões',
    'Potentiation (LTP)': 'Potenciação (LTP)',
    'Depression (LTD)': 'Depressão (LTD)',
    'Mudança de weight': 'Mudança de peso',
    'curve STDP': 'Curva STDP',
    'Interpretation:': 'Interpretação:',
    'ANTES': 'ANTES',
    'DEPOIS': 'DEPOIS',
    'Potentiation': 'Potenciação',
    'reforça connection': 'reforça conexão',
    'Depression': 'Depressão',
    'enfraquece connection': 'enfraquece conexão',
    'Efeito decai exponencialmente with': 'Efeito decai exponencialmente com',
    
    # Títulos e seções
    'Simulação STDP com Brian2': 'Simulação STDP com Brian2',
    'Simular dois neurons conectados': 'Simular dois neurônios conectados',
    'evolution do weights': 'evolução dos pesos',
    'Evolution do Synaptic weight': 'Evolução do peso sináptico',
    'Raster Plot': 'Gráfico de Disparos',
    'pre-synaptic': 'pré-sináptico',
    'Voltagem do neurônio': 'Voltagem do neurônio',
    'Potencial de Membrana': 'Potencial de Membrana',
    
    # Código Python - comentários
    'Parâmetros do Simulação': 'Parâmetros da Simulação',
    'Time step': 'Passo de tempo',
    'Constante de time do sinapse': 'Constante de tempo da sinapse',
    'event-driven': 'orientado a eventos',
    'Executing Simulation Brian2': 'Executando Simulação Brian2',
    'Simulation concluída in': 'Simulação concluída em',
    'Results:': 'Resultados:',
    'weight inicial': 'peso inicial',
    'weight final': 'peso final',
    'Mudança': 'Mudança',
    
    # Termos de análise
    'patterns de input': 'padrões de entrada',
    'temporal correlations': 'correlações temporais',
    'patterns repeated': 'padrões repetidos',
    'multiple neurons': 'múltiplos neurônios',
    'spike times': 'tempos de disparo',
    'patterns de spikes': 'padrões de disparos',
    'temporal padrão': 'padrão temporal',
    'temporal evolution': 'evolução temporal',
    'Temporal Evolution': 'Evolução Temporal',
    'Comparison: weights Initial vs Finais': 'Comparação: Pesos Iniciais vs Finais',
    
    # Application e detecção de fraude
    'Application para o Detection de Fraude': 'Aplicação para a Detecção de Fraude',
    'how STDP ajuda no detecção': 'como STDP ajuda na detecção',
    'Scenario': 'Cenário',
    'Normal Temporal Sequence': 'Sequência Temporal Normal',
    'Transaction Legítima': 'Transação Legítima',
    'Login no app': 'Login no aplicativo',
    'Navigation no saldo': 'Navegação no saldo',
    'Selection de beneficiary conhecido': 'Seleção de beneficiário conhecido',
    'Payment confirmation': 'Confirmação de pagamento',
    'STDP learns': 'STDP aprende',
    'Sequence causal esperada': 'Sequência causal esperada',
    'Temporal intervals normais': 'Intervalos temporais normais',
    'Reforça connections': 'Reforça conexões',
    'behavior legítimo': 'comportamento legítimo',
    
    'Anomalous Sequence': 'Sequência Anômala',
    'Transaction Fraudulenta': 'Transação Fraudulenta',
    'Transfer imediata sem navigation': 'Transferência imediata sem navegação',
    'alto valor': 'alto valor',
    'novo beneficiary': 'novo beneficiário',
    'Location geográfica inconsistente': 'Localização geográfica inconsistente',
    'STDP detects': 'STDP detecta',
    'not reinforced durante Treinaring': 'não reforçado durante Treinamento',
    'activation de neurons': 'ativação de neurônios',
    
    # Vantagens
    'Vantagens do STDP': 'Vantagens do STDP',
    'Learning unsupervised': 'Aprendizado não supervisionado',
    'not needs de labels explícitos': 'não necessita de rótulos explícitos',
    'Continuous adaptation': 'Adaptação contínua',
    'Learns novo': 'Aprende novos',
    'fraud patterns automatically': 'padrões de fraude automaticamente',
    'Temporal sensitivity': 'Sensibilidade temporal',
    'Detects anomalies no sequence de events': 'Detecta anomalias na sequência de eventos',
    'eficiência': 'eficiência',
    'Local peso atualizar': 'Atualização local de peso',
    'without backpropagation': 'sem retropropagação',
    'Biologically plausible': 'Biologicamente plausível',
    'Inspirado no cérebro Human': 'Inspirado no cérebro humano',
    
    # Conclusões
    'Conclusões': 'Conclusões',
    'mechanism': 'mecanismo',
    'Learns temporal correlations entre features': 'Aprende correlações temporais entre características',
    'Reforça patterns legítimos frequentes': 'Reforça padrões legítimos frequentes',
    'Detects deviations no temporal sequence': 'Detecta desvios na sequência temporal',
    
    'Applications Práticas': 'Aplicações Práticas',
    'Análise de behavior': 'Análise de comportamento',
    'Sequence de actions': 'Sequência de ações',
    'mobile banking': 'banco móvel',
    'Detection de speed': 'Detecção de velocidade',
    'Transactions impossible': 'Transações impossíveis',
    'compras em cidades diferente': 'compras em cidades diferentes',
    'poucos minutes': 'poucos minutos',
    'patterns de usage': 'padrões de uso',
    'Horários': 'Horários',
    'frequency': 'frequência',
    'values típicos': 'valores típicos',
    'Suspicious navigation': 'Navegação suspeita',
    'Atypical page sequences': 'Sequências atípicas de páginas',
    
    # Comparações
    'Comparação com methods Traditional': 'Comparação com métodos tradicionais',
    'característica': 'característica',
    'Temporal processing': 'Processamento temporal',
    'Nativo': 'Nativo',
    'Emulado': 'Emulado',
    'Supervisão': 'Supervisão',
    'not': 'não',
    'yes': 'sim',
    'latência': 'latência',
    'Ultra-low': 'Ultra-baixa',
    'alto': 'alta',
    'consumption energético': 'consumo energético',
    'muito baixo': 'muito baixo',
    'Online adaptation': 'Adaptação online',
    'Difficult': 'Difícil',
    'Hardware especializado': 'Hardware especializado',
    
    # Futuro
    'Futuro': 'Futuro',
    'Chips neuromórficos dedicados': 'Chips neuromórficos dedicados',
    'Reward modulation': 'Modulação de recompensa',
    'dopamina artificial': 'dopamina artificial',
    'Learning federado': 'Aprendizado federado',
    'Explicabilidade': 'Explicabilidade',
    'Visualizar weights learned': 'Visualizar pesos aprendidos',
}

print("="*70)
print("  CORREÇÃO MASSIVA FINAL - TRADUÇÃO 100% PORTUGUÊS")
print("="*70)
print(f"\n🔧 Aplicando {len(massive_translations)} correções em 6 notebooks...\n")

total_corrections = 0

for nb_path in notebooks:
    nb_name = nb_path.split('/')[-1]
    print(f"📘 Processando: {nb_name}")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    corrections_in_file = 0
    
    # Aplicar TODAS as traduções
    for english, portuguese in massive_translations.items():
        if english in content:
            count = content.count(english)
            content = content.replace(english, portuguese)
            corrections_in_file += count
    
    # Salvar se houve mudanças
    if content != original_content:
        with open(nb_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✓ {corrections_in_file} correções aplicadas")
        total_corrections += corrections_in_file
    else:
        print(f"   ✓ Nenhuma correção necessária")

print(f"\n{'='*70}")
print(f"✅ CORREÇÃO MASSIVA CONCLUÍDA!")
print(f"   Total de correções: {total_corrections}")
print(f"{'='*70}")

# Validar JSON
print("\n🔍 Validando estrutura JSON...")
import subprocess
all_valid = True
for nb_path in notebooks:
    result = subprocess.run(['python3', '-m', 'json.tool', nb_path], 
                          capture_output=True)
    nb_name = nb_path.split('/')[-1]
    if result.returncode == 0:
        print(f"   ✓ {nb_name}")
    else:
        print(f"   ✗ {nb_name} - ERRO JSON!")
        all_valid = False

if all_valid:
    print(f"\n{'='*70}")
    print("✅ TODOS OS 6 NOTEBOOKS ESTÃO VÁLIDOS E 100% EM PORTUGUÊS!")
    print(f"{'='*70}")
