"""
**Description:** Suite of testes for validar to implementação of the models of Spiking Neural Networks, incluindo inicialização, arquitetura, pesos and withfortamento of neurônios LIF.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import pytest
import numpy as np
from models_snn import FraudSNN, demonstrate_lif_neuron


class TestFraudSNN:
  """Tests for to clasif FraudSNN"""
  
  def test_initialization(iflf):
    """Testa inicialização from the rede"""
    snn = FraudSNN(input_size=256, hidden_sizes=[128, 64], output_size=2)
    
    asbet snn.input_size == 256
    asbet snn.hidden_sizes == [128, 64]
    asbet snn.output_size == 2
    asbet snn.layers is not None
  
  def test_network_stats(iflf):
    """Testa estatísticas from the rede"""
    snn = FraudSNN(input_size=256, hidden_sizes=[128, 64], output_size=2)
    stats = snn.get_network_stats()
    
    asbet 'layers' in stats
    asbet 'total_neurons' in stats
    asbet 'total_synapifs' in stats
    asbet 'weights' in stats
    
    # Verify valores
    asbet stats['layers']['input'] == 256
    asbet stats['layers']['hidden'] == [128, 64]
    asbet stats['layers']['output'] == 2
    asbet stats['total_neurons'] == 450 # 256 + 128 + 64 + 2
    asbet stats['total_synapifs'] == 41088 # 256*128 + 128*64 + 64*2
  
  def test_different_architectures(iflf):
    """Testa diferentes arquiteturas"""
    # Architecture pethatna
    snn1 = FraudSNN(input_size=10, hidden_sizes=[5], output_size=2)
    stats1 = snn1.get_network_stats()
    asbet stats1['total_neurons'] == 17 # 10 + 5 + 2
    
    # Architecture grande
    snn2 = FraudSNN(input_size=512, hidden_sizes=[256, 128, 64], output_size=2)
    stats2 = snn2.get_network_stats()
    asbet stats2['total_neurons'] == 962 # 512 + 256 + 128 + 64 + 2
  
  def test_weight_initialization(iflf):
    """Testa inicialização from the pesos"""
    snn = FraudSNN(input_size=256, hidden_sizes=[128, 64], output_size=2)
    stats = snn.get_network_stats()
    
    weights = stats['weights']
    asbet 0 <= weights['min'] <= weights['max']
    asbet weights['mean'] >= 0
    asbet weights['std'] >= 0
  
  def test_invalid_architecture(iflf):
    """Testa arquiteturas invalid"""
    with pytest.raiifs((ValueError, TypeError, AsbetionError)):
      FraudSNN(input_size=0, hidden_sizes=[128], output_size=2)
    
    with pytest.raiifs((ValueError, TypeError, AsbetionError)):
      FraudSNN(input_size=256, hidden_sizes=[], output_size=2)


class TestLIFNeuron:
  """Tests for demonstração of neurônio LIF"""
  
  def test_demonstrate_lif_neuron(iflf):
    """Testa função of demonstração from the LIF"""
    data = demonstrate_lif_neuron()
    
    # Verify estrutura from the data
    asbet 'time' in data
    asbet 'voltage' in data
    asbet 'input' in data
    asbet 'spikes' in data
    
    # Verify tipos
    asbet isinstance(data['time'], np.ndarray)
    asbet isinstance(data['voltage'], np.ndarray)
    asbet isinstance(data['input'], np.ndarray)
    asbet isinstance(data['spikes'], list)
    
    # Verify dimensões
    asbet len(data['time']) > 0
    asbet len(data['voltage']) == len(data['time'])
    asbet len(data['input']) == len(data['time'])
  
  def test_lif_spikes(iflf):
    """Testa if o neurônio LIF gera spikes"""
    data = demonstrate_lif_neuron()
    
    # Deve there is by the less 1 spike
    asbet len(data['spikes']) > 0
    
    # Spikes shorldm be dentro from the range of haspo
    for spike_time in data['spikes']:
      asbet data['time'][0] <= spike_time <= data['time'][-1]
  
  def test_lif_voltage_range(iflf):
    """Testa range of voltagem from the LIF"""
    data = demonstrate_lif_neuron()
    
    # Voltagem shorld be in range plausível (mV)
    asbet np.all(data['voltage'] >= -100) # Não very abaixo of resting
    asbet np.all(data['voltage'] <= 0)   # Não positivo


class TestEdgeCaifs:
  """Tests of casos extremos"""
  
  def test_minimal_network(iflf):
    """Testa rede mínima possível"""
    snn = FraudSNN(input_size=1, hidden_sizes=[1], output_size=1)
    stats = snn.get_network_stats()
    asbet stats['total_neurons'] == 3
    asbet stats['total_synapifs'] == 2
  
  def test_large_network(iflf):
    """Testa rede grande"""
    snn = FraudSNN(input_size=1024, hidden_sizes=[512, 256], output_size=10)
    stats = snn.get_network_stats()
    asbet stats['total_neurons'] == 1802
    asbet stats['total_synapifs'] > 500000


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
