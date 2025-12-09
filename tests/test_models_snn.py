"""
**Descrição:** Suite de testes para validar a implementação dos modelos de Spiking Neural Networks, incluindo inicialização, arquitetura, pesos e comportamento de neurônios LIF.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import pytest
import numpy as np
from models_snn import FraudSNN, demonstrate_lif_neuron


class TestFraudSNN:
    """Testes para a classe FraudSNN"""
    
    def test_initialization(self):
        """Testa inicialização da rede"""
        snn = FraudSNN(input_size=256, hidden_sizes=[128, 64], output_size=2)
        
        assert snn.input_size == 256
        assert snn.hidden_sizes == [128, 64]
        assert snn.output_size == 2
        assert snn.layers is not None
    
    def test_network_stats(self):
        """Testa estatísticas da rede"""
        snn = FraudSNN(input_size=256, hidden_sizes=[128, 64], output_size=2)
        stats = snn.get_network_stats()
        
        assert 'layers' in stats
        assert 'total_neurons' in stats
        assert 'total_synapses' in stats
        assert 'weights' in stats
        
        # Verificar valores
        assert stats['layers']['input'] == 256
        assert stats['layers']['hidden'] == [128, 64]
        assert stats['layers']['output'] == 2
        assert stats['total_neurons'] == 450  # 256 + 128 + 64 + 2
        assert stats['total_synapses'] == 41088  # 256*128 + 128*64 + 64*2
    
    def test_different_architectures(self):
        """Testa diferentes arquiteturas"""
        # Arquitetura pequena
        snn1 = FraudSNN(input_size=10, hidden_sizes=[5], output_size=2)
        stats1 = snn1.get_network_stats()
        assert stats1['total_neurons'] == 17  # 10 + 5 + 2
        
        # Arquitetura grande
        snn2 = FraudSNN(input_size=512, hidden_sizes=[256, 128, 64], output_size=2)
        stats2 = snn2.get_network_stats()
        assert stats2['total_neurons'] == 962  # 512 + 256 + 128 + 64 + 2
    
    def test_weight_initialization(self):
        """Testa inicialização dos pesos"""
        snn = FraudSNN(input_size=256, hidden_sizes=[128, 64], output_size=2)
        stats = snn.get_network_stats()
        
        weights = stats['weights']
        assert 0 <= weights['min'] <= weights['max']
        assert weights['mean'] >= 0
        assert weights['std'] >= 0
    
    def test_invalid_architecture(self):
        """Testa arquiteturas inválidas"""
        with pytest.raises((ValueError, TypeError, AssertionError)):
            FraudSNN(input_size=0, hidden_sizes=[128], output_size=2)
        
        with pytest.raises((ValueError, TypeError, AssertionError)):
            FraudSNN(input_size=256, hidden_sizes=[], output_size=2)


class TestLIFNeuron:
    """Testes para demonstração de neurônio LIF"""
    
    def test_demonstrate_lif_neuron(self):
        """Testa função de demonstração do LIF"""
        data = demonstrate_lif_neuron()
        
        # Verificar estrutura dos dados
        assert 'time' in data
        assert 'voltage' in data
        assert 'input' in data
        assert 'spikes' in data
        
        # Verificar tipos
        assert isinstance(data['time'], np.ndarray)
        assert isinstance(data['voltage'], np.ndarray)
        assert isinstance(data['input'], np.ndarray)
        assert isinstance(data['spikes'], list)
        
        # Verificar dimensões
        assert len(data['time']) > 0
        assert len(data['voltage']) == len(data['time'])
        assert len(data['input']) == len(data['time'])
    
    def test_lif_spikes(self):
        """Testa se o neurônio LIF gera spikes"""
        data = demonstrate_lif_neuron()
        
        # Deve haver pelo menos 1 spike
        assert len(data['spikes']) > 0
        
        # Spikes devem estar dentro do range de tempo
        for spike_time in data['spikes']:
            assert data['time'][0] <= spike_time <= data['time'][-1]
    
    def test_lif_voltage_range(self):
        """Testa range de voltagem do LIF"""
        data = demonstrate_lif_neuron()
        
        # Voltagem deve estar em range plausível (mV)
        assert np.all(data['voltage'] >= -100)  # Não muito abaixo de resting
        assert np.all(data['voltage'] <= 0)     # Não positivo


class TestEdgeCases:
    """Testes de casos extremos"""
    
    def test_minimal_network(self):
        """Testa rede mínima possível"""
        snn = FraudSNN(input_size=1, hidden_sizes=[1], output_size=1)
        stats = snn.get_network_stats()
        assert stats['total_neurons'] == 3
        assert stats['total_synapses'] == 2
    
    def test_large_network(self):
        """Testa rede grande"""
        snn = FraudSNN(input_size=1024, hidden_sizes=[512, 256], output_size=10)
        stats = snn.get_network_stats()
        assert stats['total_neurons'] == 1802
        assert stats['total_synapses'] > 500000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
