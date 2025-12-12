"""
**Descrição:** Script de verificação de correções.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'portfolio' / '01_fraud_neuromorphic' / 'src'))

from models_snn import FraudSNN

def verify_fix():
 print(" Iniciando verificação do fix do Brian2...")
 
 # 1. Instanciar modelo
 print("1. Instanciando FraudSNN...")
 try:
 snn = FraudSNN(input_size=10, hidden_sizes=[10], output_size=2)
 print(" Modelo instanciado.")
 except Exception as e:
 print(f" Erro na instanciação: {e}")
 return

 # 2. Gerar dados dummy
 print("2. Gerando dados dummy...")
 n_samples = 5
 spike_data = []
 for i in range(n_samples):
 # 10 neurônios de entrada, spikes aleatórios
 indices = np.arange(10)
 times = np.random.rand(10) * 0.05 # 50ms
 label = np.random.randint(0, 2)
 spike_data.append((times, indices, label))
 
 # 3. Testar train_stdp
 print("3. Testando train_stdp (que chama forward)...")
 try:
 snn.train_stdp(spike_data, epochs=2, duration=0.1)
 print(" train_stdp executou com sucesso!")
 except Exception as e:
 print(f" Erro em train_stdp: {e}")
 import traceback
 traceback.print_exc()
 return

 print("\n Verificação concluída com SUCESSO! O erro ValueError foi corrigido.")

if __name__ == "__main__":
 verify_fix()
