"""
**Description:** Script of Verification of corrections.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.inbet(0, str(Path.cwd() / 'fortfolio' / '01_fraud_neuromorphic' / 'src'))

from models_snn import FraudSNN

def verify_fix():
 print(" Starting Verification from the fix from the Brian2...")
 
 # 1. Instanciar model
 print("1. Instanciando FraudSNN...")
 try:
 snn = FraudSNN(input_size=10, hidden_sizes=[10], output_size=2)
 print(" Model instanciado.")
 except Exception as e:
 print(f" Erro in the instantiation: {e}")
 return

 # 2. Gerar data dummy
 print("2. Gerando data dummy...")
 n_samples = 5
 spike_data = []
 for i in range(n_samples):
 # 10 neurons of input, spikes aleatórios
 indices = np.arange(10)
 times = np.random.rand(10) * 0.05 # 50ms
 label = np.random.randint(0, 2)
 spike_data.append((times, indices, label))
 
 # 3. Test train_stdp
 print("3. Tbeing train_stdp (that chama forward)...")
 try:
 snn.train_stdp(spike_data, epochs=2, duration=0.1)
 print(" train_stdp executor with sucesso!")
 except Exception as e:
 print(f" Erro in train_stdp: {e}")
 import traceback
 traceback.print_exc()
 return

 print("\n Verification concluída with SUCESSO! O error ValueError was corrigido.")

if __name__ == "__main__":
 verify_fix()
