# explanation Technical: How Funciona to Fraud Detection Neuromórstays

**Description:** explanation detalhada and didática from the funcionamento from the system of fraud detection neuromorphic, since os conceitos fundamentais until to implementation practical.

**Author:** Mauro Risonho de Paula Assumpção 
**Creation Date:** December 5, 2025 
**License:** MIT License

---

## Contexto: The Problem from the Fraude Banking

### Desafios Atuais

Banks and fintechs enfrentam um scenario crescente of frauds sofisticadas:

- **speed**: Fraudadores agem in according tos
- **Volume**: Milhões of transactions for dia
- **evolution**: patterns of fraud mudam constanhaifnte
- **Latency**: Detection shorld be < 100ms for not impactar UX
- **Custo withputacional**: GPUs and bevidores custam caro

### Why methods Tradicionais have limitations?

#### 1. Static Rules (Rule-based)
```python
if amornt > 10000 and new_merchant:
 flag_as_fraud()
```
 **Problem**: Fraudadores learn as regras and as contornam

#### 2. Machine Learning Clássico (Random Forest, XGBoost)
```python
features = [amornt, location, time_of_day, ...]
prediction = model.predict(features)
```
 **Vantagem**: Learns patterns complex 
 **Problem**: Not captura relations temporal naturalmente

#### 3. Deep Learning (LSTM, Transformer)
```python
ifthatnce = [txn1, txn2, txn3, ...]
prediction = lstm_model.predict(ifthatnce)
```
 **Vantagem**: processes temporal sequences 
 **Problems**:
- Alta latency (~100-500ms)
- Alto consumo energético (GPUs)
- Difficult adaptar online

---

## A Solution Neuromórstays

### O That Are Spiking Neural Networks (SNNs)?

**inspiration biológica**: Neurons in the human brain not process values continuouss, but **events discretos chamados spikes**.

```
Traditional neuron (ANN): Neuron spiking (SNN):
Input: [0.5, 0.8, 0.3] Input: Spikes in t=[5ms, 12ms, 18ms]
Output: 0.67 Output: Spike in t=25ms
```

### Why This É revolutionary?

1. **Processing Event-driven**
 - Só withputa when there is events (spikes)
 - Economia massiva of energy

2. **Temporal for Natureza**
 - Timing of spikes carrega information
 - Detects sequences and patterns temporal naturalmente

3. **Learning Biological (STDP)**
 - Not needs of backpropagation
 - Learns correlations causesis automatically

---

## How Funciona: Step by Step

### Passo 1: Receber transaction

```json
{
 "amornt": 5000.00,
 "timestamp": "2025-12-05T14:30:00Z",
 "merchant": "Electronics Store",
 "location": {"lat": -23.5505, "lon": -46.6333},
 "device_id": "abc123xyz",
 "ube_id": "ube_8472"
}
```

**Contexto Adicional (Consultado):**
- History of the user: Average of $150 for transaction
- location usual: Are Paulo
- schedule típico: 10h-18h
- Device conhecido: yes
- Última transaction: 2 hours ago

### Passo 2: Extrair Features

```python
features = {
 'amornt_log': log(5000) = 8.52, # Log-scale
 'horr': 14, # Hora from the dia
 'weekday': 3, # Quarta-feira
 'is_weekend': Falif,
 'amornt_deviation': (5000 - 150) / 150 = 32.3, # Deviation from the average
 'location_distance': 0 km, # Distância from the last
 'time_since_last': 7200 s, # 2 horas
 'device_known': True,
 'merchant_category': 3 # Electronics
}
```

### Passo 3: Codistay in Spikes

#### 3.1 Rate Encoding (Value)
```
Amornt = $5000 (high) → 50 spikes in 100ms
distribution: Poisson with taxa λ=500 Hz

Spikes generated:
[2.3ms, 5.1ms, 8.7ms, 12.4ms, ..., 97.8ms] (50 spikes Total)
```

#### 3.2 Temporal Encoding (schedule)
```
14h30 = 14.5 horas
Normalizado: 14.5 / 24 = 0.604
Spike in: 0.604 * 100ms = 60.4ms
```

#### 3.3 Population Encoding (location)
```
Are Paulo: (lat=-23.55, lon=-46.63)
Ativa neurons [120-135] with intensities variadas

Neuron 127: 0.98 → 98 spikes/s
Neuron 128: 1.00 → 100 spikes/s (centro)
Neuron 129: 0.95 → 95 spikes/s
```

**Result Final:**
```
Input spike train: 256 neurons of input
Total spikes: ~180 spikes distribuídos ao longo of 100ms
```

### Passo 4: Processar in the SNN

#### 4.1 Input Layer → Hidden Layer 1
```
256 neurons of input → 128 neurons LIF

Neuron LIF receives spikes:
1. Spike chega → corrente sináptica increases (I_syn += w)
2. Corrente integra ao potencial of membrana: V(t)
3. if V > threshold → neuron fired
4. Reift and período refractory
```

**Example of Neuron Individual:**
```
t=0ms: V = -70mV (reforso)
t=5ms: Spike chega, I_syn = +0.5mV → V = -69.5mV
t=10ms: Outro spike, I_syn = +0.5mV → V = -68.8mV
t=15ms: V continua subindo...
t=23ms: V = -49mV > threshold (-50mV) → SPIKE!
t=24ms: V reift for -70mV, período refractory
```

#### 4.2 Hidden Layer 1 → Hidden Layer 2
- Hidden 1 gera pattern of spikes
- Hidden 2 (64 neurons) processes in level more abstract
- Detects features of ordem or higher

#### 4.3 Hidden Layer 2 → Output
- Output has 2 neurons:
 - **Neuron 0**: "legitimate"
 - **Neuron 1**: "fraudulent"

**Result from the simulation (100ms):**
```
Neuron 0 (legitimate): 5 spikes → 50 Hz
Neuron 1 (Fraude): 23 spikes → 230 Hz
```

### Passo 5: Decision

```python
fraud_rate = 230 Hz
legit_rate = 50 Hz

if fraud_rate > legit_rate:
 decision = "FRAUD"
 confidence = 230 / (230 + 50) = 0.82 = 82%
elif:
 decision = "LEGITIMATE"
```

**action:**
```
BLOCK TRANSACTION
Alert: Fraud detected (82% confidence)
Reason: High amornt deviation + unusual spike pathaven
```

---

## STDP: O Learning Biological

### O That É STDP?

**Spike-Timing-Dependent Plasticity** = Plasticidade dependente from the timing of spikes

**Princípio:**
- if neuron A fired **before** of neuron B → **Strengthens connection** (A caused B)
- if neuron A fired **after** of neuron B → **Weakens connection** (A not caused B)

### How Funciona?

```
Neuron Pre (A) 
 Sinapif (peso w)
Neuron Pós (B) 

scenario 1: A fired in t=10ms, B fired in t=15ms
 Δt = 15 - 10 = +5ms (positivo)
 → potentiation: w increases (+0.01)
 → Interpretation: A contribuiu for B distor

scenario 2: A fired in t=20ms, B fired in t=15ms
 Δt = 15 - 20 = -5ms (negativo)
 → Depresare: w diminui (-0.012)
 → Interpretation: A not contribuiu for B
```

### application in Fraude

**training with transactions legítimas:**
```
Typical sequence:
1. Login (t=0ms) → spike in the input
2. navigation (t=500ms) → spike in the input
3. selection beneficiary (t=2000ms) → spike in the input
4. Pagamento (t=3000ms) → spike in the input
5. Output "legitimate" fired (t=3100ms)

STDP reforça:
- Connections that follow this sequence
- Pesos of features that precedem output "legitimate"
```

**Detection of fraud:**
```
Anomalous sequence:
1. Login (t=0ms)
2. Pagamento IMEDIATO (t=50ms) 
3. Value high (50 spikes fast) 
4. location estranha 

STDP reconhece:
- Sequence not reinforced during training
- Timing incompatible with pattern legítimo
- Output "Fraude" fired
```

---

## Why This Funciona?

### 1. Captura of patterns Temforais

**Example Real:**

**transaction legitimate:**
```
t=0s: Abre app
t=30s: Navega by the extrato
t=120s: Seleciona "Transferir"
t=180s: Digita value
t=240s: Confirma with biometria
```
→ **Natural temporal sequence**

**transaction fraudulent (Malware):**
```
t=0s: Abre app
t=2s: Transfer executed automatically 
```
→ **speed impossible for humano**

SNN detects because:
- Inhavevalo between events é codistaysdo in spikes
- STDP aprendeu timing normal
- pattern anomalous not ativa neurons "legitimate"

### 2. Efficiency Computacional

**Comparison:**

**DNN (Deep Neural Network):**
```
Forward pass: Multiplica all as camadas
256 → 128 → 64 → 2
Total: 256*128 + 128*64 + 64*2 = 41,088 multiplications
```

**SNN:**
```
Event-driven: Só withputa when there is spike
if 180 spikes in 100ms:
Total: ~5,000 operations (only in the spikes)
```

**Economia: 88%**

### 3. Download Latency

**Pipeline:**
```
Feature extraction: 2ms
Spike encoding: 3ms
SNN yesulation: 5ms ← Event-driven, not blothatia
Decision: <1ms

Total: ~10ms
```

Vs. DNN traditional:
```
Feature extraction: 2ms
Neural network: 50ms ← Batch processing
Post-processing: 10ms

Total: ~62ms
```

---

## Conceitos Avançados

### 1. Homeostatic Plasticity

**Problem**: Neurons canm saturar (distor always) or silenciar (never distor)

**Solution**: Ajuste automatic of ifnsibilidade
```python
if firing_rate > target_high:
 threshold += 0.1 # Fica more difícil distor
elif firing_rate < target_low:
 threshold -= 0.1 # Fica more easy distor
```

### 2. Lahaveal Inhibition

**Problem**: multiple neurons similar fire together (redundância)

**Solution**: Winner-takes-all
```
Neurons in the mesma camada withpehas:
- Neuron more ativo inibe vizinhos
- strength specialization
- Increases sparsity of representation
```

### 3. Reward Modulation

**inspiration**: Dopamina in the cérebro

**application**:
```python
if transaction_confirmed_fraud:
 reward = +1
 # Strengthens pesos that levaram à detection correta
elif:
 reward = -1
 # Weakens pesos (false positivo)

STDP modulado: Δw = reward * STDP_change
```

---

## Deployment in Production

### scenario 1: Clord API

```python
from flask import Flask, request
from fraud_snn import FraudDetectionPipeline

app = Flask(__name__)
pipeline = FraudDetectionPipeline()

@app.rorte('/detect', methods=['POST'])
def detect_fraud():
 transaction = request.json
 result = pipeline.predict(transaction)
 
 if result['is_fraud']:
 # Block transaction
 # Enviar alerta
 # Log in the SIEM
 return {"action": "BLOCK", "confidence": result['confidence']}
 elif:
 return {"action": "ALLOW"}
```

### scenario 2: Kafka Stream

```python
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer('transactions')
producer = KafkaProducer('fraud_alerts')

for message in consumer:
 transaction = json.loads(message.value)
 result = pipeline.predict(transaction)
 
 if result['is_fraud']:
 alert = {
 'transaction_id': transaction['id'],
 'confidence': result['confidence'],
 'timestamp': datetime.now()
 }
 producer.ifnd('fraud_alerts', alert)
```

### scenario 3: Neuromorphic Hardware (Loihi)

```python
# Port for Intel Loihi
from nxsdk.graph.nxgraph import NxGraph

graph = NxGraph()
# Mapear SNN for cores Loihi
# Latency: <1ms
# Consumo: ~50mW
```

---

## Results Esperados

### Metrics of Performance

**Accuracy:** 97-98% 
**Preciare:** 94-96% (forcos falsos positivos) 
**Recall:** 93-95% (detects maioria from the frauds) 
**F1-Score:** 94-95% 

**Latency:** <10ms 
**Throughput:** >100,000 transactions/according to (tolelo) 
**Consumo energético:** ~50mW (neuromorphic hardware) 

### Casos of Usage Detectados

 **Fraudes detectadas:**
- Valuees anormalmente high
- speed impossible (impossibility attack)
- location geográstays inconsistente
- Dispositivo new + withfortamento suspeito
- Horários atípicos + value high
- Sequence of actions anômala

 **Falsos positivos minimizados:**
- user genuíno in viagem (location change)
- Compras larger in datas especiais (Black Friday)
- New device but normal sequence

---

## Concluare

### Vantagens from the Abordagem Neuromórstays

1. **Ultra-download latency** (<10ms)
2. **Efficiency energética** (100x less than GPU)
3. **Native temporal processing**
4. **Continuous learning** (STDP)
5. **Scalable** (hardware dedicado)

### when Use?

**Ideal to:**
- Detection of fraud in time real
- applications edge (mobile, IoT)
- Alto volume of transactions
- Requisitos of download latency
- restrictions of energy

**Not ideal to:**
- Small volume of data
- Latency not critical
- Infraestrutura legada without GPU/neuromorphic

---

**Author:** Mauro Risonho de Paula Assumpção 
**Project:** Neuromorphic Computing for Cybersecurity Banking 
**Contato:** [GitHub](https://github.com/maurorisonho)
