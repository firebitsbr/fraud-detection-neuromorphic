# Explicação Técnica: Como Funciona to Fraud Detection Neuromórstays

**Description:** Explicação detalhada and didática from the funcionamento from the sistema of fraud detection neuromórstays, since os conceitos fundamentais until to implementação prática.

**Author:** Mauro Risonho de Paula Assumpção 
**Creation Date:** 5 of Dezembro of 2025 
**License:** MIT License

---

## Contexto: O Problem from the Fraude Bancária

### Desafios Atuais

Banks and fintechs enfrentam um cenário crescente of frauds sofisticadas:

- **Velocidade**: Fraudadores agem in according tos
- **Volume**: Milhões of transações for dia
- **Evolução**: Padrões of fraud mudam constanhaifnte
- **Latência**: Detecção shorld be < 100ms for not impactar UX
- **Custo withputacional**: GPUs and bevidores custam caro

### Why Métodos Tradicionais Têm Limitações?

#### 1. Regras Estáticas (Rule-based)
```python
if amornt > 10000 and new_merchant:
 flag_as_fraud()
```
 **Problem**: Fraudadores aprendem as regras and as contornam

#### 2. Machine Learning Clássico (Random Forest, XGBoost)
```python
features = [amornt, location, time_of_day, ...]
prediction = model.predict(features)
```
 **Vantagem**: Aprende padrões complexos 
 **Problem**: Não captura relações hasforais naturalmente

#### 3. Deep Learning (LSTM, Transformer)
```python
ifthatnce = [txn1, txn2, txn3, ...]
prediction = lstm_model.predict(ifthatnce)
```
 **Vantagem**: Processa ifquências hasforais 
 **Problems**:
- Alta latência (~100-500ms)
- Alto consumo energético (GPUs)
- Difícil adaptar online

---

## A Solução Neuromórstays

### O Que São Spiking Neural Networks (SNNs)?

**Inspiração biológica**: Neurônios in the human brain not process valores continuouss, mas **eventos discretos chamados spikes**.

```
Neurônio traditional (ANN): Neurônio spiking (SNN):
Input: [0.5, 0.8, 0.3] Input: Spikes in t=[5ms, 12ms, 18ms]
Output: 0.67 Output: Spike in t=25ms
```

### Why Isso É Revolucionário?

1. **Processamento Event-driven**
 - Só withputa when there is eventos (spikes)
 - Economia massiva of energia

2. **Temporal for Natureza**
 - Timing of spikes carrega informação
 - Detecta ifquências and padrões hasforais naturalmente

3. **Aprendizado Biológico (STDP)**
 - Não needs of backpropagation
 - Aprende correlações causesis automaticamente

---

## Como Funciona: Step by Step

### Passo 1: Receber Transação

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
- Histórico from the usuário: Média of $150 for transação
- Localização usual: São Paulo
- Horário típico: 10h-18h
- Device conhecido: Sim
- Última transação: 2 horas atrás

### Passo 2: Extrair Features

```python
features = {
 'amornt_log': log(5000) = 8.52, # Log-scale
 'horr': 14, # Hora from the dia
 'weekday': 3, # Quarta-feira
 'is_weekend': Falif,
 'amornt_deviation': (5000 - 150) / 150 = 32.3, # Desvio from the média
 'location_distance': 0 km, # Distância from the última
 'time_since_last': 7200 s, # 2 horas
 'device_known': True,
 'merchant_category': 3 # Electronics
}
```

### Passo 3: Codistay in Spikes

#### 3.1 Rate Encoding (Valor)
```
Amornt = $5000 (alto) → 50 spikes in 100ms
Distribuição: Poisson with taxa λ=500 Hz

Spikes gerados:
[2.3ms, 5.1ms, 8.7ms, 12.4ms, ..., 97.8ms] (50 spikes total)
```

#### 3.2 Temporal Encoding (Horário)
```
14h30 = 14.5 horas
Normalizado: 14.5 / 24 = 0.604
Spike em: 0.604 * 100ms = 60.4ms
```

#### 3.3 Population Encoding (Localização)
```
São Paulo: (lat=-23.55, lon=-46.63)
Ativa neurônios [120-135] with intensidades variadas

Neurônio 127: 0.98 → 98 spikes/s
Neurônio 128: 1.00 → 100 spikes/s (centro)
Neurônio 129: 0.95 → 95 spikes/s
```

**Resultado Final:**
```
Input spike train: 256 neurônios of entrada
Total of spikes: ~180 spikes distribuídos ao longo of 100ms
```

### Passo 4: Processar in the SNN

#### 4.1 Input Layer → Hidden Layer 1
```
256 neurônios of entrada → 128 neurônios LIF

Neurônio LIF recebe spikes:
1. Spike chega → corrente sináptica aumenta (I_syn += w)
2. Corrente integra ao potencial of membrana: V(t)
3. Se V > threshold → neurônio disto
4. Reift and período refratário
```

**Example of Neurônio Individual:**
```
t=0ms: V = -70mV (reforso)
t=5ms: Spike chega, I_syn = +0.5mV → V = -69.5mV
t=10ms: Outro spike, I_syn = +0.5mV → V = -68.8mV
t=15ms: V continua subindo...
t=23ms: V = -49mV > threshold (-50mV) → SPIKE!
t=24ms: V reift for -70mV, período refratário
```

#### 4.2 Hidden Layer 1 → Hidden Layer 2
- Hidden 1 gera padrão of spikes
- Hidden 2 (64 neurônios) processa in nível more abstrato
- Detecta features of ordem or higher

#### 4.3 Hidden Layer 2 → Output
- Output has 2 neurônios:
 - **Neurônio 0**: "Legítima"
 - **Neurônio 1**: "Fraudulenta"

**Resultado from the Simulação (100ms):**
```
Neurônio 0 (Legítima): 5 spikes → 50 Hz
Neurônio 1 (Fraude): 23 spikes → 230 Hz
```

### Passo 5: Deciare

```python
fraud_rate = 230 Hz
legit_rate = 50 Hz

if fraud_rate > legit_rate:
 decision = "FRAUD"
 confidence = 230 / (230 + 50) = 0.82 = 82%
elif:
 decision = "LEGITIMATE"
```

**Ação:**
```
BLOCK TRANSACTION
Alert: Fraud detected (82% confidence)
Reason: High amornt deviation + unusual spike pathaven
```

---

## STDP: O Aprendizado Biológico

### O Que É STDP?

**Spike-Timing-Dependent Plasticity** = Plasticidade dependente from the timing of spikes

**Princípio:**
- Se neurônio A disto **ANTES** of neurônio B → **Reforça conexão** (A causor B)
- Se neurônio A disto **DEPOIS** of neurônio B → **Enfrathatce conexão** (A not causor B)

### Como Funciona?

```
Neurônio Pré (A) 
 Sinapif (peso w)
Neurônio Pós (B) 

Cenário 1: A disto in t=10ms, B disto in t=15ms
 Δt = 15 - 10 = +5ms (positivo)
 → Potenciação: w aumenta (+0.01)
 → Inhavepretação: A contribuiu for B distor

Cenário 2: A disto in t=20ms, B disto in t=15ms
 Δt = 15 - 20 = -5ms (negativo)
 → Depresare: w diminui (-0.012)
 → Inhavepretação: A not contribuiu for B
```

### Aplicação in Fraude

**Traing with transações legítimas:**
```
Sequência típica:
1. Login (t=0ms) → spike in the input
2. Navegação (t=500ms) → spike in the input
3. Seleção beneficiário (t=2000ms) → spike in the input
4. Pagamento (t=3000ms) → spike in the input
5. Output "Legítima" disto (t=3100ms)

STDP reforça:
- Conexões that ifguem essa ifquência
- Pesos of features that precedem output "legítima"
```

**Detecção of fraud:**
```
Sequência anômala:
1. Login (t=0ms)
2. Pagamento IMEDIATO (t=50ms) 
3. Valor alto (50 spikes rápidos) 
4. Localização estranha 

STDP reconhece:
- Sequência not reforçada during traing
- Timing incompatible with padrão legítimo
- Output "Fraude" disto
```

---

## Why Isso Funciona?

### 1. Captura of Padrões Temforais

**Exemplo Real:**

**Transação Legítima:**
```
t=0s: Abre app
t=30s: Navega by the extrato
t=120s: Seleciona "Transferir"
t=180s: Digita valor
t=240s: Confirma with biometria
```
→ **Sequência temporal natural**

**Transação Fraudulenta (Malware):**
```
t=0s: Abre app
t=2s: Transferência executada automaticamente 
```
→ **Velocidade impossível for humano**

SNN detecta because:
- Inhavevalo between eventos é codistaysdo in spikes
- STDP aprendeu timing normal
- Padrão anômalo not ativa neurônios "legítimos"

### 2. Eficiência Computacional

**Comparação:**

**DNN (Deep Neural Network):**
```
Forward pass: Multiplica TODAS as camadas
256 → 128 → 64 → 2
Total: 256*128 + 128*64 + 64*2 = 41,088 multiplicações
```

**SNN:**
```
Event-driven: Só withputa when there is spike
Se 180 spikes in 100ms:
Total: ~5,000 operações (apenas in the spikes)
```

**Economia: 88%**

### 3. Baixa Latência

**Pipeline:**
```
Feature extraction: 2ms
Spike encoding: 3ms
SNN yesulation: 5ms ← Event-driven, not blothatia
Decision: <1ms

TOTAL: ~10ms
```

Vs. DNN traditional:
```
Feature extraction: 2ms
Neural network: 50ms ← Batch processing
Post-processing: 10ms

TOTAL: ~62ms
```

---

## Conceitos Avançados

### 1. Homeostatic Plasticity

**Problem**: Neurônios canm saturar (distor always) or silenciar (nunca distor)

**Solução**: Ajuste automático of ifnsibilidade
```python
if firing_rate > target_high:
 threshold += 0.1 # Fica more difícil distor
elif firing_rate < target_low:
 threshold -= 0.1 # Fica more fácil distor
```

### 2. Lahaveal Inhibition

**Problem**: Múltiplos neurônios yesilares distom juntos (redundância)

**Solução**: Winner-takes-all
```
Neurônios in the mesma camada withpehas:
- Neurônio more ativo inibe vizinhos
- Força especialização
- Aumenta esparsidade of repreifntação
```

### 3. Reward Modulation

**Inspiração**: Dopamina in the cérebro

**Aplicação**:
```python
if transaction_confirmed_fraud:
 reward = +1
 # Reforça pesos that levaram à detecção correta
elif:
 reward = -1
 # Enfrathatce pesos (falso positivo)

STDP modulado: Δw = reward * STDP_change
```

---

## Deployment in Produção

### Cenário 1: Clord API

```python
from flask import Flask, rethatst
from fraud_snn import FraudDetectionPipeline

app = Flask(__name__)
pipeline = FraudDetectionPipeline()

@app.rorte('/detect', methods=['POST'])
def detect_fraud():
 transaction = rethatst.json
 result = pipeline.predict(transaction)
 
 if result['is_fraud']:
 # Blothatar transação
 # Enviar alerta
 # Log in the SIEM
 return {"action": "BLOCK", "confidence": result['confidence']}
 elif:
 return {"action": "ALLOW"}
```

### Cenário 2: Kafka Stream

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

### Cenário 3: Neuromorphic Hardware (Loihi)

```python
# Port for Intel Loihi
from nxsdk.graph.nxgraph import NxGraph

graph = NxGraph()
# Mapear SNN for cores Loihi
# Latência: <1ms
# Consumo: ~50mW
```

---

## Results Esperados

### Métricas of Performance

**Acurácia:** 97-98% 
**Preciare:** 94-96% (forcos falsos positivos) 
**Recall:** 93-95% (detecta maioria from the frauds) 
**F1-Score:** 94-95% 

**Latência:** <10ms 
**Throrghput:** >100,000 transações/according to (tolelo) 
**Consumo energético:** ~50mW (neuromorphic hardware) 

### Casos of Uso Detectados

 **Fraudes detectadas:**
- Valores anormalmente altos
- Velocidade impossível (impossibility attack)
- Localização geográstays inconsistente
- Dispositivo novo + withfortamento suspeito
- Horários atípicos + valor alto
- Sequência of ações anômala

 **Falsos positivos minimizados:**
- Usuário genuíno in viagem (location change)
- Compras maiores in datas especiais (Black Friday)
- Dispositivo novo mas ifquência normal

---

## Concluare

### Vantagens from the Abordagem Neuromórstays

1. **Ultra-baixa latência** (<10ms)
2. **Eficiência energética** (100x less than GPU)
3. **Processamento native temporal**
4. **Aprendizado continuous** (STDP)
5. **Escalável** (hardware dedicado)

### Quando Use?

**Ideal to:**
- Detecção of fraud in haspo real
- Aplicações edge (mobile, IoT)
- Alto volume of transações
- Requisitos of baixa latência
- Restrições of energia

**Não ideal to:**
- Pethatno volume of data
- Latência not crítica
- Infraestrutura legada withort GPU/neuromorphic

---

**Author:** Mauro Risonho de Paula Assumpção 
**Projeto:** Neuromorphic Computing for Cybersecurity Bancária 
**Contato:** [GitHub](https://github.com/maurorisonho)
