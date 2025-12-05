# Explicação Técnica: Como Funciona a Detecção de Fraude Neuromórfica

**Descrição:** Explicação detalhada e didática do funcionamento do sistema de detecção de fraude neuromórfica, desde os conceitos fundamentais até a implementação prática.

**Autor:** Mauro Risonho de Paula Assumpção  
**Data de Criação:** 5 de Dezembro de 2025  
**Licença:** MIT License

---

## 🎯 Contexto: O Problema da Fraude Bancária

### Desafios Atuais

Bancos e fintechs enfrentam um cenário crescente de fraudes sofisticadas:

- **Velocidade**: Fraudadores agem em segundos
- **Volume**: Milhões de transações por dia
- **Evolução**: Padrões de fraude mudam constantemente
- **Latência**: Detecção deve ser < 100ms para não impactar UX
- **Custo computacional**: GPUs e servidores custam caro

### Por Que Métodos Tradicionais Têm Limitações?

#### 1. Regras Estáticas (Rule-based)
```python
if amount > 10000 and new_merchant:
    flag_as_fraud()
```
❌ **Problema**: Fraudadores aprendem as regras e as contornam

#### 2. Machine Learning Clássico (Random Forest, XGBoost)
```python
features = [amount, location, time_of_day, ...]
prediction = model.predict(features)
```
✅ **Vantagem**: Aprende padrões complexos  
❌ **Problema**: Não captura relações temporais naturalmente

#### 3. Deep Learning (LSTM, Transformer)
```python
sequence = [txn1, txn2, txn3, ...]
prediction = lstm_model.predict(sequence)
```
✅ **Vantagem**: Processa sequências temporais  
❌ **Problemas**:
- Alta latência (~100-500ms)
- Alto consumo energético (GPUs)
- Difícil adaptar online

---

## 🧠 A Solução Neuromórfica

### O Que São Redes Neurais Spiking (SNNs)?

**Inspiração biológica**: Neurônios no cérebro humano não processam valores contínuos, mas **eventos discretos chamados spikes**.

```
Neurônio tradicional (ANN):      Neurônio spiking (SNN):
Input: [0.5, 0.8, 0.3]           Input: Spikes em t=[5ms, 12ms, 18ms]
Output: 0.67                     Output: Spike em t=25ms
```

### Por Que Isso É Revolucionário?

1. **Processamento Event-driven**
   - Só computa quando há eventos (spikes)
   - Economia massiva de energia

2. **Temporal por Natureza**
   - Timing de spikes carrega informação
   - Detecta sequências e padrões temporais naturalmente

3. **Aprendizado Biológico (STDP)**
   - Não precisa de backpropagation
   - Aprende correlações causais automaticamente

---

## 🔬 Como Funciona: Passo a Passo

### Passo 1: Receber Transação

```json
{
  "amount": 5000.00,
  "timestamp": "2025-12-05T14:30:00Z",
  "merchant": "Electronics Store",
  "location": {"lat": -23.5505, "lon": -46.6333},
  "device_id": "abc123xyz",
  "user_id": "user_8472"
}
```

**Contexto Adicional (Consultado):**
- Histórico do usuário: Média de $150 por transação
- Localização usual: São Paulo
- Horário típico: 10h-18h
- Device conhecido: Sim
- Última transação: 2 horas atrás

### Passo 2: Extrair Features

```python
features = {
    'amount_log': log(5000) = 8.52,          # Log-scale
    'hour': 14,                               # Hora do dia
    'weekday': 3,                             # Quarta-feira
    'is_weekend': False,
    'amount_deviation': (5000 - 150) / 150 = 32.3,  # Desvio da média
    'location_distance': 0 km,                # Distância da última
    'time_since_last': 7200 s,                # 2 horas
    'device_known': True,
    'merchant_category': 3                    # Electronics
}
```

### Passo 3: Codificar em Spikes

#### 3.1 Rate Encoding (Valor)
```
Amount = $5000 (alto) → 50 spikes em 100ms
Distribuição: Poisson com taxa λ=500 Hz

Spikes gerados:
[2.3ms, 5.1ms, 8.7ms, 12.4ms, ..., 97.8ms]  (50 spikes total)
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
Ativa neurônios [120-135] com intensidades variadas

Neurônio 127: 0.98 → 98 spikes/s
Neurônio 128: 1.00 → 100 spikes/s (centro)
Neurônio 129: 0.95 → 95 spikes/s
```

**Resultado Final:**
```
Input spike train: 256 neurônios de entrada
Total de spikes: ~180 spikes distribuídos ao longo de 100ms
```

### Passo 4: Processar na SNN

#### 4.1 Input Layer → Hidden Layer 1
```
256 neurônios de entrada → 128 neurônios LIF

Neurônio LIF recebe spikes:
1. Spike chega → corrente sináptica aumenta (I_syn += w)
2. Corrente integra ao potencial de membrana: V(t)
3. Se V > threshold → neurônio dispara
4. Reset e período refratário
```

**Exemplo de Neurônio Individual:**
```
t=0ms:   V = -70mV (repouso)
t=5ms:   Spike chega, I_syn = +0.5mV → V = -69.5mV
t=10ms:  Outro spike, I_syn = +0.5mV → V = -68.8mV
t=15ms:  V continua subindo...
t=23ms:  V = -49mV > threshold (-50mV) → SPIKE!
t=24ms:  V reset para -70mV, período refratário
```

#### 4.2 Hidden Layer 1 → Hidden Layer 2
- Hidden 1 gera padrão de spikes
- Hidden 2 (64 neurônios) processa em nível mais abstrato
- Detecta features de ordem superior

#### 4.3 Hidden Layer 2 → Output
- Output tem 2 neurônios:
  - **Neurônio 0**: "Legítima"
  - **Neurônio 1**: "Fraudulenta"

**Resultado da Simulação (100ms):**
```
Neurônio 0 (Legítima):  5 spikes  → 50 Hz
Neurônio 1 (Fraude):    23 spikes → 230 Hz
```

### Passo 5: Decisão

```python
fraud_rate = 230 Hz
legit_rate = 50 Hz

if fraud_rate > legit_rate:
    decision = "FRAUD"
    confidence = 230 / (230 + 50) = 0.82 = 82%
else:
    decision = "LEGITIMATE"
```

**Ação:**
```
BLOCK TRANSACTION
Alert: Fraud detected (82% confidence)
Reason: High amount deviation + unusual spike pattern
```

---

## 🧬 STDP: O Aprendizado Biológico

### O Que É STDP?

**Spike-Timing-Dependent Plasticity** = Plasticidade dependente do timing de spikes

**Princípio:**
- Se neurônio A dispara **ANTES** de neurônio B → **Reforça conexão** (A causou B)
- Se neurônio A dispara **DEPOIS** de neurônio B → **Enfraquece conexão** (A não causou B)

### Como Funciona?

```
Neurônio Pré (A)  ─────┐
                       │ Sinapse (peso w)
Neurônio Pós (B)  ─────┘

Cenário 1: A dispara em t=10ms, B dispara em t=15ms
    Δt = 15 - 10 = +5ms (positivo)
    → Potenciação: w aumenta (+0.01)
    → Interpretação: A contribuiu para B disparar

Cenário 2: A dispara em t=20ms, B dispara em t=15ms
    Δt = 15 - 20 = -5ms (negativo)
    → Depressão: w diminui (-0.012)
    → Interpretação: A não contribuiu para B
```

### Aplicação em Fraude

**Treinamento com transações legítimas:**
```
Sequência típica:
1. Login (t=0ms) → spike no input
2. Navegação (t=500ms) → spike no input
3. Seleção beneficiário (t=2000ms) → spike no input
4. Pagamento (t=3000ms) → spike no input
5. Output "Legítima" dispara (t=3100ms)

STDP reforça:
- Conexões que seguem essa sequência
- Pesos de features que precedem output "legítima"
```

**Detecção de fraude:**
```
Sequência anômala:
1. Login (t=0ms)
2. Pagamento IMEDIATO (t=50ms) ⚠️
3. Valor alto (50 spikes rápidos) ⚠️
4. Localização estranha ⚠️

STDP reconhece:
- Sequência não reforçada durante treinamento
- Timing incompatível com padrão legítimo
- Output "Fraude" dispara
```

---

## 📊 Por Que Isso Funciona?

### 1. Captura de Padrões Temporais

**Exemplo Real:**

**Transação Legítima:**
```
t=0s:    Abre app
t=30s:   Navega pelo extrato
t=120s:  Seleciona "Transferir"
t=180s:  Digita valor
t=240s:  Confirma com biometria
```
→ **Sequência temporal natural**

**Transação Fraudulenta (Malware):**
```
t=0s:    Abre app
t=2s:    Transferência executada automaticamente ⚠️
```
→ **Velocidade impossível para humano**

SNN detecta porque:
- Intervalo entre eventos é codificado em spikes
- STDP aprendeu timing normal
- Padrão anômalo não ativa neurônios "legítimos"

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
Event-driven: Só computa quando há spike
Se 180 spikes em 100ms:
Total: ~5,000 operações (apenas nos spikes)
```

**Economia: 88%**

### 3. Baixa Latência

**Pipeline:**
```
Feature extraction:  2ms
Spike encoding:      3ms
SNN simulation:      5ms  ← Event-driven, não bloqueia
Decision:            <1ms
────────────────────────
TOTAL:               ~10ms
```

Vs. DNN tradicional:
```
Feature extraction:  2ms
Neural network:      50ms  ← Batch processing
Post-processing:     10ms
────────────────────────
TOTAL:               ~62ms
```

---

## 🎓 Conceitos Avançados

### 1. Homeostatic Plasticity

**Problema**: Neurônios podem saturar (disparar sempre) ou silenciar (nunca disparar)

**Solução**: Ajuste automático de sensibilidade
```python
if firing_rate > target_high:
    threshold += 0.1  # Fica mais difícil disparar
elif firing_rate < target_low:
    threshold -= 0.1  # Fica mais fácil disparar
```

### 2. Lateral Inhibition

**Problema**: Múltiplos neurônios similares disparam juntos (redundância)

**Solução**: Winner-takes-all
```
Neurônios na mesma camada competem:
- Neurônio mais ativo inibe vizinhos
- Força especialização
- Aumenta esparsidade de representação
```

### 3. Reward Modulation

**Inspiração**: Dopamina no cérebro

**Aplicação**:
```python
if transaction_confirmed_fraud:
    reward = +1
    # Reforça pesos que levaram à detecção correta
else:
    reward = -1
    # Enfraquece pesos (falso positivo)

STDP modulado: Δw = reward * STDP_change
```

---

## 🚀 Deployment em Produção

### Cenário 1: Cloud API

```python
from flask import Flask, request
from fraud_snn import FraudDetectionPipeline

app = Flask(__name__)
pipeline = FraudDetectionPipeline()

@app.route('/detect', methods=['POST'])
def detect_fraud():
    transaction = request.json
    result = pipeline.predict(transaction)
    
    if result['is_fraud']:
        # Bloquear transação
        # Enviar alerta
        # Log no SIEM
        return {"action": "BLOCK", "confidence": result['confidence']}
    else:
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
        producer.send('fraud_alerts', alert)
```

### Cenário 3: Hardware Neuromórfico (Loihi)

```python
# Port para Intel Loihi
from nxsdk.graph.nxgraph import NxGraph

graph = NxGraph()
# Mapear SNN para cores Loihi
# Latência: <1ms
# Consumo: ~50mW
```

---

## 📈 Resultados Esperados

### Métricas de Performance

**Acurácia:** 97-98%  
**Precisão:** 94-96% (poucos falsos positivos)  
**Recall:** 93-95% (detecta maioria das fraudes)  
**F1-Score:** 94-95%  

**Latência:** <10ms  
**Throughput:** >100,000 transações/segundo (paralelo)  
**Consumo energético:** ~50mW (hardware neuromórfico)  

### Casos de Uso Detectados

✅ **Fraudes detectadas:**
- Valores anormalmente altos
- Velocidade impossível (impossibility attack)
- Localização geográfica inconsistente
- Dispositivo novo + comportamento suspeito
- Horários atípicos + valor alto
- Sequência de ações anômala

❌ **Falsos positivos minimizados:**
- Usuário genuíno em viagem (location change)
- Compras maiores em datas especiais (Black Friday)
- Dispositivo novo mas sequência normal

---

## 🎯 Conclusão

### Vantagens da Abordagem Neuromórfica

1. ✅ **Ultra-baixa latência** (<10ms)
2. ✅ **Eficiência energética** (100x menos que GPU)
3. ✅ **Processamento temporal nativo**
4. ✅ **Aprendizado contínuo** (STDP)
5. ✅ **Escalável** (hardware dedicado)

### Quando Usar?

**Ideal para:**
- Detecção de fraude em tempo real
- Aplicações edge (mobile, IoT)
- Alto volume de transações
- Requisitos de baixa latência
- Restrições de energia

**Não ideal para:**
- Pequeno volume de dados
- Latência não crítica
- Infraestrutura legada sem GPU/neuromorphic

---

**Autor:** Mauro Risonho de Paula Assumpção  
**Projeto:** Computação Neuromórfica para Cybersecurity Bancária  
**Contato:** [GitHub](https://github.com/maurorisonho)
