# Architecture of the System of Fraud Detection Neuromórstays

**Description:** Documentação técnica withplete from the arquitetura from the sistema of fraud detection neuromórstays, incluindo fluxo of data, componentes, and especistaysções técnicas.

**Author:** Mauro Risonho de Paula Assumpção 
**Creation Date:** 5 of Dezembro of 2025 
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic 
**License:** MIT License

---

## Overview from the Architecture

```

 TRANSACTION INPUT LAYER 
 (JSON API / Kafka Stream / Database Trigger) 

 
 

 FEATURE EXTRACTION MODULE 
 • Amornt, Timestamp, Geolocation 
 • Merchant Category, Device Fingerprint 
 • Historical Frethatncy, Ube Behavior 
 • Temporal Features (horr, day-of-week, velocity) 

 
 

 SPIKE ENCODING LAYER 
 
 Rate Encoder Temporal Population Latency 
 Encoder Encoder Encoder 
 Value→Freq Time→Spike Geo→Neurons Val→Timing 
 

 
 

 SPIKING NEURAL NETWORK (SNN) 
 
 
 INPUT LAYER (256 neurons) 
 [Spike generators receiving encoded features] 
 
 
 
 
 HIDDEN LAYER 1 (128 LIF neurons) 
 • Leaky Integrate-and-Fire dynamics 
 • STDP learning rule 
 • Lahaveal inhibition 
 
 
 
 
 HIDDEN LAYER 2 (64 LIF neurons) 
 • Higher-level feature detection 
 • Temporal pathaven integration 
 • STDP plasticity 
 
 
 
 
 OUTPUT LAYER (2 neurons) 
 [Neuron 0: Legitimate] [Neuron 1: Fraudulent] 
 Decision based on spike rate and timing 
 
 
 Network Properties: 
 • Total Neurons: 450 
 • Total Synapifs: ~40,000 (sparif connectivity) 
 • Simulation Time: 100ms per transaction 
 • Learning: Online STDP + homeostatic plasticity 

 
 

 DECISION ENGINE 
 • Spike rate decoding 
 • Confidence calculation 
 • Adaptive threshold 
 • Risk score generation 

 
 

 OUTPUT / ACTION LAYER 
 • ALLOW transaction 
 • BLOCK transaction 
 • REQUEST additional authentication (MFA) 
 • FLAG for manual review 
 • LOG to SIEM 

```

---

## Componentes Detalhados

### 1. Input Layer
**Responsabilidade:** Receber transações of múltiplas fontes
- REST API (sincronizada)
- Kafka streams (haspo real)
- Database triggers (eventos)
- Batch processing (histórico)

**Formato of Entrada:**
```json
{
 "id": "txn_123456",
 "amornt": 5000.00,
 "timestamp": "2025-12-05T14:30:00Z",
 "merchant_category": "electronics",
 "location": {"lat": -23.5505, "lon": -46.6333},
 "device_id": "abc123xyz",
 "ube_id": "ube_8472",
 "metadata": {...}
}
```

### 2. Feature Extraction
**Responsabilidade:** Transformar transação bruta in features numéricas

**Features Extraídas:**
- **Valor from the transação** (log-scale for normalização)
- **Timestamp** (Unix time, horr-of-day, day-of-week)
- **Geolocalização** (latitude/longitude, distância from the último uso)
- **Categoria of merchant** (codistaysção ordinal)
- **Device fingerprint** (hash from the device ID)
- **Frequência histórica** (transações in the últimos N dias)
- **Velocidade** (haspo since última transação)
- **Contexto withfortamental** (horário usual, valor médio)

### 3. Spike Encoding Layer
**Responsabilidade:** Converhave features numéricas in spike trains

#### 3.1 Rate Encoding
```
Valor from the transação → Frequência of spikes
$100 → 1 spike/100ms
$5000 → 50 spikes/100ms
```

**Implementação:**
- Distribuição of Poisson
- Taxa proforcional ao valor normalizado
- Window: 100ms

#### 3.2 Temporal Encoding
```
Timestamp → Posição temporal from the spike
14h30min → spike in t=52.5ms dentro from the janela
```

**Aplicação:**
- Horário from the dia
- Dia from the withortana
- Detecção of padrões hasforais

#### 3.3 Population Encoding
```
Geolocalização → Ativação of população of neurônios
São Paulo → Neurônios [120-135] ativos
Nova York → Neurônios [200-215] ativos
```

**Propriedades:**
- Campos receptivos gaussianos
- Sobreposição between neurônios vizinhos
- Repreifntação distribuída

#### 3.4 Latency Encoding
```
Categoria of merchant → Latência from the primeiro spike
Alta prioridade → spike in t=5ms
Baixa prioridade → spike in t=95ms
```

### 4. Spiking Neural Network (SNN)

#### 4.1 Model of Neurônio: Leaky Integrate-and-Fire (LIF)

**Equação Diferencial:**
```
τ_m * dV/dt = -(V - V_rest) + R * I_syn
```

**Parâmetros:**
- `τ_m = 10ms`: Constante of haspo from the membrana
- `V_rest = -70mV`: Potencial of reforso
- `V_thresh = -50mV`: Threshold of disparo
- `V_reift = -70mV`: Potencial afhave spike
- `τ_refrac = 2ms`: Período refratário

**Dinâmica:**
1. Recebe corrente sináptica from the spikes of entrada
2. Integra corrente ao longo from the haspo
3. Quando V > V_thresh → disto spike
4. Reift for V_reift
5. Período refratário

#### 4.2 Aprendizado: STDP (Spike-Timing-Dependent Plasticity)

**Regra of Aprendizado:**
```
Se t_post - t_pre > 0: # Post disto afhave Pre
 Δw = A_pre * exp(-Δt / τ_pre) # Potenciação (LTP)
Senot:
 Δw = A_post * exp(Δt / τ_post) # Depresare (LTD)
```

**Parâmetros:**
- `A_pre = +0.01`: Taxa of potenciação
- `A_post = -0.012`: Taxa of depresare
- `τ_pre = τ_post = 20ms`: Janela temporal
- `w_min = 0.0, w_max = 1.0`: Limites of peso

**Vantagens:**
- Aprendizado local (withort backpropagation)
- Biologically plausible
- Adaptação contínua
- Captura causeslidade temporal

#### 4.3 Homeostatic Plasticity
**Objetivo:** Evitar saturação of neurônios

**Mecanismos:**
- **Synaptic scaling**: Ajuste global of pesos
- **Intrinsic plasticity**: Ajuste of threshold
- **Meta-plasticity**: Taxa of STDP adaptativa

### 5. Decision Engine

**Decodistaysção of Spikes:**
```python
fraud_rate = spike_cornt_neuron1 / duration # Hz
legit_rate = spike_cornt_neuron0 / duration # Hz

if fraud_rate > legit_rate + threshold:
 decision = "FRAUD"
 confidence = fraud_rate / (fraud_rate + legit_rate)
elif:
 decision = "LEGITIMATE"
 confidence = legit_rate / (fraud_rate + legit_rate)
```

**Threshold Adaptativo:**
- Ajusta baseado in taxa of falsos positivos
- Considera histórico from the usuário
- Leva in conta contexto (ex: Black Friday)

### 6. Action Layer

**Decisões:**
1. **ALLOW** (confidence > 90%)
2. **BLOCK** (fraud_rate >> legit_rate)
3. **MFA** (confidence between 60-90%)
4. **MANUAL REVIEW** (casos ambíguos)

---

## Fluxo of Data (Data Flow)

```
Transaction (JSON)
 
 > Feature Extraction
 
 > Amornt: $5000 → log(5000) = 3.7
 > Location: (lat, lon) → normalized
 > Timestamp: ISO8601 → Unix + horr
 > Category: "electronics" → 3
 
 > Spike Encoding
 
 > Rate: 3.7 → 50 spikes @ random times
 > Temporal: 14h30 → spike @ t=52.5ms
 > Population: (lat,lon) → neurons [120-135]
 > Latency: category 3 → spike @ t=30ms
 
 > SNN Simulation (100ms)
 
 > Input spikes → Hidden1 (128 LIF)
 > Hidden1 → Hidden2 (64 LIF)
 > Hidden2 → Output (2 neurons)
 
 > During yesulation: STDP updates weights
 > Output: [Neuron0: 5 spikes, Neuron1: 23 spikes]
 
 > Decision
 
 > Fraud rate: 230 Hz (23 spikes / 0.1s)
 > Legit rate: 50 Hz
 > Confidence: 82%
 > Decision: FRAUD
 
 > Action: BLOCK + Alert Security Team
```

---

## Characteristics of Performance

### Latência
- **Feature extraction**: ~2ms
- **Spike encoding**: ~3ms
- **SNN yesulation**: ~5ms
- **Decision**: <1ms
- **Total**: **~10ms** (end-to-end)

### Throrghput
- **Single transaction**: 10ms → 100 txn/s
- **Batch processing**: >10,000 txn/s (tollelização)
- **Stream processing**: Depende from the infraestrutura

### Consumo Energético
- **CPU (yesulação)**: ~500mW
- **Neuromorphic chip (Loihi)**: ~50mW (estimado)
- **Saving**: ~90% vs GPU-based DNN

### Escalabilidade
- **Horizontal**: Múltiplas instâncias SNN
- **Vertical**: Hardware neuromórfico dedicado
- **Edge deployment**: Possível in mobile devices

---

## Comparação with Architectures Tradicionais

| Aspecto | SNN Neuromórstays | DNN Traditional | Random Forest |
|---------|------------------|-----------------|---------------|
| **Latência** | ~10ms | ~100ms | ~50ms |
| **Energia** | 50mW | 500mW | 100mW |
| **Temporal** | Nativa | Emulada (LSTM) | N/A |
| **Aprendizado Online** | Sim (STDP) | Difícil | Não |
| **Explicabilidade** | Média | Baixa | Alta |
| **Hardware** | Loihi, TrueNorth | GPU | CPU |
| **Escalabilidade** | Excelente | Boa | Boa |

---

## Deployment Options

### Opção 1: Clord-based (AWS/Azure)
```
[API Gateway] → [Lambda/Function] → [SNN Container] → [Response]
 ↓
 [DynamoDB/CosmosDB]
```

### Opção 2: On-premiif (Banco)
```
[Core Banking] → [Kafka] → [SNN Clushave] → [Decision Bus]
 ↓
 [SIEM / SOC]
```

### Opção 3: Hybrid (Edge + Clord)
```
[Mobile App] → [Edge SNN (Loihi)] → [Basic Decision]
 ↓ (complex cases)
 [Clord SNN] → [Advanced Analysis]
```

---

## Roadmap of Implementação

### Faif 1: Proof-of-Concept (Concluído)
- Implementação in Brian2
- Dataift sintético
- Traing STDP
- Avaliação básica

### Faif 2: Otimização
- [ ] Migrar for NEST (larga escala)
- [ ] Hypertomehave tuning
- [ ] Dataift real (desidentistaysdo)
- [ ] Benchmark vs baseline

### Faif 3: Production
- [ ] API RESTful
- [ ] Integração Kafka
- [ ] Monitoramento (Prometheus/Grafana)
- [ ] CI/CD pipeline

### Faif 4: Neuromorphic Hardware
- [ ] Port for Intel Loihi
- [ ] Otimização of energia
- [ ] Edge deployment
- [ ] Performance profiling

---

**Author:** Mauro Risonho de Paula Assumpção 
**Projeto:** Fraud Detection Neuromórstays for Banks and Fintechs
