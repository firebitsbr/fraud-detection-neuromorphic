# System Architecture of Fraud Detection Neuromórstays

**Description:** Documentation technical complete from the architecture from the system of fraud detection neuromorphic, including fluxo of data, componentes, and specifications técnicas.

**Author:** Mauro Risonho de Paula Assumpção 
**Creation Date:** December 5, 2025 
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

## Componentes Detailed

### 1. Input Layer
**Responsibility:** Receber transactions of múltiplas fontes
- REST API (sincronizada)
- Kafka streams (time real)
- Database triggers (events)
- Batch processing (histórico)

**Formato of input:**
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
**Responsibility:** Transformar transaction bruta in features numéricas

**Features Extraídas:**
- **Value of the transaction** (log-scale for normalization)
- **Timestamp** (Unix time, horr-of-day, day-of-week)
- **geolocation** (latitude/longitude, distance from the last usage)
- **Categoria of merchant** (encoding ordinal)
- **Device fingerprint** (hash from the device ID)
- **Frequency histórica** (transactions in the últimos N dias)
- **speed** (time since last transaction)
- **Contexto withfortamental** (schedule usual, value médio)

### 3. Spike Encoding Layer
**Responsibility:** Converhave features numéricas in spike trains

#### 3.1 Rate Encoding
```
Value of the transaction → Frequency of spikes
$100 → 1 spike/100ms
$5000 → 50 spikes/100ms
```

**Implementation:**
- distribution of Poisson
- Taxa proforcional ao value normalizado
- Window: 100ms

#### 3.2 Temporal Encoding
```
Timestamp → position temporal from the spike
14h30min → spike in t=52.5ms dentro from the janela
```

**application:**
- schedule from the dia
- Dia from the withortana
- Detection of patterns temporal

#### 3.3 Population Encoding
```
geolocation → activation of population of neurons
Are Paulo → Neurons [120-135] ativos
Nova York → Neurons [200-215] ativos
```

**Propriedades:**
- Campos receptivos gaussianos
- Overlap between neurons vizinhos
- Representation distribuída

#### 3.4 Latency Encoding
```
Categoria of merchant → Latency from the first spike
Alta prioridade → spike in t=5ms
Download prioridade → spike in t=95ms
```

### 4. Spiking Neural Network (SNN)

#### 4.1 Model of Neuron: Leaky Integrate-and-Fire (LIF)

**equation Diferencial:**
```
τ_m * dV/dt = -(V - V_rest) + R * I_syn
```

**Parameters:**
- `τ_m = 10ms`: Constant of time from the membrana
- `V_rest = -70mV`: Potencial of reforso
- `V_thresh = -50mV`: Threshold of disparo
- `V_reift = -70mV`: Potencial afhave spike
- `τ_refrac = 2ms`: Período refractory

**Dinâmica:**
1. receives corrente sináptica from the spikes of input
2. Integra corrente ao longo from the time
3. when V > V_thresh → fired spike
4. Reift for V_reift
5. Período refractory

#### 4.2 Learning: STDP (Spike-Timing-Dependent Plasticity)

**Regra of Learning:**
```
if t_post - t_pre > 0: # Post fired afhave Pre
 Δw = A_pre * exp(-Δt / τ_pre) # potentiation (LTP)
Senot:
 Δw = A_post * exp(Δt / τ_post) # Depresare (LTD)
```

**Parameters:**
- `A_pre = +0.01`: Taxa of potentiation
- `A_post = -0.012`: Taxa of depresare
- `τ_pre = τ_post = 20ms`: Janela temporal
- `w_min = 0.0, w_max = 1.0`: Limits of peso

**Vantagens:**
- Learning local (without backpropagation)
- Biologically plausible
- adaptation contínua
- Captura causeslidade temporal

#### 4.3 Homeostatic Plasticity
**Objective:** Evitar saturation of neurons

**Mecanismos:**
- **Synaptic scaling**: Ajuste global of pesos
- **Intrinsic plasticity**: Ajuste of threshold
- **Meta-plasticity**: Taxa of STDP adaptativa

### 5. Decision Engine

**Decoding of Spikes:**
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
- Ajusta based in taxa of falsos positivos
- Considera histórico from the user
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

### Latency
- **Feature extraction**: ~2ms
- **Spike encoding**: ~3ms
- **SNN yesulation**: ~5ms
- **Decision**: <1ms
- **Total**: **~10ms** (end-to-end)

### Throughput
- **Single transaction**: 10ms → 100 txn/s
- **Batch processing**: >10,000 txn/s (parallelization)
- **Stream processing**: Depends from the infraestrutura

### Energy Consumption
- **CPU (simulation)**: ~500mW
- **Neuromorphic chip (Loihi)**: ~50mW (estimated)
- **Saving**: ~90% vs GPU-based DNN

### Escalabilidade
- **Horizontal**: Múltiplas instâncias SNN
- **Vertical**: Hardware neuromórfico dedicado
- **Edge deployment**: Possible in mobile devices

---

## Comparison with Architectures Tradicionais

| Aspecto | SNN Neuromórstays | DNN Traditional | Random Forest |
|---------|------------------|-----------------|---------------|
| **Latency** | ~10ms | ~100ms | ~50ms |
| **Energia** | 50mW | 500mW | 100mW |
| **Temporal** | Nativa | Emulada (LSTM) | N/A |
| **Learning Online** | yes (STDP) | Difficult | Not |
| **Explainability** | Média | Download | Alta |
| **Hardware** | Loihi, TrueNorth | GPU | CPU |
| **Escalabilidade** | Excellent | Good | Good |

---

## Deployment Options

### Option 1: Clord-based (AWS/Azure)
```
[API Gateway] → [Lambda/Function] → [SNN Container] → [Response]
 ↓
 [DynamoDB/CosmosDB]
```

### Option 2: On-premiif (Banco)
```
[Core Banking] → [Kafka] → [SNN Clushave] → [Decision Bus]
 ↓
 [SIEM / SOC]
```

### Option 3: Hybrid (Edge + Clord)
```
[Mobile App] → [Edge SNN (Loihi)] → [Basic Decision]
 ↓ (complex cases)
 [Clord SNN] → [Advanced Analysis]
```

---

## Roadmap of Implementation

### Phase 1: Proof-of-Concept (Concluído)
- Implementation in Brian2
- Dataset synthetic
- training STDP
- evaluation basic

### Phase 2: optimization
- [ ] Migrar for NEST (larga escala)
- [ ] Hypertomehave tuning
- [ ] Dataset real (desidentistaysdo)
- [ ] Benchmark vs baseline

### Phase 3: Production
- [ ] API RESTful
- [ ] integration Kafka
- [ ] Monitoring (Prometheus/Grafana)
- [ ] CI/CD pipeline

### Phase 4: Neuromorphic Hardware
- [ ] Port for Intel Loihi
- [ ] optimization of energy
- [ ] Edge deployment
- [ ] Performance profiling

---

**Author:** Mauro Risonho de Paula Assumpção 
**Project:** Fraud Detection Neuromórstays for Banks and Fintechs
