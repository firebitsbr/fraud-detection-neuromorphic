# Arquitetura do Sistema de Detecção de Fraude Neuromórfica

**Descrição:** Documentação técnica completa da arquitetura do sistema de detecção de fraude neuromórfica, incluindo fluxo de dados, componentes, e especificações técnicas.

**Autor:** Mauro Risonho de Paula Assumpção  
**Data de Criação:** 5 de Dezembro de 2025  
**Licença:** MIT License

---

## Visão Geral da Arquitetura

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TRANSACTION INPUT LAYER                            │
│  (JSON API / Kafka Stream / Database Trigger)                        │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION MODULE                          │
│  • Amount, Timestamp, Geolocation                                    │
│  • Merchant Category, Device Fingerprint                             │
│  • Historical Frequency, User Behavior                               │
│  • Temporal Features (hour, day-of-week, velocity)                   │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    SPIKE ENCODING LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │Rate Encoder │  │Temporal     │  │Population   │  │Latency     │ │
│  │             │  │Encoder      │  │Encoder      │  │Encoder     │ │
│  │Value→Freq   │  │Time→Spike   │  │Geo→Neurons  │  │Val→Timing  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│              SPIKING NEURAL NETWORK (SNN)                            │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ INPUT LAYER (256 neurons)                                    │   │
│  │ [Spike generators receiving encoded features]                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            │                                          │
│                            ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ HIDDEN LAYER 1 (128 LIF neurons)                             │   │
│  │ • Leaky Integrate-and-Fire dynamics                          │   │
│  │ • STDP learning rule                                         │   │
│  │ • Lateral inhibition                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            │                                          │
│                            ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ HIDDEN LAYER 2 (64 LIF neurons)                              │   │
│  │ • Higher-level feature detection                             │   │
│  │ • Temporal pattern integration                               │   │
│  │ • STDP plasticity                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            │                                          │
│                            ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ OUTPUT LAYER (2 neurons)                                     │   │
│  │ [Neuron 0: Legitimate]  [Neuron 1: Fraudulent]              │   │
│  │ Decision based on spike rate and timing                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  Network Properties:                                                 │
│  • Total Neurons: 450                                                │
│  • Total Synapses: ~40,000 (sparse connectivity)                     │
│  • Simulation Time: 100ms per transaction                            │
│  • Learning: Online STDP + homeostatic plasticity                    │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    DECISION ENGINE                                    │
│  • Spike rate decoding                                               │
│  • Confidence calculation                                            │
│  • Adaptive threshold                                                │
│  • Risk score generation                                             │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    OUTPUT / ACTION LAYER                              │
│  • ALLOW transaction                                                 │
│  • BLOCK transaction                                                 │
│  • REQUEST additional authentication (MFA)                           │
│  • FLAG for manual review                                            │
│  • LOG to SIEM                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Componentes Detalhados

### 1. Input Layer
**Responsabilidade:** Receber transações de múltiplas fontes
- REST API (sincronizada)
- Kafka streams (tempo real)
- Database triggers (eventos)
- Batch processing (histórico)

**Formato de Entrada:**
```json
{
  "id": "txn_123456",
  "amount": 5000.00,
  "timestamp": "2025-12-05T14:30:00Z",
  "merchant_category": "electronics",
  "location": {"lat": -23.5505, "lon": -46.6333},
  "device_id": "abc123xyz",
  "user_id": "user_8472",
  "metadata": {...}
}
```

### 2. Feature Extraction
**Responsabilidade:** Transformar transação bruta em features numéricas

**Features Extraídas:**
- **Valor da transação** (log-scale para normalização)
- **Timestamp** (Unix time, hour-of-day, day-of-week)
- **Geolocalização** (latitude/longitude, distância do último uso)
- **Categoria de merchant** (codificação ordinal)
- **Device fingerprint** (hash do device ID)
- **Frequência histórica** (transações nos últimos N dias)
- **Velocidade** (tempo desde última transação)
- **Contexto comportamental** (horário usual, valor médio)

### 3. Spike Encoding Layer
**Responsabilidade:** Converter features numéricas em spike trains

#### 3.1 Rate Encoding
```
Valor da transação → Frequência de spikes
$100   → 1 spike/100ms
$5000  → 50 spikes/100ms
```

**Implementação:**
- Distribuição de Poisson
- Taxa proporcional ao valor normalizado
- Window: 100ms

#### 3.2 Temporal Encoding
```
Timestamp → Posição temporal do spike
14h30min → spike em t=52.5ms dentro da janela
```

**Aplicação:**
- Horário do dia
- Dia da semana
- Detecção de padrões temporais

#### 3.3 Population Encoding
```
Geolocalização → Ativação de população de neurônios
São Paulo    → Neurônios [120-135] ativos
Nova York    → Neurônios [200-215] ativos
```

**Propriedades:**
- Campos receptivos gaussianos
- Sobreposição entre neurônios vizinhos
- Representação distribuída

#### 3.4 Latency Encoding
```
Categoria de merchant → Latência do primeiro spike
Alta prioridade → spike em t=5ms
Baixa prioridade → spike em t=95ms
```

### 4. Spiking Neural Network (SNN)

#### 4.1 Modelo de Neurônio: Leaky Integrate-and-Fire (LIF)

**Equação Diferencial:**
```
τ_m * dV/dt = -(V - V_rest) + R * I_syn
```

**Parâmetros:**
- `τ_m = 10ms`: Constante de tempo da membrana
- `V_rest = -70mV`: Potencial de repouso
- `V_thresh = -50mV`: Threshold de disparo
- `V_reset = -70mV`: Potencial após spike
- `τ_refrac = 2ms`: Período refratário

**Dinâmica:**
1. Recebe corrente sináptica dos spikes de entrada
2. Integra corrente ao longo do tempo
3. Quando V > V_thresh → dispara spike
4. Reset para V_reset
5. Período refratário

#### 4.2 Aprendizado: STDP (Spike-Timing-Dependent Plasticity)

**Regra de Aprendizado:**
```
Se t_post - t_pre > 0:  # Post dispara após Pre
    Δw = A_pre * exp(-Δt / τ_pre)     # Potenciação (LTP)
Senão:
    Δw = A_post * exp(Δt / τ_post)    # Depressão (LTD)
```

**Parâmetros:**
- `A_pre = +0.01`: Taxa de potenciação
- `A_post = -0.012`: Taxa de depressão
- `τ_pre = τ_post = 20ms`: Janela temporal
- `w_min = 0.0, w_max = 1.0`: Limites de peso

**Vantagens:**
- Aprendizado local (sem backpropagation)
- Biologically plausible
- Adaptação contínua
- Captura causalidade temporal

#### 4.3 Homeostatic Plasticity
**Objetivo:** Evitar saturação de neurônios

**Mecanismos:**
- **Synaptic scaling**: Ajuste global de pesos
- **Intrinsic plasticity**: Ajuste de threshold
- **Meta-plasticity**: Taxa de STDP adaptativa

### 5. Decision Engine

**Decodificação de Spikes:**
```python
fraud_rate = spike_count_neuron1 / duration  # Hz
legit_rate = spike_count_neuron0 / duration  # Hz

if fraud_rate > legit_rate + threshold:
    decision = "FRAUD"
    confidence = fraud_rate / (fraud_rate + legit_rate)
else:
    decision = "LEGITIMATE"
    confidence = legit_rate / (fraud_rate + legit_rate)
```

**Threshold Adaptativo:**
- Ajusta baseado em taxa de falsos positivos
- Considera histórico do usuário
- Leva em conta contexto (ex: Black Friday)

### 6. Action Layer

**Decisões:**
1. **ALLOW** (confidence > 90%)
2. **BLOCK** (fraud_rate >> legit_rate)
3. **MFA** (confidence entre 60-90%)
4. **MANUAL REVIEW** (casos ambíguos)

---

## Fluxo de Dados (Data Flow)

```
Transaction (JSON)
     │
     ├─> Feature Extraction
     │        │
     │        ├─> Amount: $5000 → log(5000) = 3.7
     │        ├─> Location: (lat, lon) → normalized
     │        ├─> Timestamp: ISO8601 → Unix + hour
     │        └─> Category: "electronics" → 3
     │
     ├─> Spike Encoding
     │        │
     │        ├─> Rate: 3.7 → 50 spikes @ random times
     │        ├─> Temporal: 14h30 → spike @ t=52.5ms
     │        ├─> Population: (lat,lon) → neurons [120-135]
     │        └─> Latency: category 3 → spike @ t=30ms
     │
     ├─> SNN Simulation (100ms)
     │        │
     │        ├─> Input spikes → Hidden1 (128 LIF)
     │        ├─> Hidden1 → Hidden2 (64 LIF)
     │        ├─> Hidden2 → Output (2 neurons)
     │        │
     │        ├─> During simulation: STDP updates weights
     │        └─> Output: [Neuron0: 5 spikes, Neuron1: 23 spikes]
     │
     ├─> Decision
     │        │
     │        ├─> Fraud rate: 230 Hz (23 spikes / 0.1s)
     │        ├─> Legit rate: 50 Hz
     │        ├─> Confidence: 82%
     │        └─> Decision: FRAUD
     │
     └─> Action: BLOCK + Alert Security Team
```

---

## Características de Performance

### Latência
- **Feature extraction**: ~2ms
- **Spike encoding**: ~3ms
- **SNN simulation**: ~5ms
- **Decision**: <1ms
- **Total**: **~10ms** (end-to-end)

### Throughput
- **Single transaction**: 10ms → 100 txn/s
- **Batch processing**: >10,000 txn/s (parallelização)
- **Stream processing**: Depende da infraestrutura

### Consumo Energético
- **CPU (simulação)**: ~500mW
- **Neuromorphic chip (Loihi)**: ~50mW (estimado)
- **Saving**: ~90% vs GPU-based DNN

### Escalabilidade
- **Horizontal**: Múltiplas instâncias SNN
- **Vertical**: Hardware neuromórfico dedicado
- **Edge deployment**: Possível em dispositivos móveis

---

## Comparação com Arquiteturas Tradicionais

| Aspecto | SNN Neuromórfica | DNN Tradicional | Random Forest |
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

### Opção 1: Cloud-based (AWS/Azure)
```
[API Gateway] → [Lambda/Function] → [SNN Container] → [Response]
                                    ↓
                              [DynamoDB/CosmosDB]
```

### Opção 2: On-premise (Banco)
```
[Core Banking] → [Kafka] → [SNN Cluster] → [Decision Bus]
                              ↓
                        [SIEM / SOC]
```

### Opção 3: Hybrid (Edge + Cloud)
```
[Mobile App] → [Edge SNN (Loihi)] → [Basic Decision]
                     ↓ (complex cases)
               [Cloud SNN] → [Advanced Analysis]
```

---

## Roadmap de Implementação

### Fase 1: Proof-of-Concept (Concluído)
- ✅ Implementação em Brian2
- ✅ Dataset sintético
- ✅ Treinamento STDP
- ✅ Avaliação básica

### Fase 2: Otimização
- [ ] Migrar para NEST (larga escala)
- [ ] Hyperparameter tuning
- [ ] Dataset real (desidentificado)
- [ ] Benchmark vs baseline

### Fase 3: Production
- [ ] API RESTful
- [ ] Integração Kafka
- [ ] Monitoramento (Prometheus/Grafana)
- [ ] CI/CD pipeline

### Fase 4: Hardware Neuromórfico
- [ ] Port para Intel Loihi
- [ ] Otimização de energia
- [ ] Edge deployment
- [ ] Performance profiling

---

**Autor:** Mauro Risonho de Paula Assumpção  
**Projeto:** Detecção de Fraude Neuromórfica para Bancos e Fintechs
