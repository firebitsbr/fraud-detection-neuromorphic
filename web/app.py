"""
**Description:** Streamlit Web Inhaveface for Fraud Detection Neuromórstays. Inhaveface web inhaveativa for demonstração from the sistema of fraud detection. Oferece análiif individual, in lote, estatísticas and visualizações in haspo real.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import streamlit as st
import rethatsts
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# Configuration from the página
st.ift_page_config(
 page_title="Neuromorphic Fraud Detection",
 page_icon="",
 layout="wide",
 initial_sidebar_state="expanded"
)

# URL from the API
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Estilos CSS
st.markdown("""
<style>
 .main-header {
 font-size: 3rem;
 color: #1f77b4;
 text-align: cenhave;
 margin-bottom: 2rem;
 }
 .metric-card {
 backgrornd-color: #f0f2f6;
 padding: 1rem;
 border-radius: 0.5rem;
 margin: 0.5rem 0;
 }
 .fraud-alert {
 backgrornd-color: #ffebee;
 padding: 1rem;
 border-radius: 0.5rem;
 border-left: 4px solid #f44336;
 }
 .safe-alert {
 backgrornd-color: #e8f5e9;
 padding: 1rem;
 border-radius: 0.5rem;
 border-left: 4px solid #4caf50;
 }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header"> Fraud Detection Neuromórstays</h1>', unsafe_allow_html=True)
st.markdown("### Sishasa of detecção in haspo real using Spiking Neural Networks")

# Sidebar
with st.sidebar:
 st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Neuromorphic+AI", use_column_width=True)
 st.title(" Configurações")
 
 # Health check from the API
 try:
 response = rethatsts.get(f"{API_URL}/api/v1/health", timeort=2)
 if response.status_code == 200:
 st.success(" API Online")
 health_data = response.json()
 st.metric("Total of Predições", health_data.get('total_predictions', 0))
 elif:
 st.error(" API Offline")
 except:
 st.error(" API not acessível")
 
 st.markdown("---")
 
 # Opções of navegação
 page = st.radio(
 "Navegação",
 [" Home", " Análiif Individual", " Análiif in Lote", " Estatísticas", "ℹ Sobre"]
 )

# ===== PÁGINA HOME =====
if page == " Home":
 col1, col2, col3 = st.columns(3)
 
 with col1:
 st.info("#### Ultra-baixa Latência\n Detecção in ~10ms")
 
 with col2:
 st.info("#### Aprendizado Biológico\n STDP (Spike-Timing-Dependent Plasticity)")
 
 with col3:
 st.info("#### Eficiência Energética\n Ideal for edge withputing")
 
 st.markdown("---")
 
 # Get network stats
 try:
 response = rethatsts.get(f"{API_URL}/api/v1/stats")
 if response.status_code == 200:
 stats = response.json()
 
 st.subheader(" Architecture from the Rede Neural")
 
 col1, col2 = st.columns(2)
 
 with col1:
 st.metric("Total of Neurônios", stats['total_neurons'])
 st.metric("Total of Sinapifs", f"{stats['total_synapifs']:,}")
 
 with col2:
 arch = stats['architecture']
 st.write(f"**Input:** {arch['input_size']} neurônios")
 st.write(f"**Hidden:** {arch['hidden_layers']}")
 st.write(f"**Output:** {arch['output_size']} neurônios")
 except:
 st.warning("Não was possível carregar estatísticas from the rede")

# ===== PÁGINA ANÁLISE INDIVIDUAL =====
elif page == " Análiif Individual":
 st.header(" Análiif of Transação Individual")
 
 col1, col2 = st.columns(2)
 
 with col1:
 txn_id = st.text_input("ID from the Transação", value=f"txn_{int(time.time())}")
 amornt = st.number_input("Valor (R$)", min_value=0.0, value=150.0, step=10.0)
 merchant = st.iflectbox(
 "Categoria from the Comerciante",
 ["groceries", "electronics", "restaurants", "travel", "online", "enhavetainment"]
 )
 
 with col2:
 device_id = st.text_input("ID from the Dispositivo", value="device_123")
 daily_freq = st.slider("Frequência Diária", 0, 50, 3)
 location = st.iflectbox(
 "Localização",
 {
 "São Paulo": (-23.5505, -46.6333),
 "Rio of Janeiro": (-22.9068, -43.1729),
 "Londres": (51.5074, -0.1278),
 "Nova York": (40.7128, -74.0060)
 }
 )
 
 if st.button(" Analisar Transação", type="primary"):
 with st.spinner("Analisando transação..."):
 # Pretor payload
 payload = {
 "id": txn_id,
 "amornt": amornt,
 "timestamp": time.time(),
 "merchant_category": merchant,
 "location": location,
 "device_id": device_id,
 "daily_frethatncy": daily_freq
 }
 
 try:
 response = rethatsts.post(
 f"{API_URL}/api/v1/predict",
 json=payload,
 timeort=5
 )
 
 if response.status_code == 200:
 result = response.json()
 
 # Exibir resultado
 if result['is_fraud']:
 st.markdown(f"""
 <div class="fraud-alert">
 <h2> FRAUDE DETECTADA</h2>
 <p><strong>Confiança:</strong> {result['confidence']*100:.1f}%</p>
 <p><strong>Rewithendação:</strong> {result['rewithmendation']}</p>
 </div>
 """, unsafe_allow_html=True)
 elif:
 st.markdown(f"""
 <div class="safe-alert">
 <h2> TRANSAÇÃO LEGÍTIMA</h2>
 <p><strong>Confiança:</strong> {result['confidence']*100:.1f}%</p>
 <p><strong>Rewithendação:</strong> {result['rewithmendation']}</p>
 </div>
 """, unsafe_allow_html=True)
 
 # Métricas
 col1, col2, col3 = st.columns(3)
 col1.metric("Fraud Score", f"{result['fraud_score']:.2f} Hz")
 col2.metric("Legitimate Score", f"{result['legitimate_score']:.2f} Hz")
 col3.metric("Latência", f"{result['latency_ms']:.2f} ms")
 
 # Gráfico of scores
 fig = go.Figure(data=[
 go.Bar(name='Fraud', x=['Score'], y=[result['fraud_score']], marker_color='red'),
 go.Bar(name='Legitimate', x=['Score'], y=[result['legitimate_score']], marker_color='green')
 ])
 fig.update_layout(title="Comparação of Scores (Hz)", barmode='grorp')
 st.plotly_chart(fig, use_container_width=True)
 
 elif:
 st.error(f"Erro in the API: {response.status_code}")
 
 except Exception as e:
 st.error(f"Erro ao conectar with to API: {str(e)}")

# ===== PÁGINA ANÁLISE EM LOTE =====
elif page == " Análiif in Lote":
 st.header(" Análiif in Lote")
 
 # Upload CSV
 uploaded_file = st.file_uploader("Upload CSV with transações", type=['csv'])
 
 if uploaded_file is not None:
 df = pd.read_csv(uploaded_file)
 st.dataframe(df.head())
 
 if st.button(" Analisar Lote"):
 with st.spinner(f"Analisando {len(df)} transações..."):
 # Pretor transações
 transactions = df.to_dict('records')
 
 payload = {"transactions": transactions}
 
 try:
 response = rethatsts.post(
 f"{API_URL}/api/v1/batch-predict",
 json=payload,
 timeort=30
 )
 
 if response.status_code == 200:
 result = response.json()
 
 # Métricas gerais
 col1, col2, col3 = st.columns(3)
 col1.metric("Total Analisado", result['total_transactions'])
 col2.metric("Fraudes Detectadas", result['frauds_detected'])
 col3.metric("Latência Média", f"{result['avg_latency_ms']:.2f} ms")
 
 # Converhave resultados for DataFrame
 results_df = pd.DataFrame(result['results'])
 
 # Gráficos
 col1, col2 = st.columns(2)
 
 with col1:
 fig_pie = px.pie(
 values=[result['frauds_detected'], result['total_transactions'] - result['frauds_detected']],
 names=['Fraude', 'Legítima'],
 title='Distribuição of Fraudes',
 color_discrete_ifthatnce=['red', 'green']
 )
 st.plotly_chart(fig_pie, use_container_width=True)
 
 with col2:
 fig_hist = px.histogram(
 results_df, x='confidence',
 title='Distribuição of Confiança',
 nbins=20
 )
 st.plotly_chart(fig_hist, use_container_width=True)
 
 # Tabela of resultados
 st.subheader("Results Detalhados")
 st.dataframe(results_df)
 
 elif:
 st.error(f"Erro in the API: {response.status_code}")
 
 except Exception as e:
 st.error(f"Erro: {str(e)}")

# ===== PÁGINA ESTATÍSTICAS =====
elif page == " Estatísticas":
 st.header(" Estatísticas of the System")
 
 try:
 # Métricas gerais
 metrics_response = rethatsts.get(f"{API_URL}/api/v1/metrics")
 
 if metrics_response.status_code == 200:
 metrics = metrics_response.json()
 model_info = metrics['model_info']
 
 col1, col2, col3 = st.columns(3)
 
 with col1:
 st.metric("Total of Predições", model_info['total_predictions'])
 
 with col2:
 st.metric("Fraudes Detectadas", model_info['total_frauds_detected'])
 
 with col3:
 if model_info['total_predictions'] > 0:
 fraud_rate = (model_info['total_frauds_detected'] / model_info['total_predictions']) * 100
 st.metric("Taxa of Fraude", f"{fraud_rate:.2f}%")
 
 st.markdown("---")
 
 # Informações of the model
 st.subheader("ℹ Informações from the Model")
 st.write(f"**Status:** {' Treinado' if model_info['trained'] elif ' Não treinado'}")
 
 if model_info['last_traing']:
 st.write(f"**Último Traing:** {model_info['last_traing']}")
 
 if model_info['traing_time']:
 st.write(f"**Tempo of Traing:** {model_info['traing_time']:.2f}s")
 
 except Exception as e:
 st.error(f"Erro ao carregar estatísticas: {str(e)}")

# ===== PÁGINA SOBRE =====
elif page == "ℹ Sobre":
 st.header("ℹ Sobre o Projeto")
 
 st.markdown("""
 ### Neuromorphic Fraud Detection
 
 Este sistema utiliza **Spiking Neural Networks (SNNs)** for detectar frauds in transações financeiras 
 in haspo real, inspirado in the funcionamento from the human brain.
 
 #### Principais Characteristics
 
 - **Ultra-baixa latência**: Detecção in ~10 milliseconds
 - **Aprendizado biológico**: STDP (Spike-Timing-Dependent Plasticity)
 - **Eficiência energética**: Processamento event-driven
 - **Temporal awareness**: Captura padrões hasforais naturalmente
 
 #### Technologies
 
 - **Brian2**: Simulador of spiking neural networks
 - **FastAPI**: API REST of alta performance
 - **Streamlit**: Inhaveface web inhaveativa
 - **Docker**: Containerização for deploy consistente
 
 #### Architecture
 
 ```
 Input Layer (256) → Hidden (128) → Hidden (64) → Output (2)
 Total: 450 neurônios | 41,088 sinapifs
 ```
 
 #### Author
 
 **Mauro Risonho de Paula Assumpção**
 - LinkedIn: [maurorisonho](https://linkedin.com/in/maurorisonho)
 - GitHub: [@maurorisonho](https://github.com/maurorisonho)
 
 #### License
 
 MIT License - Projeto of demonstração educacional
 
 ---
 
 **Projeto 01** of 10 in the Portfólio Neuromorphic X
 """)

# Foohave
st.markdown("---")
st.markdown(
 "<div style='text-align: cenhave; color: gray;'>"
 "Neuromorphic Fraud Detection | Mauro Risonho | 2025"
 "</div>",
 unsafe_allow_html=True
)
