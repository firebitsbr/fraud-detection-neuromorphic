"""
**Descrição:** Streamlit Web Interface para Detecção de Fraude Neuromórfica. Interface web interativa para demonstração do sistema de detecção de fraude. Oferece análise individual, em lote, estatísticas e visualizações em tempo real.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# Configuração da página
st.set_page_config(
    page_title="Neuromorphic Fraud Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL da API
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Estilos CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .safe-alert {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 Detecção de Fraude Neuromórfica</h1>', unsafe_allow_html=True)
st.markdown("### Sistema de detecção em tempo real usando Spiking Neural Networks")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Neuromorphic+AI", use_column_width=True)
    st.title("⚙️ Configurações")
    
    # Health check da API
    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Online")
            health_data = response.json()
            st.metric("Total de Predições", health_data.get('total_predictions', 0))
        else:
            st.error("❌ API Offline")
    except:
        st.error("❌ API não acessível")
    
    st.markdown("---")
    
    # Opções de navegação
    page = st.radio(
        "Navegação",
        ["🏠 Home", "🔍 Análise Individual", "📊 Análise em Lote", "📈 Estatísticas", "ℹ️ Sobre"]
    )

# ===== PÁGINA HOME =====
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("#### ⚡ Ultra-baixa Latência\n Detecção em ~10ms")
    
    with col2:
        st.info("#### 🧬 Aprendizado Biológico\n STDP (Spike-Timing-Dependent Plasticity)")
    
    with col3:
        st.info("#### 🔋 Eficiência Energética\n Ideal para edge computing")
    
    st.markdown("---")
    
    # Get network stats
    try:
        response = requests.get(f"{API_URL}/api/v1/stats")
        if response.status_code == 200:
            stats = response.json()
            
            st.subheader("📐 Arquitetura da Rede Neural")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total de Neurônios", stats['total_neurons'])
                st.metric("Total de Sinapses", f"{stats['total_synapses']:,}")
            
            with col2:
                arch = stats['architecture']
                st.write(f"**Input:** {arch['input_size']} neurônios")
                st.write(f"**Hidden:** {arch['hidden_layers']}")
                st.write(f"**Output:** {arch['output_size']} neurônios")
    except:
        st.warning("Não foi possível carregar estatísticas da rede")

# ===== PÁGINA ANÁLISE INDIVIDUAL =====
elif page == "🔍 Análise Individual":
    st.header("🔍 Análise de Transação Individual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        txn_id = st.text_input("ID da Transação", value=f"txn_{int(time.time())}")
        amount = st.number_input("Valor (R$)", min_value=0.0, value=150.0, step=10.0)
        merchant = st.selectbox(
            "Categoria do Comerciante",
            ["groceries", "electronics", "restaurants", "travel", "online", "entertainment"]
        )
    
    with col2:
        device_id = st.text_input("ID do Dispositivo", value="device_123")
        daily_freq = st.slider("Frequência Diária", 0, 50, 3)
        location = st.selectbox(
            "Localização",
            {
                "São Paulo": (-23.5505, -46.6333),
                "Rio de Janeiro": (-22.9068, -43.1729),
                "Londres": (51.5074, -0.1278),
                "Nova York": (40.7128, -74.0060)
            }
        )
    
    if st.button("🚀 Analisar Transação", type="primary"):
        with st.spinner("Analisando transação..."):
            # Preparar payload
            payload = {
                "id": txn_id,
                "amount": amount,
                "timestamp": time.time(),
                "merchant_category": merchant,
                "location": location,
                "device_id": device_id,
                "daily_frequency": daily_freq
            }
            
            try:
                response = requests.post(
                    f"{API_URL}/api/v1/predict",
                    json=payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Exibir resultado
                    if result['is_fraud']:
                        st.markdown(f"""
                        <div class="fraud-alert">
                            <h2>⚠️ FRAUDE DETECTADA</h2>
                            <p><strong>Confiança:</strong> {result['confidence']*100:.1f}%</p>
                            <p><strong>Recomendação:</strong> {result['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                            <h2>✅ TRANSAÇÃO LEGÍTIMA</h2>
                            <p><strong>Confiança:</strong> {result['confidence']*100:.1f}%</p>
                            <p><strong>Recomendação:</strong> {result['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Métricas
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Fraud Score", f"{result['fraud_score']:.2f} Hz")
                    col2.metric("Legitimate Score", f"{result['legitimate_score']:.2f} Hz")
                    col3.metric("Latência", f"{result['latency_ms']:.2f} ms")
                    
                    # Gráfico de scores
                    fig = go.Figure(data=[
                        go.Bar(name='Fraud', x=['Score'], y=[result['fraud_score']], marker_color='red'),
                        go.Bar(name='Legitimate', x=['Score'], y=[result['legitimate_score']], marker_color='green')
                    ])
                    fig.update_layout(title="Comparação de Scores (Hz)", barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"Erro na API: {response.status_code}")
            
            except Exception as e:
                st.error(f"Erro ao conectar com a API: {str(e)}")

# ===== PÁGINA ANÁLISE EM LOTE =====
elif page == "📊 Análise em Lote":
    st.header("📊 Análise em Lote")
    
    # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV com transações", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        if st.button("🚀 Analisar Lote"):
            with st.spinner(f"Analisando {len(df)} transações..."):
                # Preparar transações
                transactions = df.to_dict('records')
                
                payload = {"transactions": transactions}
                
                try:
                    response = requests.post(
                        f"{API_URL}/api/v1/batch-predict",
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Métricas gerais
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Analisado", result['total_transactions'])
                        col2.metric("Fraudes Detectadas", result['frauds_detected'])
                        col3.metric("Latência Média", f"{result['avg_latency_ms']:.2f} ms")
                        
                        # Converter resultados para DataFrame
                        results_df = pd.DataFrame(result['results'])
                        
                        # Gráficos
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_pie = px.pie(
                                values=[result['frauds_detected'], result['total_transactions'] - result['frauds_detected']],
                                names=['Fraude', 'Legítima'],
                                title='Distribuição de Fraudes',
                                color_discrete_sequence=['red', 'green']
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_hist = px.histogram(
                                results_df, x='confidence',
                                title='Distribuição de Confiança',
                                nbins=20
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Tabela de resultados
                        st.subheader("Resultados Detalhados")
                        st.dataframe(results_df)
                        
                    else:
                        st.error(f"Erro na API: {response.status_code}")
                
                except Exception as e:
                    st.error(f"Erro: {str(e)}")

# ===== PÁGINA ESTATÍSTICAS =====
elif page == "📈 Estatísticas":
    st.header("📈 Estatísticas do Sistema")
    
    try:
        # Métricas gerais
        metrics_response = requests.get(f"{API_URL}/api/v1/metrics")
        
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            model_info = metrics['model_info']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Predições", model_info['total_predictions'])
            
            with col2:
                st.metric("Fraudes Detectadas", model_info['total_frauds_detected'])
            
            with col3:
                if model_info['total_predictions'] > 0:
                    fraud_rate = (model_info['total_frauds_detected'] / model_info['total_predictions']) * 100
                    st.metric("Taxa de Fraude", f"{fraud_rate:.2f}%")
            
            st.markdown("---")
            
            # Informações do modelo
            st.subheader("ℹ️ Informações do Modelo")
            st.write(f"**Status:** {'✅ Treinado' if model_info['trained'] else '❌ Não treinado'}")
            
            if model_info['last_training']:
                st.write(f"**Último Treinamento:** {model_info['last_training']}")
            
            if model_info['training_time']:
                st.write(f"**Tempo de Treinamento:** {model_info['training_time']:.2f}s")
        
    except Exception as e:
        st.error(f"Erro ao carregar estatísticas: {str(e)}")

# ===== PÁGINA SOBRE =====
elif page == "ℹ️ Sobre":
    st.header("ℹ️ Sobre o Projeto")
    
    st.markdown("""
    ### 🧠 Detecção de Fraude Neuromórfica
    
    Este sistema utiliza **Spiking Neural Networks (SNNs)** para detectar fraudes em transações financeiras 
    em tempo real, inspirado no funcionamento do cérebro humano.
    
    #### 🎯 Principais Características
    
    - **Ultra-baixa latência**: Detecção em ~10 milissegundos
    - **Aprendizado biológico**: STDP (Spike-Timing-Dependent Plasticity)
    - **Eficiência energética**: Processamento event-driven
    - **Temporal awareness**: Captura padrões temporais naturalmente
    
    #### 🔬 Tecnologias
    
    - **Brian2**: Simulador de redes neurais spiking
    - **FastAPI**: API REST de alta performance
    - **Streamlit**: Interface web interativa
    - **Docker**: Containerização para deploy consistente
    
    #### 📊 Arquitetura
    
    ```
    Input Layer (256) → Hidden (128) → Hidden (64) → Output (2)
    Total: 450 neurônios | 41,088 sinapses
    ```
    
    #### 👤 Autor
    
    **Mauro Risonho de Paula Assumpção**
    - LinkedIn: [maurorisonho](https://linkedin.com/in/maurorisonho)
    - GitHub: [@maurorisonho](https://github.com/maurorisonho)
    
    #### 📄 Licença
    
    MIT License - Projeto de demonstração educacional
    
    ---
    
    **Projeto 01** de 10 no Portfólio Neuromorphic X
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Neuromorphic Fraud Detection | Mauro Risonho | 2025"
    "</div>",
    unsafe_allow_html=True
)
