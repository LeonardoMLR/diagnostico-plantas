import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Análise de Folhas", page_icon="🌿", layout="centered")

# --- CSS PARA VISUAL CLEAN (Verde e Branco) ---
st.markdown("""
    <style>
    /* Fundo verde bem clarinho */
    .stApp {
        background-color: #e8f5e9;
    }
    
    /* Cabeçalho */
    .main-header {
        text-align: center;
        padding: 30px 0 20px 0;
    }
    .main-header h1 {
        color: #003300;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .main-header p {
        color: #4caf50;
        font-size: 1.1rem;
    }

    /* Ícone no topo */
    .icon-header {
        background-color: #00b300;
        color: white;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        margin-bottom: 10px;
    }

    /* Card principal (Branco com bordas arredondadas) */
    .main-card {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Cards de Resultado */
    .result-box {
        background-color: #f1f8f1;
        border: 1px solid #c8e6c9;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        display: flex;
        align-items: center;
    }
    .result-box.error {
        background-color: #ffebee;
        border-color: #ffcdd2;
    }
    .result-box.warning {
        background-color: #fff8e1;
        border-color: #ffecb3;
    }
    
    .result-icon {
        font-size: 24px;
        margin-right: 15px;
    }

    .result-text-main {
        font-weight: bold;
        color: #1b5e20;
        font-size: 1.1rem;
        margin-bottom: 2px;
    }
    .result-box.error .result-text-main { color: #b71c1c; }
    .result-box.warning .result-text-main { color: #f57f17; }
    
    .result-text-sub {
        color: #2e7d32;
        font-size: 0.9rem;
    }

    /* Botão Principal Verde */
    .stButton>button {
        width: 100%;
        background-color: #00b300;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px;
        font-weight: bold;
        font-size: 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #008000;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- CABEÇALHO ---
st.markdown("""
    <div class="main-header">
        <div class="icon-header">🍃</div>
        <h1>Análise de Folhas</h1>
        <p>Faça upload de uma imagem para verificar se é uma folha e seu estado de saúde</p>
    </div>
""", unsafe_allow_html=True)

# --- CARREGAMENTO DOS MODELOS ---
@st.cache_resource
def load_system():
    # Carrega os ficheiros exatos que você tem na pasta
    ident_model = tf.keras.models.load_model('ensemble_identificacao_final.keras')
    health_model = tf.keras.models.load_model('modelo_saude_balanceado.keras')
    return ident_model, health_model

try:
    m_ident, m_health = load_system()
except Exception as e:
    st.error(f"Erro ao carregar os ficheiros .keras. Verifique os nomes na pasta! Erro: {e}")
    st.stop()

# --- ENVOLVENDO A INTERFACE NUM "CARD BRANCO" ---
st.markdown('<div class="main-card">', unsafe_allow_html=True)

arquivo = st.file_uploader("", type=["jpg", "jpeg", "png"])

if arquivo:
    img = Image.open(arquivo)
    # Mostra a imagem com cantos arredondados
    st.image(img, use_container_width=True, output_format="PNG")
    
    # Preparação da Imagem (Apenas redimensionar. O modelo já faz o Rescaling interno)
    img_res = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_res)
    img_array = np.expand_dims(img_array, axis=0)

    # 1. Teste de Identificação
    p_ident = m_ident.predict(img_array, verbose=0)[0][0]
    
    if p_ident >= 0.5:
        # NÃO É FOLHA
        st.markdown("""
            <div class="result-box error">
                <div class="result-icon">❌</div>
                <div>
                    <div class="result-text-main">Não é uma folha!</div>
                    <div class="result-text-sub">A imagem não contém uma folha ou o fundo prejudicou a análise.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # É FOLHA
        conf_ident = (1 - p_ident) * 100
        st.markdown(f"""
            <div class="result-box">
                <div class="result-icon">✔️</div>
                <div>
                    <div class="result-text-main">É uma folha!</div>
                    <div class="result-text-sub">Confiança: {conf_ident:.1f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. Teste de Saúde
        p_health = m_health.predict(img_array, verbose=0)[0][0]
        
        if p_health < 0.5:
            # SAUDÁVEL
            st.markdown("""
                <div class="result-box">
                    <div class="result-icon">✔️</div>
                    <div>
                        <div class="result-text-main">Folha Saudável</div>
                        <div class="result-text-sub">Nenhuma anomalia detetada.</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # DOENTE
            conf_h = p_health * 100
            st.markdown(f"""
                <div class="result-box warning">
                    <div class="result-icon">⚠️</div>
                    <div>
                        <div class="result-text-main">Atenção: Anomalia Detetada</div>
                        <div class="result-text-sub">A folha apresenta sinais de doença (Confiança: {conf_h:.1f}%)</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.write("") # Espaçamento
    if st.button("Analisar Outra Imagem"):
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True) # Fim do Card Branco