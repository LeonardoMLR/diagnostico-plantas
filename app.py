import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuração da página - Tema escuro para destaque
st.set_page_config(page_title="Fresh Prince of Plants", page_icon="🌿", layout="wide")

# --- ESTILIZAÇÃO CSS ---
st.markdown("""
    <style>
    /* Estilo do aviso de fundo neutro */
    .aviso-caixa {
        background-color: #ff4b4b1a; 
        border: 1px solid #ff4b4b;
        padding: 15px;
        border-radius: 10px;
        color: #ff4b4b;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Estilo do balão de fala dinâmico */
    .balao-fala {
        position: relative;
        background: #262730; 
        border: 3px solid #2ecc71; 
        border-radius: 20px;
        padding: 20px;
        color: white;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    /* AJUSTE AQUI: A pontinha do balão apontando para a esquerda (cabeça do Will) */
    .balao-fala:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 25%; /* Move a ponta para a esquerda */
        width: 0;
        height: 0;
        border: 20px solid transparent;
        border-top-color: #2ecc71; 
        border-bottom: 0;
        margin-left: -20px;
        margin-bottom: -20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Diagnóstico Inteligente com Will Smith")

# Aviso Fundo Neutro
st.markdown('<div class="aviso-caixa">⚠️ AVISO: Fundo Neutro! Para melhores resultados, use uma foto de UMA ÚNICA FOLHA com um FUNDO LISO (sem nada atrás).</div>', unsafe_allow_html=True)

TAMANHO_RESIZE = (224, 224)

@st.cache_resource
def carregar_modelos():
    m_identificacao = tf.keras.models.load_model('modelo_identificacao_v1.keras')
    m_saude = tf.keras.models.load_model('modelo_saude_balanceado.keras')
    return m_identificacao, m_saude

try:
    m_ident, m_saud = carregar_modelos()
except Exception as e:
    st.error("Erro ao carregar modelos. Certifique-se que os arquivos .keras estão na pasta.")
    st.stop()

# Área de Upload
arquivo_subido = st.file_uploader("Escolha a imagem da folha...", type=["png", "jpg", "jpeg", "webp", "jfif"])

if arquivo_subido is not None:
    imagem = Image.open(arquivo_subido)
    imagem_redimensionada = imagem.resize(TAMANHO_RESIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(imagem_redimensionada)
    img_array = np.expand_dims(img_array, axis=0)

    # Predições
    prev_ident = m_ident.predict(img_array, verbose=0)[0][0]
    
    texto_will = ""
    titulo_diagnostico = ""
    cor_balao = "#2ecc71" 

    if prev_ident >= 0.5:
        texto_will = "Rapaz... isso aí não é uma planta não ou não está em um fundo liso! 🤨"
        titulo_diagnostico = "⚠️ ERRO DE CONTEXTO"
        cor_balao = "#ff4b4b" 
    else:
        prev_saude = m_saud.predict(img_array, verbose=0)[0][0]
        if prev_saude < 0.5:
            texto_will = "Olha que planta bonita! Está supimpa e saudável! "
            titulo_diagnostico = "✅ DIAGNÓSTICO: PLANTA SAUDÁVEL"
            cor_balao = "#2ecc71" 
        else:
            texto_will = "Certamente é uma folha, mas é melhor cuidar, se não ela vai com Deus."
            titulo_diagnostico = " DIAGNÓSTICO: ANOMALIA DETECTADA (DOENTE)"
            cor_balao = "#f1c40f" 

    # --- LAYOUT LADO A LADO ---
    col1, col2 = st.columns([1, 1], gap="medium") 

    with col1:
        # Título encima do Will
        st.subheader(titulo_diagnostico)
        
        # Balão de fala
        st.markdown(f"""
            <div class="balao-fala" style="border-color: {cor_balao};">
                {texto_will}
            </div>
            <style>
                .balao-fala:after {{ border-top-color: {cor_balao}; }}
            </style>
        """, unsafe_allow_html=True)
        
        # Lê a imagem sem fundo do Will
        st.image("will_apresentador-removebg-preview.png", width=350)

    with col2:
        # Bloco invisível para empurrar a folha para baixo e alinhar com o Will
        st.markdown("<div style='height: 95px;'></div>", unsafe_allow_html=True)
        st.write("### Imagem Analisada")
        st.image(imagem, use_container_width=True)