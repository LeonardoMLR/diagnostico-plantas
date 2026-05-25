import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Análise de Folhas", page_icon="🌿", layout="centered")

# --- DICIONÁRIO DE TRADUÇÃO DAS DOENÇAS (Modelo 04) ---
# A ordem deve ser estritamente alfabética, correspondendo ao class_indices do gerador
CLASSES_DOENCAS = {
    0: "Pimentão - Mancha Bacteriana",
    1: "Batata - Pinta Preta (Alternariose)",
    2: "Batata - Requeima",
    3: "Tomate - Mancha Bacteriana",
    4: "Tomate - Pinta Preta (Alternariose)",
    5: "Tomate - Requeima",
    6: "Tomate - Bolor da Folha",
    7: "Tomate - Mancha de Septoria",
    8: "Tomate - Ácaro Rajado",
    9: "Tomate - Mancha Alvo",
    10: "Tomate - Vírus do Enrolamento Amarelo",
    11: "Tomate - Vírus do Mosaico"
}

# --- CSS PARA VISUAL CLEAN (Verde e Branco) ---
st.markdown("""
    <style>
    .stApp { background-color: #e8f5e9; }
    .main-header { text-align: center; padding: 30px 0 20px 0; }
    .main-header h1 { color: #003300; font-size: 2.2rem; font-weight: 800; margin-bottom: 5px; }
    .main-header p { color: #4caf50; font-size: 1.1rem; }
    .icon-header {
        background-color: #00b300; color: white; border-radius: 50%;
        width: 60px; height: 60px; display: inline-flex;
        align-items: center; justify-content: center; font-size: 30px; margin-bottom: 10px;
    }
    .main-card {
        background-color: white; border-radius: 15px; padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;
    }
    .result-box {
        background-color: #f1f8f1; border: 1px solid #c8e6c9; border-radius: 10px;
        padding: 15px; margin-top: 15px; display: flex; align-items: center;
    }
    .result-box.error { background-color: #ffebee; border-color: #ffcdd2; }
    .result-box.warning { background-color: #fff8e1; border-color: #ffecb3; }
    .result-box.danger { background-color: #fce4ec; border-color: #f8bbd0; }
    .result-icon { font-size: 24px; margin-right: 15px; }
    .result-text-main { font-weight: bold; color: #1b5e20; font-size: 1.1rem; margin-bottom: 2px; }
    .result-box.error .result-text-main { color: #b71c1c; }
    .result-box.warning .result-text-main { color: #f57f17; }
    .result-box.danger .result-text-main { color: #880e4f; }
    .result-text-sub { color: #2e7d32; font-size: 0.9rem; }
    .stButton>button {
        width: 100%; background-color: #00b300; color: white; border-radius: 8px;
        border: none; padding: 12px; font-weight: bold; font-size: 1rem; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #008000; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- CABEÇALHO ---
st.markdown("""
    <div class="main-header">
        <div class="icon-header">🍃</div>
        <h1>Análise de Folhas</h1>
        <p>Faça upload de uma imagem para diagnóstico completo via Inteligência Artificial</p>
    </div>
""", unsafe_allow_html=True)

# --- CARREGAMENTO DOS MODELOS EM CASCATA ---
@st.cache_resource
def load_system():
    # Atualizado com os nomes exatos do seu print de diretório
    m1_ident = tf.keras.models.load_model('modelo_01_planta_ou_nao.keras')
    m2_health = tf.keras.models.load_model('modelo_02_saudavel_ou_doente.keras')
    m4_disease = tf.keras.models.load_model('modelo_04_m7_profundo.keras')
    return m1_ident, m2_health, m4_disease

try:
    m_ident, m_health, m_disease = load_system()
except Exception as e:
    st.error(f"Erro ao carregar os ficheiros .keras. Verifique a pasta! Erro: {e}")
    st.stop()

st.markdown('<div class="main-card">', unsafe_allow_html=True)

arquivo = st.file_uploader("", type=["jpg", "jpeg", "png"])

if arquivo:
    img = Image.open(arquivo)
    st.image(img, use_container_width=True, output_format="PNG")
    
    # Preparação da Imagem
    img_res = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_res)
    img_array = np.expand_dims(img_array, axis=0)

    # ==========================================
    # CASCATA PASSO 1: IDENTIFICAÇÃO (É Folha?)
    # ==========================================
    p_ident = m_ident.predict(img_array, verbose=0)[0][0]
    
    if p_ident >= 0.5:
        # NÃO É FOLHA (Para o processamento aqui)
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
        # É FOLHA -> Continua para o passo 2
        conf_ident = (1 - p_ident) * 100
        st.markdown(f"""
            <div class="result-box">
                <div class="result-icon">✔️</div>
                <div>
                    <div class="result-text-main">Identificado: Folha</div>
                    <div class="result-text-sub">Confiança: {conf_ident:.1f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # ==========================================
        # CASCATA PASSO 2: SAÚDE (Saudável ou Doente?)
        # ==========================================
        p_health = m_health.predict(img_array, verbose=0)[0][0]
        
        if p_health < 0.5:
            # SAUDÁVEL (Para o processamento aqui)
            st.markdown("""
                <div class="result-box">
                    <div class="result-icon">🌿</div>
                    <div>
                        <div class="result-text-main">Folha Saudável</div>
                        <div class="result-text-sub">Nenhuma anomalia detectada.</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # DOENTE -> Continua para o passo 3
            conf_h = p_health * 100
            st.markdown(f"""
                <div class="result-box warning">
                    <div class="result-icon">⚠️</div>
                    <div>
                        <div class="result-text-main">Anomalia Detectada</div>
                        <div class="result-text-sub">A folha apresenta sinais de doença (Confiança: {conf_h:.1f}%). Iniciando diagnóstico específico...</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # ==========================================
            # CASCATA PASSO 3: DIAGNÓSTICO (Modelo 04)
            # ==========================================
            pred_disease = m_disease.predict(img_array, verbose=0)[0]
            indice_doenca = np.argmax(pred_disease) # Pega o índice da maior probabilidade
            conf_disease = pred_disease[indice_doenca] * 100
            
            nome_doenca = CLASSES_DOENCAS.get(indice_doenca, "Doença Desconhecida")

            st.markdown(f"""
                <div class="result-box danger">
                    <div class="result-icon">🔬</div>
                    <div>
                        <div class="result-text-main">Diagnóstico: {nome_doenca}</div>
                        <div class="result-text-sub">Precisão do Diagnóstico: {conf_disease:.1f}%</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.write("") 
    if st.button("Analisar Outra Imagem"):
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)