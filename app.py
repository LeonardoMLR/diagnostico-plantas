import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from rembg import remove

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Análise de Folhas", page_icon="🌿", layout="centered")

# --- DICIONÁRIO DE TRADUÇÃO DAS DOENÇAS ---
CLASSES_DOENCAS = {
    0: "Mancha Bacteriana",
    1: "Pinta Preta (Alternariose)",
    2: "Requeima",
    3: "Mancha Bacteriana",
    4: "Pinta Preta (Alternariose)",
    5: "Requeima",
    6: "Bolor da Folha",
    7: "Mancha de Septoria",
    8: "Ácaro Rajado",
    9: "Mancha Alvo",
    10: "Vírus do Enrolamento Amarelo",
    11: "Vírus do Mosaico"
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
# Este bloco puxa os pesos treinados para a memória
@st.cache_resource
def load_system():
    m1_ident = tf.keras.models.load_model('modelo_01_identificacao_v2_mix.keras')
    m2_health = tf.keras.models.load_model('modelo_02_saudavel_ou_doente.keras')
    m4_disease = tf.keras.models.load_model('modelo_03_multiclasse.keras')
    return m1_ident, m2_health, m4_disease

# Tenta carregar os modelos. Se não achar os arquivos .keras, avisa na tela.
try:
    m_ident, m_health, m_disease = load_system()
except Exception as e:
    st.error(f"Erro ao carregar os ficheiros .keras. Verifique se eles estão na mesma pasta no GitHub! Erro: {e}")
    st.stop()

# --- ÁREA PRINCIPAL DO APP ---
st.markdown('<div class="main-card">', unsafe_allow_html=True)

arquivo = st.file_uploader("", type=["jpg", "jpeg", "png"])

if arquivo:
    # 1. Carrega a imagem original
    img_original = Image.open(arquivo)
    
    st.markdown('<div class="result-text-main">Tratando a imagem...</div>', unsafe_allow_html=True)
    
    with st.spinner("A IA está removendo o fundo da foto..."):
        # 2. Remove o fundo (deixa transparente)
        img_sem_fundo = remove(img_original)
        
        # 3. Cria um fundo branco limpo do exato tamanho da imagem
        fundo_branco = Image.new("RGBA", img_sem_fundo.size, "WHITE")
        
        # 4. Cola a folha recortada sobre o fundo branco e converte para RGB
        fundo_branco.paste(img_sem_fundo, (0, 0), img_sem_fundo)
        img_final = fundo_branco.convert('RGB')

    # Mostra a imagem tratada no site para o usuário ver
    st.image(img_final, caption="Imagem padronizada para análise", use_container_width=True)
    
    # Preparação da Imagem para o Keras usando a imagem tratada
    img_res = img_final.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_res)
    img_array = np.expand_dims(img_array, axis=0)

    # ==========================================
    # CASCATA PASSO 1: IDENTIFICAÇÃO (É Folha?)
    # ==========================================
    p_ident = m_ident.predict(img_array, verbose=0)[0][0]
    st.write(f"Probabilidade bruta (Passo 1): {p_ident:.4f}") 
    
    if p_ident >= 0.5:
        # NÃO É FOLHA
        st.markdown("""
            <div class="result-box error">
                <div class="result-icon">❌</div>
                <div>
                    <div class="result-text-main">Não é uma folha!</div>
                    <div class="result-text-sub">A imagem não contém uma folha reconhecível.</div>
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
        # CASCATA PASSO 2: SAÚDE (Binário Atual: Saudável ou Doente?)
        # ==========================================
        p_health = m_health.predict(img_array, verbose=0)[0][0]
        st.write(f"Probabilidade bruta (Passo 2): {p_health:.4f}") 
        
        if p_health >= 0.5:
            # SAUDÁVEL
            conf_h = p_health * 100
            st.markdown(f"""
                <div class="result-box">
                    <div class="result-icon">🌿</div>
                    <div>
                        <div class="result-text-main">Folha Saudável</div>
                        <div class="result-text-sub">Nenhuma anomalia detectada (Confiança: {conf_h:.1f}%).</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # DOENTE -> Continua para o passo 3
            conf_h = (1 - p_health) * 100 
            st.markdown(f"""
                <div class="result-box warning">
                    <div class="result-icon">⚠️</div>
                    <div>
                        <div class="result-text-main">Anomalia Detectada</div>
                        <div class="result-text-sub">Sinais de doença (Confiança: {conf_h:.1f}%). Iniciando diagnóstico específico...</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # ==========================================
            # CASCATA PASSO 3: DIAGNÓSTICO (Multiclasse Doenças)
            # ==========================================
            pred_disease = m_disease.predict(img_array, verbose=0)[0]
            indice_doenca = np.argmax(pred_disease)
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
