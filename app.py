import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from rembg import remove # Importação da ferramenta de remoção de fundo

# ... (Todo o seu cabeçalho, CSS e dicionário de classes continuam aqui) ...

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
        
        # 4. Cola a folha recortada sobre o fundo branco e converte para RGB seguro
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
