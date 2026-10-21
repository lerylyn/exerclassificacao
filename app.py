import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Configuração inicial da página (deve ser a primeira chamada)
st.set_page_config(
    page_title="QualiUsinagem AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mapeamento das classes
CLASSES = {
    0: 'Aprovada',
    1: 'Retrabalho',
    2: 'Refugo'
}

# Inicializando a memória da sessão para manter o resultado na tela
if 'resultado_atual' not in st.session_state:
    st.session_state.resultado_atual = None
    st.session_state.peso_analisado = None
    st.session_state.espessura_analisada = None

@st.cache_resource(show_spinner="Treinando modelo de IA...")
def load_data_and_train_model():
    X, y = make_classification(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0, 
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    
    # Escala simulando valores reais
    peso = (X[:, 0] * 30) + 500       
    espessura = (X[:, 1] * 3) + 20    
    X_scaled = np.column_stack((peso, espessura))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    df = pd.DataFrame(X_scaled, columns=['Peso (gramas)', 'Espessura (mm)'])
    df['Classe_Num'] = y
    df['Status'] = df['Classe_Num'].map(CLASSES)
    
    return model, cm, df

# Carrega modelo e dados
model, cm, df_data = load_data_and_train_model()

# Função que executa a predição e salva na memória do Streamlit
def avaliar_peca(peso, espessura):
    input_data = np.array([[peso, espessura]])
    prediction = model.predict(input_data)[0]
    st.session_state.resultado_atual = CLASSES[prediction]
    st.session_state.peso_analisado = peso
    st.session_state.espessura_analisada = espessura

# ==========================================
# SIDEBAR (Menu Lateral - UX/UI)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2043/2043064.png", width=60)
    st.title("Parâmetros de Inspeção")
    st.write("Insira as medidas da peça para avaliação do modelo.")
    st.divider()
    
    peso_input = st.slider(
        "Peso da peça (g):", 
        min_value=float(df_data['Peso (gramas)'].min() - 20), 
        max_value=float(df_data['Peso (gramas)'].max() + 20), 
        value=500.0
    )

    espessura_input = st.slider(
        "Espessura da peça (mm):", 
        min_value=float(df_data['Espessura (mm)'].min() - 2), 
        max_value=float(df_data['Espessura (mm)'].max() + 2), 
        value=20.0
    )
    
    st.divider()
    # O botão agora chama a função que salva o estado, em vez de apenas retornar True/False
    st.button("🔍 Avaliar Peça", on_click=avaliar_peca, args=(peso_input, espessura_input), use_container_width=True, type="primary")

# ==========================================
# TELA PRINCIPAL
# ==========================================
st.title("⚙️ QualiUsinagem AI")
st.markdown("App web que utiliza IA para controle de qualidade em usinagem, classificando peças em tempo real como aprovadas, retrabalho ou refugo.")

st.divider()

# Exibição do Resultado com base na memória de sessão (Não some mais!)
if st.session_state.resultado_atual is not None:
    resultado = st.session_state.resultado_atual
    peso_tela = st.session_state.peso_analisado
    espessura_tela = st.session_state.espessura_analisada

    st.subheader("Resultado da Inspeção")
    st.caption(f"Dados analisados ➔ Peso: **{peso_tela:.2f}g** | Espessura: **{espessura_tela:.2f}mm**")
    
    if resultado == 'Aprovada':
        st.success(f"### ✅ STATUS: {resultado.upper()}\n A peça está dentro dos padrões de qualidade.")
    elif resultado == 'Retrabalho':
        st.warning(f"### ⚠️ STATUS: {resultado.upper()}\n A peça apresenta desvios e necessita de ajustes.")
    else:
        st.error(f"### ❌ STATUS: {resultado.upper()}\n A peça está fora das tolerâncias e deve ser descartada.")
else:
    # Mensagem exibida antes do usuário interagir pela primeira vez
    st.info("👈 Ajuste os parâmetros no menu lateral e clique em **Avaliar Peça** para inspecionar.")

st.divider()

# ==========================================
# SEÇÃO DE ANÁLISE DE DADOS
# ==========================================
st.subheader("📊 Análise de Desempenho do Modelo")
st.write("Visualização da Matriz de Confusão do modelo treinado.")

col1, col2 = st.columns([1, 1])

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES.values(), 
                yticklabels=CLASSES.values(), ax=ax,
                linewidths=.5, cbar=False)
    ax.set_ylabel('Classe Real', fontweight='bold')
    ax.set_xlabel('Classe Predita', fontweight='bold')
    fig.patch.set_facecolor('none') 
    ax.set_facecolor('none')
    st.pyplot(fig)

with col2:
    st.info("""
    **Como ler a Matriz:**
    - **Diagonal principal (azul escuro):** Acertos do modelo (ex: previu Aprovada e era Aprovada de fato).
    - **Outros quadros:** Erros de classificação. Quanto menores os números fora da diagonal principal, melhor a inteligência artificial.
    """)

st.divider()

st.subheader("📋 Base de Dados Sintética")
with st.expander("Visualizar Tabela de Dados Completa", expanded=False):
    st.dataframe(
        df_data[['Peso (gramas)', 'Espessura (mm)', 'Status']].style.format({
            "Peso (gramas)": "{:.2f} g",
            "Espessura (mm)": "{:.2f} mm"
        }),
        use_container_width=True,
        height=300
    )
