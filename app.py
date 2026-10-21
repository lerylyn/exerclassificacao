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

@st.cache_resource(show_spinner="Treinando modelo de IA...")
def load_data_and_train_model():
    # Geração de dados sintéticos (3 classes, 2 features)
    X, y = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_informative=2, 
        n_redundant=0, 
        n_classes=3, 
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Ajustando a escala das features para simular valores reais de usinagem
    peso = (X[:, 0] * 30) + 500       # Peso em gramas (média ~500g)
    espessura = (X[:, 1] * 3) + 20    # Espessura em mm (média ~20mm)
    
    # Reconstruindo X com os valores escalados
    X_scaled = np.column_stack((peso, espessura))
    
    # Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Treinamento do modelo SVM
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    
    # Gerando predições e a matriz de confusão
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Criando o DataFrame para exibição
    df = pd.DataFrame(X_scaled, columns=['Peso (gramas)', 'Espessura (mm)'])
    df['Classe_Num'] = y
    df['Status'] = df['Classe_Num'].map(CLASSES)
    
    return model, cm, df

# Carregamento do modelo e dados
model, cm, df_data = load_data_and_train_model()

# ==========================================
# SIDEBAR (Menu Lateral - UX/UI)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2043/2043064.png", width=60) # Ícone ilustrativo
    st.title("Parâmetros de Inspeção")
    st.write("Insira as medidas da peça para avaliação do modelo.")
    
    st.divider()
    
    peso_input = st.slider(
        "Peso da peça (g):", 
        min_value=float(df_data['Peso (gramas)'].min() - 20), 
        max_value=float(df_data['Peso (gramas)'].max() + 20), 
        value=500.0,
        help="Deslize para selecionar o peso da peça em gramas."
    )

    espessura_input = st.slider(
        "Espessura da peça (mm):", 
        min_value=float(df_data['Espessura (mm)'].min() - 2), 
        max_value=float(df_data['Espessura (mm)'].max() + 2), 
        value=20.0,
        help="Deslize para selecionar a espessura da peça em milímetros."
    )
    
    st.divider()
    avaliar_btn = st.button("🔍 Avaliar Peça", use_container_width=True, type="primary")

# ==========================================
# TELA PRINCIPAL
# ==========================================
st.title("⚙️ QualiUsinagem AI")
st.markdown("App web que utiliza IA para controle de qualidade em usinagem, classificando peças em tempo real como aprovadas, retrabalho ou refugo.")

# Espaço reservado para o resultado da avaliação
resultado_placeholder = st.container()

st.divider()

# Lógica de Avaliação (Disparada pelo botão no Sidebar)
if avaliar_btn:
    input_data = np.array([[peso_input, espessura_input]])
    prediction = model.predict(input_data)[0]
    resultado = CLASSES[prediction]
