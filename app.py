import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Mapeamento das classes
CLASSES = {
    0: 'Aprovada',
    1: 'Retrabalho',
    2: 'Refugo'
}

@st.cache_resource
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
    X[:, 0] = (X[:, 0] * 30) + 500  # Peso em gramas (média ~500g)
    X[:, 1] = (X[:, 1] * 3) + 20    # Espessura em mm (média ~20mm)
    
    # Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinamento do modelo SVM
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    
    # Gerando predições e a matriz de confusão
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, cm, X

# Configuração da página
st.set_page_config(page_title="Controle de Qualidade", layout="centered")

# Carregamento do modelo e dados
model, cm, X_data = load_data_and_train_model()

st.title("⚙️ Controle de Qualidade de Usinagem")
st.write("Insira os dados da peça inspecionada para avaliação pelo modelo SVM.")

# Interface para entrada de dados
st.subheader("Parâmetros da Peça")
peso_input = st.slider("Peso (gramas)", 
                       min_value=float(np.min(X_data[:, 0]) - 50), 
                       max_value=float(np.max(X_data[:, 0]) + 50), 
                       value=500.0)

espessura_input = st.slider("Espessura (mm)", 
                            min_value=float(np.min(X_data[:, 1]) - 5), 
                            max_value=float(np.max(X_data[:, 1]) + 5), 
                            value=20.0)

# Botão de avaliação
if st.button("Avaliar Peça"):
    # Realizando a predição com os dados do usuário
    input_data = np.array([[peso_input, espessura_input]])
    prediction = model.predict(input_data)[0]
    resultado = CLASSES[prediction]
    
    # Exibição do resultado com cores correspondentes
    if resultado == 'Aprovada':
        st.success(f"**Resultado:** Peça {resultado} ✅")
    elif resultado == 'Retrabalho':
        st.warning(f"**Resultado:** Peça para {resultado} ⚠️")
    else:
        st.error(f"**Resultado:** Peça classificada como {resultado} ❌")

st.divider()

# Exibição da Matriz de Confusão
st.subheader("Desempenho do Modelo (Matriz de Confusão)")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASSES.values(), 
            yticklabels=CLASSES.values(), ax=ax)
ax.set_ylabel('Classe Real')
ax.set_xlabel('Classe Predita')
st.pyplot(fig)
