import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load('model/modelo_titanic.pkl')


try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"Error crítico al cargar el modelo: {e}")
    st.info("Pista: Si el error dice 'No module named sklearn', es que falta instalar scikit-learn en este entorno.")
    st.stop()

st.title("Predicción de Supervivencia del Titanic")
st.write("Introduce los datos del pasajero para saber si la IA cree que sobreviviría.")
st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", caption="El insumergible",
         use_container_width=True)

st.sidebar.header("Datos del Pasajero")

pclass = st.sidebar.selectbox("Clase del Billete", [1, 2, 3], format_func=lambda x: f"{x}ª Clase")

sex_input = st.sidebar.radio("Sexo", ["Hombre", "Mujer"])
sex = 0 if sex_input == "Hombre" else 1

age = st.sidebar.slider("Edad", 0, 100, 30)

st.sidebar.subheader("Familia a bordo")
sibsp = st.sidebar.number_input("Hermanos / Cónyuges", 0, 10, 0)
parch = st.sidebar.number_input("Padres / Hijos", 0, 10, 0)

fare = st.sidebar.slider("Precio del Billete ($)", 0, 550, 32)

embarked_input = st.sidebar.selectbox("Puerto de Embarque", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
if "Southampton" in embarked_input:
    embarked = 0
elif "Cherbourg" in embarked_input:
    embarked = 1
else:
    embarked = 2

family_size = sibsp + parch + 1

# Creamos un DataFrame con LAS MISMAS COLUMNAS y EN EL MISMO ORDEN que en el entrenamiento
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked],
    'FamilySize': [family_size]
})

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    calcular = st.button("Calcular Supervivencia", type="primary", use_container_width=True)

if calcular:

    # Predecir clase (0 o 1)
    prediction = model.predict(input_data)[0]
    # Predecir probabilidad (ej. 75%)
    probability = model.predict_proba(input_data)[0][1]  # Cogemos la probabilidad de ser 1 (vivir)

    st.divider()

    if prediction == 1:
        st.balloons()
        st.success(f"**¡SOBREVIVE!** La IA estima una probabilidad de supervivencia del **{probability:.1%}**.")
        st.write("Este pasajero habría conseguido un bote salvavidas.")
    else:
        st.error(f"**NO SOBREVIVE.** La IA estima una probabilidad de supervivencia de solo **{probability:.1%}**.")
        st.write("Este pasajero lamentablemente se habría hundido con el barco.")

    # Mostramos los datos técnicos si el usuario quiere verlos
    with st.expander("Ver datos técnicos enviados al modelo"):
        st.write(input_data)
