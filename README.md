# Titanic Survivability Prediction

Este proyecto implementa un **modelo de Machine Learning** para predecir la supervivencia de los pasajeros del Titanic y una **aplicación web interactiva** con Streamlit para explorar y probar el modelo.<br>[Prueba el modelo](https://herrera99-titanic-survivability-model-streamlit-appapp-eunll3.streamlit.app/)

## 📊 Funcionalidades

- Modelo de **Random Forest** entrenado con características relevantes:
  - Clase del billete (`Pclass`)
  - Sexo (`Sex`)
  - Edad (`Age`)
  - Número de hermanos/cónyuges (`SibSp`)
  - Número de padres/hijos (`Parch`)
  - Tarifa del billete (`Fare`)
  - Puerto de embarque (`Embarked`)
  - Tamaño de la familia (`FamilySize`)
- Predicción de supervivencia (0 = no sobrevive, 1 = sobrevive)
- Probabilidad estimada de supervivencia
- Visualizaciones exploratorias:
  - Supervivencia por sexo
  - Supervivencia por clase
  - Distribución de edad y tarifa
  - Probabilidad de supervivencia según tamaño de la familia
- Interfaz web con **Streamlit** que permite introducir datos de un pasajero y ver la predicción en tiempo real.

## 🛠 Tecnologías

- Python 3.10+
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [joblib](https://joblib.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [Streamlit](https://streamlit.io/)

## 🚀 Cómo ejecutar la app

1. Clonar el repositorio:

```bash
git clone https://github.com/TU_USUARIO/titanic-survivability-model.git
cd titanic-survivability-model/streamlit_app
