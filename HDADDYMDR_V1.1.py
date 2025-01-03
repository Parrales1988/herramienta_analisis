import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from kaggle.api.kaggle_api_extended import KaggleApi

# Configuración inicial
st.title("Herramienta de análisis de datos y modelos de regresión")
st.subheader("Explora, analiza y aplica modelos de regresión a tus datos de manera sencilla")
st.sidebar.header("Menú")

# Función para cargar datasets desde Kaggle
def cargar_dataset():
    st.sidebar.subheader("Importar Dataset desde Kaggle")
    st.sidebar.write("Para obtener el enlace del dataset, accede a Kaggle, elige un dataset, y copia la URL desde el navegador.")
    st.sidebar.write("Ejemplo de enlace correcto: https://www.kaggle.com/dataset-owner/dataset-name")
    dataset_url = st.sidebar.text_input("Enlace del dataset (Kaggle)", "")
    kaggle_username = st.sidebar.text_input("Usuario de Kaggle", "")
    kaggle_key = st.sidebar.text_input("API Key de Kaggle", "", type="password")

    if st.sidebar.button("Cargar Dataset"):
        if not dataset_url or not kaggle_username or not kaggle_key:
            st.error("Por favor, complete todos los campos.")
        else:
            try:
                # Configuración de las credenciales de Kaggle
                os.environ['KAGGLE_USERNAME'] = kaggle_username
                os.environ['KAGGLE_KEY'] = kaggle_key
                
                # Configuración de las credenciales de Kaggle
                api = KaggleApi()
                api.authenticate()

                # Descarga del dataset
                dataset_info = dataset_url.split('/')[-1]
                api.dataset_download_files(dataset_info, path="datasets", unzip=True)
                st.success("Dataset descargado exitosamente.")

                # Lectura del archivo CSV
                dataset_path = f"datasets/{dataset_info.split('/')[-1]}.csv"
                data = pd.read_csv(dataset_path)
                return data
            except Exception as e:
                st.error(f"Error al descargar el dataset: {e}")

# Función para realizar EDA
def realizar_eda(data):
    st.subheader("Análisis Exploratorio de Datos (EDA)")
    st.write("En esta sección puedes explorar tus datos cargados. Aquí puedes visualizar las primeras filas, estadísticas descriptivas y relaciones entre variables.")

    st.write("Primeras filas del dataset:")
    st.dataframe(data.head())

    st.write("Información del dataset:")
    st.text(data.info())

    st.write("Estadísticas descriptivas:")
    st.write(data.describe())

    if st.checkbox("Mostrar correlación entre variables"):
        corr = data.corr()
        st.write("Matriz de correlación:")
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot()

# Función para realizar regresión
def aplicar_modelo_regresion(data):
    st.subheader("Aplicación de Modelo de Regresión")
    st.write("En esta sección puedes construir un modelo de regresión lineal para analizar tus datos.")
    st.write("Podrás observar los coeficientes del modelo, que indican la influencia de cada variable predictora en la variable objetivo.")
    st.write("También se calculan métricas como el Error Cuadrático Medio (MSE) para medir la precisión y el Coeficiente de Determinación (R^2) para evaluar la calidad del ajuste del modelo.")

    target = st.selectbox("Seleccione la variable objetivo (Y):", options=data.columns)
    features = st.multiselect("Seleccione las variables predictoras (X):", options=data.columns)

    if st.button("Ejecutar Regresión"):
        if not target or not features:
            st.error("Seleccione la variable objetivo y al menos una variable predictora.")
        else:
            X = data[features]
            y = data[target]

            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Modelo de regresión lineal
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)

            # Predicción
            y_pred = modelo.predict(X_test)

            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write("Resultados del Modelo:")
            st.write(f"Error Cuadrático Medio (MSE): {mse}")
            st.write(f"Coeficiente de Determinación (R^2): {r2}")

            st.write("Coeficientes del modelo:")
            coef_df = pd.DataFrame({"Variable": features, "Coeficiente": modelo.coef_})
            st.write(coef_df)

# Flujo principal de la aplicación
menu_opciones = ["Cargar Dataset", "EDA", "Regresión"]
opcion = st.sidebar.selectbox("Seleccione una opción", menu_opciones)

data = None
if opcion == "Cargar Dataset":
    data = cargar_dataset()
elif opcion == "EDA":
    if 'data' in st.session_state:
        realizar_eda(st.session_state['data'])
    else:
        st.error("Primero debe cargar un dataset.")
elif opcion == "Regresión":
    if 'data' in st.session_state:
        aplicar_modelo_regresion(st.session_state['data'])
    else:
        st.error("Primero debe cargar un dataset.")
