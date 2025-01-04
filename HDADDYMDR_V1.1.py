import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
import json
import kaggle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException

# Configuración inicial
st.title("Herramienta de análisis de datos y modelos de regresión")
st.subheader("Explora, analiza y aplica modelos de regresión a tus datos de manera sencilla")
st.sidebar.header("Menú")

# Función para cargar datasets desde Kaggle
def cargar_dataset_kaggle():
    st.sidebar.subheader("Importar Dataset desde Kaggle")
    st.sidebar.write("Ingrese sus credenciales de Kaggle:")
    kaggle_username = st.sidebar.text_input("Nombre de usuario de Kaggle", "")
    kaggle_key = st.sidebar.text_input("Clave API de Kaggle", "", type="password")
    
    st.sidebar.write("Para obtener el enlace del dataset, accede a Kaggle, elige un dataset, y copia la URL desde el navegador.")
    st.sidebar.write("Ejemplo de enlace correcto: https://www.kaggle.com/dataset-owner/dataset-name")
    dataset_url = st.sidebar.text_input("Enlace del dataset (Kaggle)", "")

    if st.sidebar.button("Cargar Dataset desde Kaggle"):
        if not kaggle_username or not kaggle_key or not dataset_url:
            st.error("Por favor, complete todos los campos.")
        else:
            try:
                # Crear la carpeta .kaggle si no existe
                kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
                if not os.path.exists(kaggle_dir):
                    os.makedirs(kaggle_dir)

                # Crear el archivo kaggle.json con las credenciales
                kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
                kaggle_json_content = {
                    "username": kaggle_username,
                    "key": kaggle_key
                }
                with open(kaggle_json_path, "w") as f:
                    json.dump(kaggle_json_content, f)

                # Establecer permisos correctos al archivo kaggle.json
                os.chmod(kaggle_json_path, 0o600)

                # Configuración de la API de Kaggle
                api = KaggleApi()
                api.authenticate()

                st.success("Conexión exitosa a Kaggle.")

                # Descarga el dataset
                #kaggle.api.dataset_download_files('https://www.kaggle.com/datasets/jackdaoud/marketing-data', path='.', unzip=True)

                # Descarga del dataset utilizando el identificador correcto
                dataset_info = "/".join(dataset_url.split('/')[-2:])
                api.dataset_download_files(dataset_info, path=".", unzip=True)
                st.success("Dataset descargado exitosamente.")
                
                # Mostrar lista de archivos descargados para verificar el nombre correcto
                st.write("Archivos descargados:")
                for file_name in os.listdir("."):
                    if file_name.endswith(".csv"):
                        st.write(file_name)
                
                # Intentar leer el archivo CSV
                dataset_path = f"./{dataset_info.split('/')[-1]}.csv"
                if os.path.exists(dataset_path):
                    data = pd.read_csv(dataset_path)
                    st.session_state['data'] = data
                    st.session_state['data_loaded'] = True
                else:
                    st.error(f"Archivo no encontrado: {dataset_path}")
            except ApiException as e:
                st.error(f"Error al descargar el dataset: {e}")
            except Exception as e:
                st.error(f"Error al descargar el dataset: {e}")

# Función para cargar datasets desde un archivo CSV
def cargar_dataset_csv():
    st.sidebar.subheader("Importar Dataset desde un archivo CSV")
    uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type="csv")

    if st.sidebar.button("Cargar Dataset desde CSV"):
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("Dataset cargado exitosamente.")
                st.session_state['data'] = data
                st.session_state['data_loaded'] = True
            except Exception as e:
                st.error(f"Error al cargar el dataset: {e}")
        else:
            st.error("Por favor, suba un archivo CSV.")

# Función para realizar EDA
def realizar_eda(data):
    st.subheader("Análisis Exploratorio de Datos (EDA)")
    st.write("En esta sección puedes explorar tus datos cargados. Aquí puedes visualizar las primeras filas, estadísticas descriptivas y relaciones entre variables.")

    st.write("Primeras filas del dataset:")
    st.dataframe(data.head())

    st.write("Información del dataset:")
    try:
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    except Exception as e:
        st.error(f"Error al obtener la información del dataset: {e}")

    st.write("Estadísticas descriptivas:")
    st.write(data.describe())

    selected_columns = st.multiselect("Seleccione las columnas para calcular la correlación:", options=data.columns)
    if st.checkbox("Mostrar correlación entre variables") and selected_columns:
        try:
            # Convertir columnas categóricas a numéricas
            data_numeric = data[selected_columns].copy()
            for column in data_numeric.select_dtypes(include=['object']).columns:
                data_numeric[column] = data_numeric[column].astype('category').cat.codes
            
            corr = data_numeric.corr()
            st.write("Matriz de correlación:")
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al mostrar la correlación: {e}")

    if st.button("Volver al Menú Principal"):
        if 'data' in st.session_state:
            del st.session_state['data']
        st.session_state['view'] = 'menu'

# Función para validar columnas para regresión
def validar_columnas_para_regresion(data):
    columnas_validas = []
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            if data[column].isnull().sum() == 0:
                columnas_validas.append(column)
    return columnas_validas

# Función para realizar regresión
def aplicar_modelo_regresion(data):
    st.subheader("Aplicación de Modelo de Regresión")
    st.write("En esta sección puedes construir un modelo de regresión lineal para analizar tus datos.")
    st.write("Podrás observar los coeficientes del modelo, que indican la influencia de cada variable predictora en la variable objetivo.")
    st.write("También se calculan métricas como el Error Cuadrático Medio (MSE) para medir la precisión y el Coeficiente de Determinación (R^2) para evaluar la calidad del ajuste del modelo.")
    
    columnas_validas = validar_columnas_para_regresion(data)
    st.write("Columnas válidas para regresión:")
    st.write(columnas_validas)

    target = st.selectbox("Seleccione la variable objetivo (Y):", options=columnas_validas, key="target")
    features = st.multiselect("Seleccione las variables predictoras (X):", options=columnas_validas, key="features")

    if st.button("Ejecutar Regresión"):
        if not target or not features:
            st.error("Seleccione la variable objetivo y al menos una variable predictora.")
        else:
            X = data[features]
            y = data[target]

            # Comprobar si hay valores nulos y eliminarlos
            if X.isnull().values.any() or y.isnull().values.any():
                st.warning("El dataset contiene valores nulos. Estos valores se evitaron para ejecutar la regresión.")
                X = X.dropna()
                y = y[X.index]

            # Comprobar si hay valores no numéricos y convertirlos
            try:
                X = X.apply(pd.to_numeric, errors='coerce')
                y = pd.to_numeric(y, errors='coerce')
                X = X.dropna()
                y = y[X.index]
            except Exception as e:
                st.error(f"Error al convertir datos a numéricos: {e}")
                return

            # Verificar si hay suficientes datos para dividir
            if len(X) < 2:
                st.error("No hay suficientes datos para dividir en conjuntos de entrenamiento y prueba. Seleccione diferentes columnas.")
                return

            # División de datos
            try:
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
            except Exception as e:
                st.error(f"Error al entrenar o evaluar el modelo: {e}")

    if st.button("Volver al Menú Principal"):
        if 'data' in st.session_state:
            del st.session_state['data']
        st.session_state['view'] = 'menu'

# Flujo principal de la aplicación
if 'view' not in st.session_state:
    st.session_state['view'] = 'menu'

if st.session_state['view'] == 'menu':
    if 'data' not in st.session_state:
        menu_opciones = ["Cargar Dataset Kaggle", "Cargar Dataset CSV"]
        opcion = st.sidebar.selectbox("Seleccione una opción", menu_opciones)

        if opcion == "Cargar Dataset Kaggle":
            cargar_dataset_kaggle()
        elif opcion == "Cargar Dataset CSV":
            cargar_dataset_csv()
    else:
        opciones = ["EDA", "Regresión"]
        opcion = st.sidebar.selectbox("Seleccione una opción de análisis", opciones, key="main_option")

        if opcion == "EDA":
            st.session_state['view'] = 'eda'
        elif opcion == "Regresión":
            st.session_state['view'] = 'regresion'

if st.session_state['view'] == 'eda':
    realizar_eda(st.session_state['data'])
elif st.session_state['view'] == 'regresion':
    aplicar_modelo_regresion(st.session_state['data'])

# Mantener el sidebar visible
if st.session_state['view'] in ['eda', 'regresion']:
    opciones = ["EDA", "Regresión"]
    opcion = st.sidebar.selectbox("Seleccione una opción de análisis", opciones, key="sidebar_option")
    if opcion == "EDA":
        st.session_state['view'] = 'eda'
    elif opcion == "Regresión":
        st.session_state['view'] = 'regresion'
