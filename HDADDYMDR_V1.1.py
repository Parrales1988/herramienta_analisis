import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
import json
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException
from fpdf import FPDF
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
                # Configuración de las credenciales de Kaggle
                os.environ['KAGGLE_USERNAME'] = kaggle_username
                os.environ['KAGGLE_KEY'] = kaggle_key

                # Configuración de la API de Kaggle
                api = KaggleApi()
                api.authenticate()

                st.success("Conexión exitosa a Kaggle.")

                # Descarga el dataset
                dataset_info = "/".join(dataset_url.split('/')[-2:])
                with st.spinner('Descargando dataset...'):
                    progress_bar = st.progress(0)
                    api.dataset_download_files(dataset_info, path=".", unzip=True)
                    progress_bar.progress(100)
                st.success("Dataset descargado exitosamente.")
                
                # Mostrar lista de archivos descargados para verificar el nombre correcto
                st.write("Archivos descargados:")
                for file_name in os.listdir("."):
                    if file_name.endswith(".csv"):
                        st.write(file_name)
                
                        # Intentar leer el archivo CSV
                        dataset_path = os.path.join(".", file_name)
                        if os.path.exists(dataset_path):
                            data = pd.read_csv(dataset_path)
                            st.session_state['data'] = data
                            st.session_state['data_loaded'] = True
                            st.success("Dataset cargado exitosamente.")
                            st.button("OK", key="kaggle_ok")
                            break
                else:
                    st.error("Archivo CSV no encontrado.")
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
                with st.spinner('Cargando dataset...'):
                    progress_bar = st.progress(0)
                    data = pd.read_csv(uploaded_file)
                    for percent_complete in range(100):
                        progress_bar.progress(percent_complete + 1)
                st.success("Dataset cargado exitosamente.")
                st.button("OK", key="csv_ok")
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

    if st.button("Volver al Menú Principal", key="eda_volver"):
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

                # Coeficientes del modelo
                coef_df = pd.DataFrame({"Variable": features, "Coeficiente": modelo.coef_})
                
                # Guardar resultados en session_state
                st.session_state['mse'] = mse
                st.session_state['r2'] = r2
                st.session_state['coef_df'] = coef_df

                # Mostrar resultados
                st.subheader("Resultados del Modelo de Regresión Lineal")
                st.write("### Coeficientes del Modelo")
                st.write(coef_df)

                st.write("### Métricas del Modelo")
                st.write(f"Error Cuadrático Medio (MSE): {mse}")
                st.write(f"Coeficiente de Determinación (R^2): {r2}")

                st.write("### Predicciones")
                st.write(pd.DataFrame({"Real": y_test, "Predicción": y_pred}).head())
                
            except Exception as e:
                st.error(f"Error al entrenar o evaluar el modelo: {e}")

    if st.button("Volver al Menú Principal", key="regresion_volver"):
        if 'data' in st.session_state:
            del st.session_state['data']
        st.session_state['view'] = 'menu'

# Función para crear y exportar informe ejecutivo
def crear_informe_ejecutivo(data, results):
    # Verificar que el data y results sean válidos
    if data is None or results is None:
        st.error("Datos o resultados no disponibles para generar el informe.")
        return

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()

    # Título del informe
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Informe Ejecutivo del Análisis de Datos", ln=True, align='C')
    pdf.ln(10)

    # Primeras filas del dataset
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Primeras filas del dataset:", ln=True)
    pdf.ln(5)
    for i, row in data.head().iterrows():
        pdf.cell(200, 10, txt=str(row.values), ln=True)
    pdf.ln(10)

    # Información del dataset
    buffer = io.StringIO()
    data.info(buf=buffer)
    info = buffer.getvalue()
    pdf.cell(200, 10, txt="Información del dataset:", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=info[:1000])  # Limitar a 1000 caracteres para no exceder el formato A4
    pdf.ln(10)

    # Estadísticas descriptivas
    pdf.cell(200, 10, txt="Estadísticas descriptivas:", ln=True)
    pdf.ln(5)
    for col, val in data.describe().iterrows():
        pdf.multi_cell(0, 10, txt=f"{col}: {val.values}")
    pdf.ln(10)

    # Resultados del modelo de regresión
    if results:
        pdf.cell(200, 10, txt="Resultados del Modelo de Regresión:", ln=True)
        pdf.ln(5)
        for key, value in results.items():
            pdf.multi_cell(0, 10, txt=f"{key}: {value}")

        # Añadir gráfica de correlación
        fig, ax = plt.subplots(figsize=(8, 6))

        # Convertir columnas categóricas a numéricas para la gráfica de correlación
        data_numeric = data.copy()
        for column in data_numeric.select_dtypes(include=['object']).columns:
            data_numeric[column] = data_numeric[column].astype('category').cat.codes

        sns.heatmap(data_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        fig.tight_layout()

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        buf.seek(0)

        # Usar un archivo temporal seguro
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(buf.getbuffer())
            temp_file_path = temp_file.name

        # Añadir la imagen al PDF
        pdf.image(temp_file_path, x=None, y=None, w=180, h=0, type='PNG')

        # Eliminar el archivo temporal
        os.remove(temp_file_path)

    # Guardar el PDF en un archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        pdf.output(temp_file.name)
        pdf_filename = temp_file.name

    # Leer el archivo temporal y guardarlo en un buffer de bytes
    with open(pdf_filename, "rb") as f:
        pdf_buffer = io.BytesIO(f.read())

    # Eliminar el archivo temporal
    os.remove(pdf_filename)

    st.success("Informe ejecutivo creado y listo para descargar.")
    
    # Mostrar el enlace de descarga en Streamlit
    download_button = st.download_button(
        label="Descargar Informe Ejecutivo",
        data=pdf_buffer,
        file_name="informe_ejecutivo.pdf",
        mime="application/pdf"
    )
    
    if download_button:
        if st.button("Volver al Menú Principal", key="informe_volver"):
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
        st.session_state['view'] = 'analisis'

if st.session_state['view'] == 'analisis' or st.session_state['view'] in ['eda', 'regresion', 'informe']:
    opciones = ["EDA", "Regresión", "Generar Informe Ejecutivo"]
    opcion = st.sidebar.selectbox("Seleccione una opción de análisis", opciones, key="main_option")

    if opcion == "EDA":
        st.session_state['view'] = 'eda'
    elif opcion == "Regresión":
        st.session_state['view'] = 'regresion'
    elif opcion == "Generar Informe Ejecutivo":
        st.session_state['view'] = 'informe'

if st.session_state['view'] == 'eda':
    realizar_eda(st.session_state['data'])
elif st.session_state['view'] == 'regresion':
    aplicar_modelo_regresion(st.session_state['data'])
elif st.session_state['view'] == 'informe':
    if st.button("Generar Informe Ejecutivo", key="generar_informe"):
        results = {
            "MSE": st.session_state.get("mse"),
            "R2": st.session_state.get("r2"),
            "Coeficientes": st.session_state.get("coef_df")
        }
        crear_informe_ejecutivo(st.session_state['data'], results)
    if st.button("Volver al Menú Principal", key="volver_menu"):
        if 'data' in st.session_state:
            del st.session_state['data']
        st.session_state['view'] = 'menu'
