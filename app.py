import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import predict_model
import tempfile

# Definir el path donde está el modelo entrenado
path = "/code/Python/Corte_2/Quiz_2_2/Punto_5/"

# Cargar el modelo preentrenado
with open(path + 'models/ridge_model.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Título de la aplicación
st.title("API de Predicción de Precios")

# Menú principal
menu = st.sidebar.selectbox("Selecciona una opción", ["Predicción Individual", "Predicción por Archivo"])

# Función para predecir un caso individual
def prediccion_individual():
    st.header("Predicción Individual")
    
    # Entradas del usuario
    input_data = {
        'Email': st.text_input('Correo electrónico', value='amandadean@gmail.com'),
        'Address': st.selectbox('Address', options=['Munich', 'Ausburgo', 'Berlin', 'Frankfurt'], index=0),
        'dominio': st.selectbox('Dominio', options=['Gmail', 'Hotmail', 'Yahoo', 'Otro'], index=0),
        'Tec': st.selectbox('Tecnología', options=['Smartphone', 'Portátil', 'PC', 'iPhone'], index=0),
        'Avg. Session Length': st.number_input('Duración promedio de la sesión', value=32.063775),
        'Time on App': st.number_input('Tiempo en la aplicación', value=10.71915),
        'Time on Website': st.number_input('Tiempo en el sitio web', value=37.712509),
        'Length of Membership': st.number_input('Duración de la membresía', value=3.004743),
    }

    # Botón para predecir
    if st.button("Predecir"):
        try:
            # Convertir las entradas en un DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Realizar la predicción
            yhat = modelo.predict(input_df)
            
            # Mostrar el resultado
            st.markdown(f"**Predicción:** {np.round(yhat[0], 2)}")

        except ValueError as e:
            st.error(f"Error en la entrada de datos: {str(e)}")

# Función para predecir con un archivo
def prediccion_por_archivo():
    st.header("Predicción por Archivo (Excel o CSV)")
    
    # Subir archivo
    uploaded_file = st.file_uploader("Cargar archivo Excel o CSV", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            # Crear un archivo temporal para manejar el archivo subido
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                tmp_path = temp_file.name

            # Leer el archivo subido
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(tmp_path)
            else:
                df = pd.read_excel(tmp_path)

            # Limpiar y convertir las columnas numéricas deseadas
            def clean_and_convert(column_name):
                df[column_name] = pd.to_numeric(df[column_name].astype(str).str.replace(',', '.'), errors='coerce')

            # Aplicar la limpieza de datos
            for column in ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']:
                clean_and_convert(column)

            # Realizar predicción usando el modelo
            predictions = predict_model(modelo, data=df)

            # Redondear las predicciones a 4 decimales y agregarlas al DataFrame original
            df['Predicciones'] = predictions["prediction_label"].round(4)

            # Mostrar el DataFrame con las predicciones
            st.write("Predicciones generadas correctamente:")
            st.write(df)

            # Preparar archivo para descarga
            if uploaded_file.name.endswith(".csv"):
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Descargar archivo con predicciones",
                                   data=csv_data,
                                   file_name="predicciones_con_resultados.csv",
                                   mime="text/csv")
            else:
                excel_data = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                df.to_excel(excel_data.name, index=False)
                st.download_button(label="Descargar archivo con predicciones",
                                   data=excel_data.read(),
                                   file_name="predicciones_con_resultados.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Lógica para mostrar la funcionalidad seleccionada en el menú
if menu == "Predicción Individual":
    prediccion_individual()
elif menu == "Predicción por Archivo":
    prediccion_por_archivo()

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    st.rerun()