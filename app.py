import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# --- PARCHE AVANZADO DE COMPATIBILIDAD PARA SCIKIT-LEARN (Solución definitiva para '_loss') ---
try:
    # Intentar importar la estructura moderna de pérdidas
    import sklearn.ensemble._gb_losses as losses
    sys.modules['sklearn.ensemble.losses'] = losses
except ImportError:
    try:
        # Para versiones donde '_loss' cambió de ubicación interna
        import sklearn.ensemble._parameter_validation as pv
        # Crear un módulo simulado en sys.modules para interceptar el unpickling
        import types
        dummy_loss = types.ModuleType("sklearn.ensemble._loss")
        # Inyectar clases comunes que el modelo de boosting busca durante la carga
        try:
            from sklearn.ensemble._loss import HalfBinomialLoss, BinomialLoss
            dummy_loss.HalfBinomialLoss = HalfBinomialLoss
            dummy_loss.BinomialLoss = BinomialLoss
        except ImportError:
            pass
        sys.modules['sklearn.ensemble._loss'] = dummy_loss
    except Exception:
        pass

# Configuración de la página de Streamlit
st.set_page_config(page_title="Predicción de Rendimiento Estudiantil", layout="wide")

st.title("🎯 Aplicación Predictiva: Rendimiento Estudiantil")
st.markdown("Esta aplicación procesa datos de estudiantes y predice si el alumno aprobará (**Pass**) o reprobará (**Fail**) utilizando un modelo optimizado de Boosting.")

# --- CARGA DE ARCHIVOS / MODELOS (Rutas dinámicas locales del repositorio) ---
@st.cache_resource
def cargar_recursos():
    # Obtener el directorio donde se encuentra este archivo script (app.py)
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas relativas locales exclusivas para el despliegue en GitHub
    modelo_path = os.path.join(dir_actual, 'optimized_boosting_model.joblib')
    scaler_path = os.path.join(dir_actual, 'min_max_scaler.joblib')
    
    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)
    return modelo, scaler

try:
    modelo_boosting, scaler = cargar_recursos()
    st.success("¡Modelo y escalador cargados correctamente desde el repositorio!")
except Exception as e:
    st.error(f"Error al cargar los recursos: {e}. Asegúrate de subir 'optimized_boosting_model.joblib' y 'min_max_scaler.joblib' en la misma carpeta que 'app.py' en tu repositorio de GitHub.")
    st.stop()

# --- FORMULARIO DE ENTRADA PARA UN NUEVO REGISTRO ---
st.header("📝 Ingresar Datos del Estudiante")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Edad (Age)", min_value=15.0, max_value=25.0, value=17.0, step=1.0)
    study_time_hours_week = st.number_input("Horas de estudio a la semana (StudyTime_hours_week)", min_value=0.0, max_value=40.0, value=12.0, step=0.5)
    failures = st.slider("Número de reprobaciones previas (Failures)", min_value=0, max_value=4, value=0)
    absences = st.number_input("Ausencias (Absences)", min_value=0.0, max_value=100.0, value=2.0, step=1.0)

with col2:
    internet = st.selectbox("¿Tiene acceso a Internet? (Internet)", options=["yes", "no"], index=0)
    free_time = st.slider("Tiempo libre después de clases (FreeTime: 1-Muy bajo, 5-Muy alto)", min_value=1, max_value=5, value=3)
    go_out = st.slider("Salidas con amigos (GoOut: 1-Muy poco, 5-Mucho)", min_value=1, max_value=5, value=2)
    health = st.slider("Estado de salud (Health: 1-Muy malo, 5-Muy bueno)", min_value=1.0, max_value=5.0, value=4.0, step=1.0)

with col3:
    mother_education = st.selectbox("Educación de la madre (MotherEducation: 0-Ninguna a 4-Superior)", options=[0, 1, 2, 3, 4], index=2)
    father_education = st.selectbox("Educación del padre (FatherEducation: 0-Ninguna a 4-Superior)", options=[0, 1, 2, 3, 4], index=2)
    travel_time = st.slider("Tiempo de viaje a la escuela (TravelTime: 1 a 4)", min_value=1, max_value=4, value=1)

# --- PROCESAMIENTO Y PREDICCIÓN ---
if st.button("🔮 Predecir Rendimiento"):
    # 1. Crear DataFrame con las variables ingresadas
    datos_entrada = pd.DataFrame([{
        'Age': age,
        'StudyTime_hours_week': study_time_hours_week,
        'Failures': failures,
        'Absences': absences,
        'Internet': 1 if internet == "yes" else 0,
        'FreeTime': free_time,
        'GoOut': go_out,
        'Health': health,
        'MotherEducation': mother_education,
        'FatherEducation': father_education,
        'TravelTime': travel_time
    }])

    # 2. Normalizar las variables usando el escalador entrenado
    columnas_modelo = [
        'Age', 'StudyTime_hours_week', 'Failures', 'Absences',
        'Internet', 'FreeTime', 'GoOut', 'Health',
        'MotherEducation', 'FatherEducation', 'TravelTime'
    ]

    try:
        datos_entrada_normalizados = datos_entrada.copy()
        datos_entrada_normalizados[columnas_modelo] = scaler.transform(datos_entrada[columnas_modelo])

        # 3. Realizar predicción
        prediccion = modelo_boosting.predict(datos_entrada_normalizados[columnas_modelo])[0]

        # 4. Mostrar resultado decorado
        st.subheader("📊 Resultado de la Predicción:")
        if prediccion == 1:
            st.success("🎉 **Aprobado (Pass / 1)**: Basado en los hábitos e historial del estudiante, se predice que aprobará de forma satisfactoria.")
        else:
            st.error("⚠️ **Reprobado (Fail / 0)**: Se detectaron factores de riesgo de reprobación. Es recomendable aplicar un plan de acompañamiento académico.")

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento de la predicción: {e}")
