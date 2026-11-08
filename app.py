import streamlit as st
from PIL import Image
import random
import time

# Configuración de la página
st.set_page_config(
    page_title="Detector de Placas Clonadas",
    page_icon="🚓",
    layout="wide"
)

# Título y Encabezado
st.title("🚓 Detector de Placas Clonadas — Apoyo Operativo")
st.caption("Sistema de asistencia visual en terreno para la detección de discrepancias con el RUNT.")

st.markdown("---")

# Layout de dos columnas principales
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Captura de Datos en Terreno")
    
    # Campo para la placa del RUNT
    placa = st.text_input("Placa del vehículo (Consulta RUNT)", placeholder="Ej: ABC123").strip().upper()
    
    # Entrada de foto
    opcion_foto = st.radio("Cargar fotografía por:", ("Subir imagen", "Tomar foto con la cámara"))
    
    imagen = None
    if opcion_foto == "Subir imagen":
        imagen_file = st.file_uploader("Seleccionar imagen del vehículo", type=["jpg", "jpeg", "png"])
        if imagen_file:
            imagen = Image.open(imagen_file)
    else:
        imagen_file = st.camera_input("Capturar foto en vía pública")
        if imagen_file:
            imagen = Image.open(imagen_file)
            
    if imagen:
        st.image(imagen, caption="Fotografía capturada", use_container_width=True)

    btn_analizar = st.button("🔍 Analizar Vehículo", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Resultado de Clasificación e Inspección")
    
    if btn_analizar:
        if not placa:
            st.warning("⚠️ Por favor, ingrese la placa del vehículo.")
        elif imagen is None:
            st.warning("⚠️ Por favor, tome o suba una foto del vehículo.")
        else:
            with st.spinner("Procesando imagen con modelo local ResNet-50 (~141 ms)..."):
                time.sleep(1) # Simulación de inferencia
                
                # Ejemplo de predicción (aquí conectarás el modelo PyTorch real)
                marcas_disponibles = ["Chevrolet Aveo 2011", "Renault Logan 2012", "Mazda 3 Sedan 2010", "Toyota Corolla 2012", "Hyundai Tucson 2012"]
                modelo_predicho = random.choice(marcas_disponibles)
                confianza = round(random.uniform(0.70, 0.98), 2)
                
                # Simulación de datos oficiales devueltos por el RUNT
                runt_marca = modelo_predicho if random.random() > 0.3 else "Chevrolet Spark 2015 (Registro RUNT)"

            # Tarjetas visuales de resultados
            st.markdown("### 🤖 Predicción del Modelo Visual")
            st.info(f"**Modelo Identificado:** {modelo_predicho}\n\n**Confianza del modelo:** {confianza * 100:.1f}%")
            
            st.markdown("### 📋 Registro Oficial en RUNT")
            st.write(f"**Registrado en RUNT:** {runt_marca}")
            
            st.markdown("---")
            st.markdown("### 🚦 Diagnóstico Operativo")
            
            # Evaluación de alertas según los criterios del proyecto
            if confianza < 0.20:
                st.warning("⚠️ **Confianza Baja (< 20%):** La resolución o el ángulo requieren una **verificación manual**.")
            elif runt_marca == modelo_predicho:
                st.success("✅ **COINCIDENCIA:** El vehículo fotografiado coincide visualmente con los datos del RUNT.")
            else:
                st.error("🚨 **ALERTA DE DISCREPANCIA:** El vehículo fotografiado NO coincide con la marca/modelo del RUNT.")
                st.info("💡 **Procedimiento:** Sugerir **verificación física** de chasis/motor. *(Herramienta de apoyo, nunca una acusación directa)*.")
    else:
        st.info("Ingrese la placa y cargue la foto para iniciar la validación.")

# Pie de página sobre privacidad y arquitectura
st.markdown("---")
with st.expander("ℹ️ Detalles del Modelo e Infraestructura"):
    st.markdown("""
    * **Modelo:** ResNet-50 fine-tuning (89.2% exactitud, 94.7% exactitud por marca, 98.0% top-5).
    * **Inferencia local:** Ejecución optimizada en CPU (~141 ms por foto).
    * **Privacidad:** No envía fotos a servicios externos de pago o nubes públicas no autorizadas.
    * **Limitación:** Entrenado con conjunto cerrado de 196 clases (EE. UU. hasta 2012).
    """)