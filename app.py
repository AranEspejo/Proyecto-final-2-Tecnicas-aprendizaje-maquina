"""
Verificador de consistencia vehículo–RUNT
Proyecto de Aplicación 2 · Técnicas de Aprendizaje de Máquina · Pontificia Universidad Javeriana

Usa el MISMO modelo y el MISMO preprocesamiento del cuaderno:
  carros_utils.py  (preprocesamiento, arquitectura e inferencia)
  config.json      (arquitectura, clases, umbral)
  modelo_final.pt  (pesos de la ResNet-50 ajustada)
"""
import json
import re
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

import carros_utils as cu

RAIZ = Path(__file__).parent
MAX_MB = 10                  # tamaño máximo de la foto (también en .streamlit/config.toml)
LADO_MIN_AVISO = 150         # por debajo de esto (lado mayor, en px) se avisa que la exactitud puede caer
PLACA_RE = re.compile(r"^[A-Z]{3}\d{3}$|^[A-Z]{3}\d{2}[A-Z]$")   # carro: ABC123 · moto: ABC12D
OTRA_MARCA = "Otra marca (no está en la lista)"

# Cifras del informe. Copiarlas del cuaderno (evaluación en prueba) para que coincidan con la presentación.
METRICAS = {"exactitud": "89,2 %", "marca": "94,7 %", "top5": "98,0 %"}


# --------------------------------------------------------------------------- modelo
@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    """Carga una sola vez el modelo final guardado en el cuaderno (sin descargar pesos de internet)."""
    cfg = json.loads((RAIZ / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(2)
    modelo = cu.crear_modelo(cfg["arq"], len(cfg["clases"]), dropout=cfg["dropout"], pretrained=False)
    modelo.load_state_dict(torch.load(RAIZ / "modelo_final.pt", map_location="cpu", weights_only=True))
    modelo.eval()
    transform = cu.construir_transform(cfg["arq"], cfg["modo_geom"], entrenamiento=False)
    marcas = sorted({cu.extraer_marca(n) for n in cfg["clases"]})
    return {"modelo": modelo, "transform": transform, "cfg": cfg, "marcas": marcas}


# --------------------------------------------------------------------------- lógica (sin Streamlit)
def abrir_imagen(archivo):
    """Valida y abre la foto subida. Lanza ValueError con un mensaje claro para el usuario."""
    if archivo.size == 0:
        raise ValueError("El archivo está vacío.")
    if archivo.size > MAX_MB * 1024 * 1024:
        raise ValueError(f"El archivo pesa más de {MAX_MB} MB.")
    try:
        img = Image.open(archivo)
        img.load()
    except Exception as e:
        raise ValueError("No se pudo leer el archivo como imagen (¿está dañado o no es JPG/PNG?).") from e
    return img


def analizar(img, ctx):
    """Aplica el pipeline del cuaderno. Devuelve (top-3 [(clase, prob)], milisegundos de inferencia)."""
    t0 = time.perf_counter()
    top3, _ = cu.predecir(ctx["modelo"], img, ctx["transform"], ctx["cfg"]["clases"], k=3, device="cpu")
    return top3, (time.perf_counter() - t0) * 1000


def diagnosticar(top3, marca_runt, umbral, marcas_modelo):
    """Compara la MARCA registrada en el RUNT con la predicha. Devuelve (nivel, mensaje).

    Se compara por marca porque es lo que el modelo acierta más (marca vs. modelo/año exacto) y
    porque una confusión entre años del mismo modelo no es indicio de clonación.
    """
    nombre1, conf = top3[0]
    marca_pred = cu.extraer_marca(nombre1)
    marcas_top3 = [cu.extraer_marca(n) for n, _ in top3]

    if marca_runt == OTRA_MARCA or marca_runt not in marcas_modelo:
        return "info", ("**No comparable:** la marca registrada en el RUNT no está entre las que el modelo conoce. "
                        "Con este vehículo no se puede decir si coincide o no; el modelo igual daría una respuesta, "
                        "pero sería de otra marca. Hacer la verificación manual.")
    if conf < umbral:
        return "warn", (f"**Confianza baja ({conf:.0%} < {umbral:.0%}):** la resolución, el ángulo o el tipo de vehículo "
                        "no permiten una conclusión. Requiere **verificación manual**.")
    if marca_runt == marca_pred:
        return "ok", (f"**Coincide la marca:** el vehículo parece **{marca_pred}**, igual que el RUNT. "
                      "El modelo exacto y el año se aciertan menos que la marca: no los use como prueba.")
    if marca_runt in marcas_top3:
        return "warn", (f"**Inconcluso:** el modelo ve **{marca_pred}** como primera opción, pero **{marca_runt}** "
                        "aparece entre sus 3 primeras. No es suficiente para alertar; repetir con otra foto o verificar manualmente.")
    return "error", (f"**Posible discrepancia de marca:** el vehículo parece **{marca_pred}** y el RUNT registra "
                     f"**{marca_runt}**. Sugerir **verificación física** de chasis y motor. "
                     "*Herramienta de apoyo: nunca una acusación directa.*")


def mostrar_nivel(nivel, mensaje):
    {"ok": st.success, "warn": st.warning, "error": st.error, "info": st.info}[nivel](mensaje)


# --------------------------------------------------------------------------- interfaz
def main():
    st.set_page_config(page_title="Verificador vehículo–RUNT", page_icon="🚓", layout="wide")
    ctx = cargar_modelo()
    umbral = float(ctx["cfg"]["umbral"])

    st.title("🚓 Verificador de consistencia vehículo–RUNT")
    st.caption("Apoyo visual en terreno: compara la marca que ve el modelo en la foto con la marca registrada en el RUNT. "
               "Solo prioriza verificaciones; no prueba que una placa esté clonada.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Datos en terreno")
        placa = st.text_input("Placa del vehículo", placeholder="Ej: ABC123").strip().upper()
        marca_runt = st.selectbox("Marca registrada en el RUNT (según la consulta)", ctx["marcas"] + [OTRA_MARCA])

        modo = st.radio("Cargar fotografía por:", ("Subir imagen", "Tomar foto con la cámara"))
        if modo == "Subir imagen":
            archivo = st.file_uploader(f"Foto del vehículo (JPG o PNG, máx. {MAX_MB} MB)", type=["jpg", "jpeg", "png"])
        else:
            archivo = st.camera_input("Capturar foto")

        img = None
        if archivo is not None:
            try:
                img = abrir_imagen(archivo)
                st.image(img, caption=f"Foto cargada ({img.size[0]} × {img.size[1]} px)")
                if max(img.size) < LADO_MIN_AVISO:
                    st.warning(f"Imagen de muy baja resolución (lado mayor < {LADO_MIN_AVISO} px): "
                               "con fotos así el modelo falla casi siempre.")
            except ValueError as e:
                st.error(str(e))

        analizar_ya = st.button("🔍 Analizar vehículo", type="primary")

    with col2:
        st.subheader("2. Resultado")
        if not analizar_ya:
            st.info("Ingrese la placa, elija la marca del RUNT, cargue la foto y presione *Analizar vehículo*.")
        elif not PLACA_RE.match(placa):
            st.warning("⚠️ Placa inválida. Formato: 3 letras y 3 números (ABC123); motos: 3 letras, 2 números y 1 letra (ABC12D).")
        elif img is None:
            st.warning("⚠️ Cargue una foto válida del vehículo.")
        else:
            with st.spinner("Analizando la imagen…"):
                top3, ms = analizar(img, ctx)

            st.markdown(f"**Placa:** `{placa}`  ·  **RUNT:** {marca_runt}")
            st.markdown("### Lo que ve el modelo (3 primeras opciones)")
            for i, (nombre, p) in enumerate(top3, 1):
                st.write(f"**{i}. {nombre}** — {p:.1%}")
                st.progress(min(max(float(p), 0.0), 1.0))
            st.caption(f"Marca de la primera opción: {cu.extraer_marca(top3[0][0])} · inferencia en CPU: {ms:.0f} ms")

            st.markdown("### Diagnóstico")
            mostrar_nivel(*diagnosticar(top3, marca_runt, umbral, ctx["marcas"]))

    st.markdown("---")
    with st.expander("ℹ️ Cómo interpretar este resultado, límites y privacidad"):
        st.markdown(f"""
* **Modelo:** ResNet-50 con ajuste fino, entrenada con Stanford Cars (196 clases). En prueba: exactitud {METRICAS['exactitud']} en el modelo exacto, **{METRICAS['marca']} por marca** y {METRICAS['top5']} dentro de las 5 primeras opciones. Por eso la decisión se toma por **marca**.
* **Solo conoce {len(ctx['cfg']['clases'])} modelos de {len(ctx['marcas'])} marcas** (mercado de EE. UU., hasta 2012). Muchos carros comunes en Colombia no están. Siempre responde algo, aunque el vehículo no sea de esas clases.
* **No detecta clonaciones de la misma marca y modelo**: solo discrepancias evidentes de marca.
* **Falla con fotos muy pixeladas**, poco nítidas, parciales o que no muestran el vehículo completo.
* **Confianza menor a {umbral:.0%}:** verificar manualmente.
* **Privacidad:** la foto y la placa se procesan en memoria y no se guardan ni se envían a APIs externas. En esta demo en la nube use solo fotos de prueba, no fotos de ciudadanos reales.
        """)


if __name__ == "__main__":
    main()