# ==============================================================================
# MÓDULOS B, C y D · DASHBOARD WEB + IA GENERATIVA + SEGURIDAD
# Challenge Lab: Sistema Predictivo de Mantenimiento Industrial + Groq API
# Clase: ISOF CLASS | Docente: Dra. Aiblis Susej Herrera Palafox
# ------------------------------------------------------------------------------
# Ejecutar con:  streamlit run app.py
# Requiere:      modelo_mantenimiento.pkl  (generado por entrenar.py)
# NO requiere:   sensores_planta.csv  → la app funciona aunque el CSV no exista.
# ==============================================================================

import hashlib
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

try:
    from groq import Groq

    GROQ_INSTALADO = True
except ImportError:  # la app sigue operando con el diagnóstico de ML
    Groq = None
    GROQ_INSTALADO = False


# ------------------------------------------------------------------------------
# 1. CONSTANTES DEL DOMINIO
# ------------------------------------------------------------------------------
ARCHIVO_MODELO = "modelo_mantenimiento.pkl"

# Orden EXACTO de las features usado en entrenar.py
FEATURES = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]

ESTADOS = {
    0: {
        "nombre": "Operación Normal",
        "icono": "✅",
        "color": "#34d399",
        "prioridad": "RUTINA",
        "accion": "Monitoreo preventivo programado",
    },
    1: {
        "nombre": "Mantenimiento Preventivo",
        "icono": "⚠️",
        "color": "#fbbf24",
        "prioridad": "PROGRAMADA",
        "accion": "Intervención antes de la siguiente ventana de producción",
    },
    2: {
        "nombre": "Fallo Crítico",
        "icono": "🚨",
        "color": "#f87171",
        "prioridad": "URGENTE",
        "accion": "Paro inmediato de la línea y bloqueo de seguridad",
    },
}

# Valores por defecto del panel (máquina en régimen estable)
DEFAULTS = {
    "in_temp": 62.5,
    "in_vib": 1.8,
    "in_horas": 450,
    "in_pres": 95.0,
}

# Casos de la matriz de pruebas de estrés del laboratorio
CASOS_PRUEBA = {
    "Caso 1 · Operación Estable": {"in_temp": 60.0, "in_vib": 1.5, "in_horas": 300, "in_pres": 92.0},
    "Caso 2 · Desgaste Preventivo": {"in_temp": 80.0, "in_vib": 4.5, "in_horas": 2000, "in_pres": 116.0},
    "Caso 3 · Sobrecarga Crítica": {"in_temp": 98.0, "in_vib": 8.8, "in_horas": 4000, "in_pres": 146.0},
}

PROMPT_SISTEMA = """Eres el Jefe de Mantenimiento Industrial de una planta de manufactura
automatizada. Tienes 20 años de experiencia en confiabilidad, análisis de vibraciones,
termografía y gestión de órdenes de trabajo bajo normas ISO 10816.

Recibes el diagnóstico de un modelo de Machine Learning (Random Forest) que clasifica el
estado de una máquina en 3 niveles. Tu trabajo es traducir ese diagnóstico técnico a un
documento operativo claro, realista y accionable para los técnicos de piso.

REGLAS ESTRICTAS:
1. Responde SIEMPRE en español, en formato Markdown limpio (sin bloques ```).
2. Nunca inventes lecturas distintas a las que recibes ni recomiendes repuestos específicos
   con número de parte inexistente.
3. Si el estado es 0 (Operación Normal): entrega SOLO un dictamen breve de monitoreo
   preventivo de máximo 5 líneas. NO generes orden de trabajo.
4. Si el estado es 1 o 2: genera una Orden de Trabajo técnica completa con EXACTAMENTE estas
   secciones y este orden:
   ## 🧾 ORDEN DE TRABAJO — <FOLIO>
   **Folio:** <FOLIO>
   **Prioridad:** <prioridad>
   ### 🔍 Diagnóstico
   ### 🛠️ Acciones Técnicas
   (exactamente 3 acciones, numeradas, cada una con su criterio de aceptación)
   ### 🦺 Equipo de Seguridad (EPP)
   (lista de equipo de protección personal obligatorio)
5. Usa el folio recibido de forma literal, sin modificarlo.
6. Si citas normas (ISO 10816, etc.), úsalas como referencia de zonas de severidad;
   no inventes límites térmicos exactos atribuidos a una norma.
7. Sé conciso: máximo 400 palabras. Nada de introducciones ni despedidas."""


# ------------------------------------------------------------------------------
# 2. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Mantenimiento Predictivo IA · ISOF",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .hero {
          border: 1px solid #23283b;
          background: linear-gradient(135deg, #141724 0%, #10121d 100%);
          border-radius: 16px;
          padding: 22px 26px;
          margin-bottom: 8px;
          position: relative;
          overflow: hidden;
      }
      .hero::before {
          content: '';
          position: absolute; top: 0; left: 0; right: 0; height: 4px;
          background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #3b82f6 100%);
      }
      .hero h1 { font-size: 1.55rem; margin: 6px 0 6px 0; color: #ffffff; }
      .hero p { color: #94a3b8; font-size: 0.92rem; margin: 0; }
      .chip {
          display: inline-block; font-size: 0.72rem; font-weight: 700;
          letter-spacing: .5px; text-transform: uppercase;
          padding: 4px 12px; border-radius: 20px; margin-right: 8px;
      }
      .chip-purple { background: rgba(192,132,252,.16); color: #c084fc; border: 1px solid rgba(192,132,252,.45); }
      .chip-green  { background: rgba(52,211,153,.14); color: #34d399; border: 1px solid rgba(52,211,153,.35); }
      .chip-blue   { background: rgba(96,165,250,.14); color: #60a5fa; border: 1px solid rgba(96,165,250,.35); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 3. CARGA DEL MODELO (Módulo B) — la app depende SOLO del .pkl
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta: str = ARCHIVO_MODELO):
    """Carga el modelo entrenado una sola vez por sesión del servidor."""
    return joblib.load(ruta)


try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error(
        "❌ No se encontró **modelo_mantenimiento.pkl**.\n\n"
        "Ejecuta primero el entrenamiento en tu terminal:  `python entrenar.py`"
    )
    st.stop()
except Exception as error:
    st.error(f"❌ No se pudo cargar el modelo entrenado: {error}")
    st.stop()


# ------------------------------------------------------------------------------
# 4. RESOLUCIÓN SEGURA DE LA API KEY (Módulo D)
# ------------------------------------------------------------------------------
def obtener_api_key():
    """Busca la clave en st.secrets (nube) y, si no existe, en el entorno local."""
    try:
        clave = st.secrets["GROQ_API_KEY"]
        if clave and not str(clave).startswith("gsk_TU"):
            return str(clave).strip(), None
    except Exception:
        # Secret ausente o archivo secrets.toml inexistente: se intenta el fallback.
        pass

    clave = os.environ.get("GROQ_API_KEY", "").strip()
    if clave:
        return clave, None

    return None, (
        "No se encontró **GROQ_API_KEY**. Configúrala en `.streamlit/secrets.toml` "
        "o en la variable de entorno `GROQ_API_KEY`."
    )


# ------------------------------------------------------------------------------
# 5. DETECCIÓN DINÁMICA DEL MODELO ACTIVO EN GROQ (Módulo C)
# ------------------------------------------------------------------------------
def detectar_modelo_groq(client):
    """Evita errores 404 usando modelos realmente disponibles en la cuenta."""
    modelos = []
    try:
        modelos = [
            m.id
            for m in client.models.list().data
            if not any(
                excluida in m.id.lower()
                for excluida in [
                    "whisper", "tts", "playai", "orpheus", "guard", "safeguard",
                    "embed", "rerank", "moderation", "realtime", "vision",
                ]
            )
        ]
    except Exception:
        modelos = []

    if not modelos:
        return "llama-3.3-70b-versatile"

    # Preferencia por calidad de redacción técnica en español.
    preferencia = [
        "llama-3.3-70b", "gpt-oss-120b", "llama-3.1-8b", "qwen",
        "gpt-oss-20b", "llama", "mixtral", "gemma", "allam",
    ]
    return next(
        (m for pref in preferencia for m in modelos if pref in m.lower()),
        modelos[0],
    )


def limpiar_markdown(texto: str) -> str:
    """Quita cercas de código que algunos modelos añaden alrededor de la respuesta."""
    texto = (texto or "").strip()
    if texto.startswith("```"):
        lineas = texto.splitlines()
        lineas = lineas[1:]
        if lineas and lineas[-1].strip().startswith("```"):
            lineas = lineas[:-1]
        texto = "\n".join(lineas).strip()
    return texto


# ------------------------------------------------------------------------------
# 6. GENERACIÓN DE LA ORDEN DE TRABAJO (Módulo C) — cacheada para no repetir llamadas
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def generar_documento(estado, temperatura, vibracion, horas, presion, probabilidad, folio, nonce):
    """Llama a Groq y devuelve un dictamen. Nunca lanza excepción: la app no colapsa."""
    if not GROQ_INSTALADO:
        return {"ok": False, "error": "El paquete 'groq' no está instalado.", "modelo": None}

    clave, mensaje = obtener_api_key()
    if not clave:
        return {"ok": False, "error": mensaje, "modelo": None}

    info = ESTADOS[estado]
    try:
        client = Groq(api_key=clave)
        modelo_groq = detectar_modelo_groq(client)

        if estado == 0:
            instruccion = (
                "El modelo clasificó la máquina en OPERACIÓN NORMAL. "
                "Redacta un dictamen breve de monitoreo preventivo (máximo 5 líneas) "
                "con la lectura destacada y la recomendación de la próxima inspección. "
                "No generes orden de trabajo."
            )
        else:
            instruccion = (
                f"El modelo clasificó la máquina en {info['nombre'].upper()} "
                f"(prioridad {info['prioridad']}). Redacta la Orden de Trabajo completa "
                f"con el folio {folio}, usando exactamente las secciones indicadas en tus reglas."
            )

        prompt_usuario = f"""DIAGNÓSTICO DEL MODELO DE MACHINE LEARNING
- Estado predicho: {estado} → {info['nombre']}
- Probabilidad del estado: {probabilidad:.1%}
- Temperatura: {temperatura:.1f} °C
- Vibración: {vibracion:.1f} mm/s
- Horas de operación acumuladas: {int(horas)} h
- Presión de línea: {presion:.1f} PSI
- Fecha del reporte: {datetime.now().strftime('%d/%m/%Y %H:%M')}

TAREA: {instruccion}"""

        completion = client.chat.completions.create(
            model=modelo_groq,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": prompt_usuario},
            ],
            temperature=0.4,
            max_tokens=900,
        )
        texto = limpiar_markdown(completion.choices[0].message.content)

        # Garantía mínima de estructura: toda orden de trabajo debe llevar folio.
        # El estado 0 solo genera un dictamen, por lo que no lleva folio de orden.
        if estado in (1, 2) and folio not in texto:
            texto = f"**Folio:** {folio}\n\n{texto}"

        return {"ok": True, "texto": texto, "modelo": modelo_groq, "error": None}

    except Exception as error:
        # Caso de prueba 5: clave inválida, cuota agotada, red caída o 404.
        return {"ok": False, "error": f"{type(error).__name__}: {error}", "modelo": None}


def documento_respaldo(estado, temperatura, vibracion, horas, presion, folio):
    """Plantilla local de contingencia si la IA no está disponible."""
    info = ESTADOS[estado]
    if estado == 0:
        return (
            f"### {info['icono']} Dictamen de monitoreo preventivo\n\n"
            f"La máquina opera dentro de parámetros estables "
            f"({temperatura:.1f} °C · {vibracion:.1f} mm/s · {int(horas)} h · {presion:.1f} PSI). "
            "Se recomienda mantener el plan de inspección visual semanal y repetir la "
            "medición de vibración en la próxima ventana de mantenimiento."
        )
    return f"""## 🧾 ORDEN DE TRABAJO — {folio}
**Folio:** {folio}
**Prioridad:** {info['prioridad']}

### 🔍 Diagnóstico
Clasificación automática: **{info['nombre']}** con lecturas de {temperatura:.1f} °C,
{vibracion:.1f} mm/s de vibración, {int(horas)} h acumuladas y {presion:.1f} PSI de presión.

### 🛠️ Acciones Técnicas
1. **Aislar el equipo** y verificar el estado del rodamiento con medición puntual.
2. **Revisar lubricación y alineación**, registrando el valor posterior en bitácora.
3. **Programar prueba de operación** de 30 minutos y validar contra los límites ISO.

### 🦺 Equipo de Seguridad (EPP)
- Casco y calzado de seguridad dieléctrico.
- Guantes anticorte y protección auditiva.
- Lentes de seguridad y bloqueo/etiquetado (LOTO) antes de intervenir."""


# ------------------------------------------------------------------------------
# 7. ENCABEZADO
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <span class="chip chip-purple">Challenge Lab</span>
      <span class="chip chip-green">Random Forest · 3 clases</span>
      <span class="chip chip-blue">Groq API</span>
      <h1>🏭 Sistema Híbrido de Mantenimiento Predictivo</h1>
      <p>Machine Learning para clasificar la severidad de la falla + IA generativa para emitir
      la orden de trabajo técnica. ISOF CLASS · Dra. Aiblis Susej Herrera Palafox.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 8. CONTROLES LATERALES (Módulo B)
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Lecturas de la máquina")

with st.sidebar.expander("🧪 Cargar caso de prueba", expanded=False):
    st.caption("Atajos para la matriz de evaluación del laboratorio.")
    for nombre, valores in CASOS_PRUEBA.items():
        if st.button(nombre, use_container_width=True, key=f"btn_{nombre}"):
            for clave, valor in valores.items():
                st.session_state[clave] = valor
            st.rerun()

# Valores iniciales (solo si la clave aún no existe en la sesión).
for clave, valor in DEFAULTS.items():
    if clave not in st.session_state:
        st.session_state[clave] = valor

temperatura = st.sidebar.slider(
    "🌡️ Temperatura (°C)", 30.0, 130.0, step=0.5, key="in_temp",
    help="Rango típico de operación: 55–70 °C. Por encima de 90 °C es crítico.",
)
vibracion = st.sidebar.slider(
    "📳 Vibración (mm/s)", 0.0, 12.0, step=0.1, key="in_vib",
    help="Límite ISO 10816 para máquinas clase II: 2.8 mm/s (alerta), 4.5 mm/s (alarma).",
)
horas_operacion = st.sidebar.slider(
    "⏱️ Horas de operación", 0, 5000, step=10, key="in_horas",
    help="Horas acumuladas desde la última intervención mayor.",
)
presion = st.sidebar.slider(
    "🎚️ Presión de línea (PSI)", 80.0, 160.0, step=0.5, key="in_pres",
    help="Presión nominal de trabajo: 90–100 PSI.",
)

st.sidebar.divider()
col_a, col_b = st.sidebar.columns(2)
if col_a.button("♻️ Reiniciar", use_container_width=True, help="Vuelve a las lecturas por defecto"):
    for clave, valor in DEFAULTS.items():
        st.session_state[clave] = valor
    st.rerun()
if col_b.button("🔄 Regenerar", use_container_width=True, help="Vuelve a pedir el documento a Groq"):
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "**Módulos:** A · Entrenamiento (RandomForest) · B · Dashboard · "
    "C · Orden de Trabajo con Groq · D · Secrets y despliegue en la nube."
)


# ------------------------------------------------------------------------------
# 9. PREDICCIÓN DEL MODELO
# ------------------------------------------------------------------------------
datos_entrada = pd.DataFrame(
    [[temperatura, vibracion, horas_operacion, presion]], columns=FEATURES
)

estado = int(modelo.predict(datos_entrada)[0])

# predict_proba() → vector de probabilidades alineado con modelo.classes_
probabilidades = modelo.predict_proba(datos_entrada)[0]
mapa_proba = {int(clase): float(prob) for clase, prob in zip(modelo.classes_, probabilidades)}
prob_estado = mapa_proba.get(estado, max(probabilidades))
info = ESTADOS[estado]

# Folio determinista: mismas lecturas ⇒ mismo folio (permite cachear la llamada a la IA).
semilla = f"{estado}-{temperatura:.1f}-{vibracion:.1f}-{int(horas_operacion)}-{presion:.1f}"
folio = f"OT-{datetime.now().year}-{hashlib.md5(semilla.encode()).hexdigest()[:4].upper()}"


# ------------------------------------------------------------------------------
# 10. TARJETAS DE LECTURA (st.metric en 4 columnas)
# ------------------------------------------------------------------------------
st.subheader("📊 Lecturas actuales de los sensores")

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperatura", f"{temperatura:.1f} °C", help="Sensor térmico del eje principal")
    col2.metric("Vibración", f"{vibracion:.1f} mm/s", help="Acelerómetro en el rodamiento")
    col3.metric("Operación", f"{int(horas_operacion):,} h", help="Horómetro acumulado")
    col4.metric("Presión", f"{presion:.1f} PSI", help="Manómetro de la línea hidráulica")


# ------------------------------------------------------------------------------
# 11. DIAGNÓSTICO DEL MODELO (alertas diferenciadas por clase)
# ------------------------------------------------------------------------------
st.subheader("🧠 Diagnóstico del modelo de Machine Learning")

mensaje = (
    f"{info['icono']} **{info['nombre'].upper()}** — "
    f"probabilidad {prob_estado:.1%} "
    f"(clase {estado}) · Acción sugerida: {info['accion']}"
)

if estado == 0:
    st.success(mensaje)
elif estado == 1:
    st.warning(mensaje)
else:
    st.error(mensaje)

# Distribución completa de probabilidades
with st.container(border=True):
    st.caption("Probabilidades por clase — `predict_proba()`")
    for clase in sorted(ESTADOS):
        meta = ESTADOS[clase]
        p = mapa_proba.get(clase, 0.0)
        fila = st.columns([2, 5, 1.2])
        fila[0].markdown(f"{meta['icono']} **{clase} · {meta['nombre']}**")
        fila[1].progress(min(max(p, 0.0), 1.0))
        fila[2].markdown(f"`{p:6.1%}`")

if prob_estado < 0.60:
    st.info(
        "ℹ️ La confianza del modelo es baja para estas lecturas. "
        "Interprétalas como una señal temprana y respáldalas con inspección física."
    )


# ------------------------------------------------------------------------------
# 12. ORDEN DE TRABAJO CON GROQ API (Módulo C)
# ------------------------------------------------------------------------------
st.subheader("🤖 " + ("Monitoreo preventivo con IA" if estado == 0 else "Orden de trabajo técnica con IA"))

resultado = generar_documento(
    estado,
    float(temperatura),
    float(vibracion),
    int(horas_operacion),
    float(presion),
    float(prob_estado),
    folio,
    st.session_state.get("nonce", 0),
)

if resultado["ok"]:
    with st.container(border=True):
        st.markdown(resultado["texto"])
    sufijo = f" · folio `{folio}`" if estado in (1, 2) else ""
    st.caption(f"⚡ Generado con Groq · modelo `{resultado['modelo']}`{sufijo}")
else:
    st.warning(
        "No fue posible generar el documento con la IA. "
        "El diagnóstico de Machine Learning se mantiene operativo: se muestra una "
        "orden de trabajo preliminar generada localmente."
    )
    with st.expander("🔎 Detalle técnico del incidente con la API"):
        st.code(str(resultado["error"]))
    with st.container(border=True):
        st.markdown(
            documento_respaldo(estado, temperatura, vibracion, horas_operacion, presion, folio)
        )
    st.caption("📄 Documento de contingencia local · sin intervención de la IA")


# ------------------------------------------------------------------------------
# 13. PIE DE PÁGINA
# ------------------------------------------------------------------------------
st.divider()
st.caption(
    "Sistema híbrido ML + IA generativa · Challenge Lab ISOF CLASS · "
    "El modelo `.pkl` es la única dependencia del dashboard: no lee el CSV en tiempo de ejecución."
)
