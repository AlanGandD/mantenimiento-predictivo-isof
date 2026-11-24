# 🏭 Sistema Híbrido de Mantenimiento Predictivo Industrial

**Challenge Lab — ISOF CLASS** · Docente: Dra. Aiblis Susej Herrera Palafox
Machine Learning (**Random Forest**, 3 clases) + IA generativa (**Groq API**) para emitir
órdenes de trabajo técnico a partir de telemetría de sensores.

---

## 📌 ¿Qué hace el sistema?

1. **Clasifica** el estado de una máquina en 3 niveles de severidad usando 4 lecturas de sensores.
2. **Calcula** la probabilidad de cada clase con `predict_proba()`.
3. **Redacta** automáticamente —vía Groq— un dictamen de rutina (estado 0) o una **Orden de
   Trabajo** en Markdown con folio, diagnóstico, 3 acciones técnicas y EPP (estados 1 y 2).
4. **Resiste fallos**: si la API falla, la app sigue operando y muestra una orden de contingencia.

| Clase | Significado | Acción |
|:---:|---|---|
| `0` | Operación Normal | Monitoreo preventivo |
| `1` | Mantenimiento Preventivo | Intervención programada |
| `2` | Fallo Crítico | Paro inmediato |

---

## 📂 Estructura del proyecto

```
challenge_ml/
├── sensores_planta.csv          # Dataset ampliado: 36 registros (12 por clase)
├── entrenar.py                  # MÓDULO A · Pipeline de entrenamiento
├── app.py                       # MÓDULOS B, C y D · Dashboard + Groq + seguridad
├── modelo_mantenimiento.pkl     # Modelo exportado (generado por entrenar.py)
├── pruebas_estres.py            # Arnés de la matriz de 5 casos (AppTest)
├── requirements.txt
├── .gitignore                   # Excluye secrets.toml del repositorio
├── README.md
└── .streamlit/
    ├── config.toml              # Tema visual
    └── secrets.toml.example     # Plantilla de la API Key (el real NO se sube)
```

---

## ⚙️ Instalación y ejecución local

> ⚠️ Trabaja en una carpeta **local**, fuera de OneDrive, para evitar bloqueos al leer el
> CSV y al exportar el `.pkl`.

```bash
# 1. Entorno virtual (recomendado)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 2. Dependencias
python -m pip install -r requirements.txt

# 3. Entrenar el modelo (imprime la precisión y genera el .pkl)
python entrenar.py

# 4. Levantar el dashboard
streamlit run app.py
```

Salida esperada del entrenamiento:

```
🎯 Precisión del modelo (Accuracy): 100.00%
💾 Modelo exportado como 'modelo_mantenimiento.pkl'
✅ CUMPLE la rúbrica: accuracy 100.00% ≥ 80%
```

---

## 🔑 Configuración de la API Key

**Local** — crea el archivo `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_tu_llave_real"
```

**Nube** — en Streamlit Community Cloud: *Advanced settings → Secrets* y pega la misma línea.

La app resuelve la clave en este orden: `st.secrets` → variable de entorno `GROQ_API_KEY` → aviso
en pantalla (nunca se rompe). El archivo `secrets.toml` está en `.gitignore`, por lo que **no se sube
a GitHub**.

---

## 🧩 Trazabilidad con la rúbrica

| Módulo | Implementación | Dónde |
|---|---|---|
| **A · Entrenamiento** (30%) | `pandas` + `dropna()` + split 80/20 + `RandomForestClassifier` + `accuracy_score` + `joblib.dump` | `entrenar.py` |
| **B · Dashboard** (25%) | `@st.cache_resource`, 4 sliders en `st.sidebar`, 4 `st.metric` en columnas, alertas `success`/`warning`/`error` + probabilidades | `app.py` §3 y §8–§11 |
| **C · Groq** (25%) | `client.models.list()` dinámico, System Prompt de Jefe de Mantenimiento, dictamen (0) u Orden de Trabajo (1/2) | `app.py` §5–§6 |
| **D · Nube y seguridad** (20%) | `st.secrets` con `try/except`, `.gitignore`, repo limpio y despliegue | `app.py` §4 + este README |

---

## 📊 Dataset

`sensores_planta.csv` contiene **36 registros** (los 24 de la base + 12 nuevos coherentes,
**4 por cada estado**), sin comillas dobles y perfectamente balanceado:

| Columnas | Rol |
|---|---|
| `temperatura_c`, `vibracion_mms`, `horas_operacion`, `presion_psi` | Features de entrada |
| `estado` | Objetivo multiclase (0, 1, 2) |

Separación clara por clase: estado 0 ≈ 55–67 °C / 1.0–2.4 mm/s; estado 1 ≈ 74–86 °C / 3.6–5.5 mm/s;
estado 2 ≈ 90–100 °C / 7.0–9.4 mm/s. Por eso el modelo generaliza con precisión del 100 %.

---

## 🧪 Matriz de pruebas de estrés

```bash
python pruebas_estres.py
```

El arnés usa `streamlit.testing.v1.AppTest`: levanta la app real, mueve los sliders y verifica el
renderizado. Resultado de la última ejecución:

| # | Escenario | Verificación | Estado |
|:---:|---|---|:---:|
| 1 | Operación Estable (60 °C, 1.5, 300 h, 92 PSI) | `st.success` exclusivo con probabilidad | ✅ |
| 2 | Desgaste Preventivo (80 °C, 4.5, 2000 h, 116 PSI) | `st.warning` + OT de Groq con todas las secciones | ✅ |
| 3 | Sobrecarga Crítica (98 °C, 8.8, 4000 h, 146 PSI) | `st.error` + OT urgente con EPP | ✅ |
| 4 | Desacoplamiento (CSV renombrado) | La app arranca igual: solo depende del `.pkl` | ✅ |
| 5 | Resiliencia de API (clave inválida) | `try/except` captura el error, el diagnóstico de ML permanece | ✅ |

---

## ☁️ Despliegue en Streamlit Community Cloud

1. **Crear el repositorio** en GitHub (`mantenimiento-predictivo-isof`) y subir el contenido de esta carpeta:
   ```bash
   git init
   git add .
   git commit -m "Sistema híbrido de mantenimiento predictivo con Groq"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/mantenimiento-predictivo-isof.git
   git push -u origin main
   ```
2. Verifica que **`modelo_mantenimiento.pkl`** esté incluido en el repo (pesa poco y es indispensable).
3. Entra a <https://share.streamlit.io> → **New app** → elige el repo, rama `main` y archivo `app.py`.
4. En **Advanced settings → Secrets** agrega:
   ```toml
   GROQ_API_KEY = "gsk_tu_llave_real"
   ```
5. **Deploy** y copia la URL pública (`https://tu-app.streamlit.app`).

---

## 📦 Evidencias para la plataforma

- [ ] URL pública de la app desplegada.
- [ ] URL del repositorio de GitHub.
- [ ] Captura de la terminal con el `Accuracy` de `entrenar.py`.
- [ ] Captura de la Orden de Trabajo generada en la app.

---

## 🛠️ Decisiones de diseño relevantes

- **Folio determinista** (`OT-AAAA-XXXX`): derivado de un hash de las lecturas, de modo que las
  mismas entradas produzcan el mismo folio y la respuesta de la IA pueda cachearse sin gastar cuota.
- **Caché de la llamada a Groq** (`@st.cache_data`, 30 min): cada rerun de Streamlit no genera una
  petición nueva. El botón *🔄 Regenerar* invalida la caché.
- **Orden de contingencia local**: si no hay API disponible se muestra una plantilla equivalente
  generada en Python, para que el entregable nunca aparezca incompleto.
- **Desacople del CSV**: `app.py` no abre el CSV en ningún momento; solo carga el `.pkl`.
