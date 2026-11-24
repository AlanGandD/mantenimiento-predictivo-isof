# ==============================================================================
# MÓDULO A · ENTRENAMIENTO DEL MODELO DE MANTENIMIENTO PREDICTIVO
# Challenge Lab: Sistema Predictivo de Mantenimiento Industrial + Groq API
# Clase: ISOF CLASS | Docente: Dra. Aiblis Susej Herrera Palafox
# ------------------------------------------------------------------------------
# Ejecutar con:  python entrenar.py
# Genera:        modelo_mantenimiento.pkl
# ==============================================================================

import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ARCHIVO_DATOS = "sensores_planta.csv"
ARCHIVO_MODELO = "modelo_mantenimiento.pkl"

# Orden EXACTO de las features: el mismo que usará app.py al predecir.
FEATURES = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
OBJETIVO = "estado"
ETIQUETAS = {
    0: "Operación Normal",
    1: "Mantenimiento Preventivo",
    2: "Fallo Crítico",
}
UMBRAL_ACCURACY = 0.80


def cargar_datos(ruta: str = ARCHIVO_DATOS) -> pd.DataFrame:
    """Lee el CSV y normaliza encabezados/espacios para evitar fallos de formato."""
    try:
        df = pd.read_csv(ruta, sep=",")
    except FileNotFoundError:
        print(f"❌ No se encontró '{ruta}'. Coloca el CSV en la misma carpeta que este script.")
        sys.exit(1)
    except Exception as error:  # CSV corrupto, codificación extraña, etc.
        print(f"❌ Error al leer '{ruta}': {error}")
        sys.exit(1)

    # Limpieza de encabezados y de celdas de texto (espacios o comillas accidentales).
    df.columns = [str(col).strip().strip('"').lower() for col in df.columns]
    for columna in df.columns:
        if df[columna].dtype == object:
            df[columna] = df[columna].astype(str).str.strip().str.strip('"')

    print(f"📂 Archivo cargado: {len(df)} registros | Columnas: {list(df.columns)}")
    return df


def main() -> None:
    print("=" * 66)
    print(" SISTEMA DE MANTENIMIENTO PREDICTIVO · Entrenamiento del modelo")
    print("=" * 66)

    # 1. Lectura del dataset
    df = cargar_datos()

    # 2. Validación de esquema
    faltantes = [col for col in FEATURES + [OBJETIVO] if col not in df.columns]
    if faltantes:
        print(f"❌ Faltan columnas obligatorias: {faltantes}")
        sys.exit(1)

    # 3. Limpieza: eliminar nulos (requisito del laboratorio)
    antes = len(df)
    df = df.dropna()
    print(f"🧹 Limpieza con dropna(): {antes} → {len(df)} registros válidos")

    if len(df) < 30:
        print(f"⚠️  Advertencia: el laboratorio exige ≥ 30 registros y hay {len(df)}.")

    # 4. Variables de entrada (X) y objetivo (y)
    X = df[FEATURES].astype(float)
    y = df[OBJETIVO].astype(int)

    print(f"📊 Distribución de clases: {y.value_counts().sort_index().to_dict()}")

    # 5. División 80% entrenamiento / 20% prueba (estratificada para no perder clases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✂️  Split 80/20 → entrenamiento: {len(X_train)} | prueba: {len(X_test)}")

    # 6. Entrenamiento del clasificador
    print("🧠 Entrenando RandomForestClassifier (200 árboles)...")
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    # 7. Evaluación de precisión (métrica exigida por la rúbrica)
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)

    print(f"🎯 Precisión del modelo (Accuracy): {precision * 100:.2f}%")
    print(f"\n📋 Reporte por clase:\n{classification_report(y_test, y_pred, digits=3, zero_division=0)}")
    print(f"🧩 Matriz de confusión:\n{confusion_matrix(y_test, y_pred)}")

    # 8. Exportación del modelo
    joblib.dump(modelo, ARCHIVO_MODELO)
    print(f"💾 Modelo exportado como '{ARCHIVO_MODELO}'")

    # 9. Verificación del entregable
    if precision >= UMBRAL_ACCURACY:
        print(f"✅ CUMPLE la rúbrica: accuracy {precision:.2%} ≥ {UMBRAL_ACCURACY:.0%}")
    else:
        print(f"❌ NO cumple: accuracy {precision:.2%} < {UMBRAL_ACCURACY:.0%}. Ajusta el dataset.")
        sys.exit(1)

    print("\n▶️  Siguiente paso:  streamlit run app.py")
    print("=" * 66)


if __name__ == "__main__":
    main()
