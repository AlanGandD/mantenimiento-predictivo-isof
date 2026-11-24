# ==============================================================================
# MATRIZ DE PRUEBAS DE ESTRÉS (5 casos del Challenge Lab)
# Se ejecuta con:  python pruebas_estres.py
# Usa el framework oficial streamlit.testing.v1.AppTest: levanta la app de verdad,
# mueve los sliders y verifica qué se renderiza en cada escenario.
# ==============================================================================

import ast
import os
import shutil
import sys
import tomllib

import streamlit as st
from streamlit.testing.v1 import AppTest

APP = "app.py"
CSV = "sensores_planta.csv"
TIMEOUT = 180

resultados = []


def registrar(caso: str, ok: bool, detalle: str, omitido: bool = False) -> None:
    marca = "OMITIDO" if omitido else ("PASA" if ok else "FALLA")
    icono = "⏭️" if omitido else ("✅" if ok else "❌")
    print(f"{icono} {caso}: {marca} → {detalle}")
    resultados.append({"caso": caso, "ok": ok or omitido, "omitido": omitido})


def leer_clave() -> str:
    """Lee la API Key desde el entorno o desde .streamlit/secrets.toml."""
    clave = os.environ.get("GROQ_API_KEY", "").strip()
    if clave:
        return clave
    try:
        with open(".streamlit/secrets.toml", "rb") as archivo:
            return str(tomllib.load(archivo).get("GROQ_API_KEY", "")).strip()
    except Exception:
        return ""


def app_nueva(clave: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.secrets["GROQ_API_KEY"] = clave
    return at


def fijar_lecturas(at: AppTest, temperatura, vibracion, horas, presion) -> None:
    at.slider(key="in_temp").set_value(float(temperatura))
    at.slider(key="in_vib").set_value(float(vibracion))
    at.slider(key="in_horas").set_value(int(horas))
    at.slider(key="in_pres").set_value(float(presion))
    at.run()


def texto(elementos) -> str:
    return " ".join(str(elemento.value) for elemento in elementos)


def probar_caso_ml(numero, nombre, lecturas, tipo_alerta, palabra_clave):
    """Casos 1-3: la app debe mostrar el banner correcto con su probabilidad."""
    caso = f"Caso {numero} · {nombre}"
    try:
        at = app_nueva(CLAVE)
        at.run()
        fijar_lecturas(at, *lecturas)

        if at.exception:
            registrar(caso, False, f"excepción en la app: {at.exception[0].value}")
            return

        conteos = {"success": len(at.success), "warning": len(at.warning), "error": len(at.error)}
        cuerpo = texto(getattr(at, tipo_alerta)).lower()

        ok = (
            conteos[tipo_alerta] > 0
            and sum(conteos.values()) == conteos[tipo_alerta]  # no debe haber otra alerta
            and palabra_clave.lower() in cuerpo
            and "probabilidad" in cuerpo
        )
        detalle = f"banner {tipo_alerta} exclusivo con probabilidad · «{texto(getattr(at, tipo_alerta))[:70]}»"
        registrar(caso, ok, detalle if ok else f"conteos={conteos} · texto=«{cuerpo[:80]}»")
    except Exception as error:
        registrar(caso, False, f"error del arnés: {error}")


def probar_orden_groq(numero, nombre, lecturas, debe_contener):
    """Verifica que Groq devuelva la orden de trabajo estructurada."""
    caso = f"Caso {numero} · {nombre}"
    if not CLAVE_VALIDA:
        registrar(caso, True, "sin API Key configurada: verificación de IA omitida", omitido=True)
        return

    at = app_nueva(CLAVE)
    at.run()
    fijar_lecturas(at, *lecturas)

    if at.exception:
        registrar(caso, False, f"excepción en la app: {at.exception[0].value}")
        return

    cuerpo = " ".join(str(elemento.value) for elemento in at.markdown)
    faltantes = [token for token in debe_contener if token.lower() not in cuerpo.lower()]
    fallo_ia = any("no fue posible generar" in str(e.value).lower() for e in at.warning)

    ok = not faltantes and not fallo_ia
    detalle = "documento de Groq con Folio, Diagnóstico, 3 Acciones y EPP" if ok else f"faltan={faltantes} · fallo_ia={fallo_ia}"
    registrar(caso, ok, detalle)


def probar_sin_csv():
    """Caso 4: la app debe funcionar aunque el CSV no exista."""
    caso = "Caso 4 · Desacoplamiento (CSV ausente)"
    respaldo = CSV + ".bak"
    try:
        if os.path.exists(CSV):
            shutil.move(CSV, respaldo)

        at = app_nueva(CLAVE)
        at.run()

        # Verificación estática con AST: app.py no debe abrir el CSV en ejecución.
        with open(APP, encoding="utf-8") as archivo:
            arbol = ast.parse(archivo.read())
        llamadas_csv = [
            nodo for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Attribute)
            and nodo.func.attr == "read_csv"
        ]
        literales = [
            nodo.value for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)
        ]
        menciona_csv = any("sensores_planta" in texto_ for texto_ in literales)
        sin_dependencia = not llamadas_csv and not menciona_csv

        ok = (not at.exception) and sin_dependencia and len(at.success) > 0
        detalle = (
            "arrancó con el CSV renombrado y siguió prediciendo desde el .pkl"
            if ok
            else f"excepcion={bool(at.exception)} · sin_dependencia={sin_dependencia} · datos={menciona_csv}"
        )
        registrar(caso, ok, detalle)
    finally:
        if os.path.exists(respaldo):
            shutil.move(respaldo, CSV)


def probar_resiliencia_api():
    """Caso 5: con clave inválida la app no debe colapsar."""
    caso = "Caso 5 · Resiliencia de API (clave inválida)"
    st.cache_data.clear()
    at = app_nueva("gsk_clave_invalida_para_la_prueba")
    at.run()
    fijar_lecturas(at, 61.0, 1.6, 350, 93.0)

    if at.exception:
        registrar(caso, False, f"la app colapsó: {at.exception[0].value}")
        return

    aviso = any("no fue posible generar" in str(e.value).lower() for e in at.warning)
    diagnostico_ml = len(at.success) > 0
    respaldo = any("orden de trabajo" in str(e.value).lower() for e in at.markdown)

    ok = aviso and diagnostico_ml and respaldo
    detalle = (
        "try/except capturó el error, el diagnóstico de ML siguió visible y hubo orden de contingencia"
        if ok
        else f"aviso={aviso} · ml={diagnostico_ml} · respaldo={respaldo}"
    )
    registrar(caso, ok, detalle)


if __name__ == "__main__":
    CLAVE = leer_clave()
    CLAVE_VALIDA = bool(CLAVE) and not CLAVE.startswith("gsk_TU")

    print("=" * 82)
    print(" MATRIZ DE PRUEBAS DE ESTRÉS · Mantenimiento Predictivo ISOF")
    print("=" * 82)

    probar_caso_ml(1, "Operación Estable", (60, 1.5, 300, 92), "success", "operación normal")
    probar_caso_ml(2, "Desgaste Preventivo", (80, 4.5, 2000, 116), "warning", "mantenimiento preventivo")
    probar_caso_ml(3, "Sobrecarga Crítica", (98, 8.8, 4000, 146), "error", "fallo crítico")

    probar_orden_groq(2, "Orden de trabajo preventiva", (80, 4.5, 2000, 116),
                      ["Folio", "Diagnóstico", "Acciones Técnicas", "EPP"])
    probar_orden_groq(3, "Orden de trabajo urgente", (98, 8.8, 4000, 146),
                      ["Folio", "Diagnóstico", "Acciones Técnicas", "EPP"])

    probar_sin_csv()
    probar_resiliencia_api()

    total, aprobados = len(resultados), sum(1 for r in resultados if r["ok"])
    omitidos = sum(1 for r in resultados if r["omitido"])
    print("-" * 82)
    print(f" RESULTADO: {aprobados}/{total} verificaciones superadas"
          + (f" ({omitidos} omitidas por falta de API Key)" if omitidos else ""))
    print("=" * 82)
    sys.exit(0 if aprobados == total else 1)
