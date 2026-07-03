'''
Agente con function calling nativo de Gemini: a diferencia de la versión
manual (ReAct por texto plano), aquí el SDK maneja todo el ciclo de
llamar herramientas, esperar resultados y decidir el siguiente paso.
Solo se le pasan las funciones de Python directamente en `tools`.
Requiere: GEMINI_API_KEY (vía variables de entorno que use genai.Client()),
TWELVEDATA_API_KEY.
'''

import math
import os
import requests
import time
from datetime import datetime
from google import genai
from google.genai import types
from google.genai import errors

client = genai.Client()
MODELO = "gemini-2.5-flash"


def calculadora(expresion: str) -> str:
    '''Evalúa aritmética de Python. Ej: 2**10, 100*1.05, sqrt(16).
    Retorna un string "ERROR: ..." si la expresión es inválida.'''
    permitido = {k: getattr(math, k) for k in ("sqrt", "log", "exp", "pi", "e", "sin", "cos")}
    try:
        return str(eval(expresion, {"__builtins__": {}}, permitido))
    except Exception as err:
        return f"ERROR: {err}"


def hora_actual() -> str:
    '''Devuelve la fecha y hora actuales.'''
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tipo_cambio() -> str:
    '''Devuelve el tipo de cambio USD/MXN actual vía Twelve Data.
    Retorna un string "ERROR: ..." si falta la API key o si la API falla.'''
    API_KEY = os.environ.get("TWELVEDATA_API_KEY")

    if not API_KEY:
        return "ERROR: falta la variable de entorno TWELVEDATA_API_KEY"
    try:
        url = "https://api.twelvedata.com/price"
        params = {"symbol": "USD/MXN", "apikey": API_KEY}
        r = requests.get(url, params=params, timeout=10)
        datos = r.json()
        if "price" in datos:
            return datos["price"]
        return f"ERROR: {datos.get('message', datos)}"
    except Exception as err:
        return f"ERROR: {err}"


def precio_accion(ticker: str) -> str:
    '''Devuelve el precio actual de una acción en USD (ej. AAPL, NVDA) vía Twelve Data.
    Retorna un string "ERROR: ..." si el ticker no existe o la API falla.'''
    ticker = ticker.strip().upper()
    API_KEY = os.environ.get("TWELVEDATA_API_KEY")

    if not API_KEY:
        return "ERROR: falta la variable de entrono TWELVEDATA_API_KEY"
    try:
        url = "https://api.twelvedata.com/price"
        params = {"symbol": ticker, "apikey": API_KEY}
        r = requests.get(url, params=params, timeout=10)
        datos = r.json()
        if "price" in datos:
            return datos["price"]
        return f"ERROR: {datos.get('message', datos)}"
    except Exception as err:
        return f"ERROR: {err}"


def agente(objetivo: str) -> str:
    '''Envía el objetivo del usuario a Gemini con function calling nativo
    habilitado. El modelo decide solo qué herramientas llamar, en qué orden,
    y cuándo detenerse — todo eso pasa dentro de generate_content(), sin
    loop manual ni scratchpad. Retorna directamente resp.text con la
    respuesta final ya generada.
    Nota: a diferencia de la versión ReAct manual, aquí no hay reintentos
    ante errores 429/5xx; una llamada que falle se propaga tal cual.'''
    config = types.GenerateContentConfig(
        # Pasar las funciones de Python directamente (no un dict de schemas)
        # es lo que activa el function calling automático: el SDK infiere
        # nombre, parámetros y docstring de cada función para describírselas
        # al modelo, y ejecuta la que el modelo pida sin que tengamos que
        # parsear texto ni hacer despacho manual.
        tools=[calculadora, hora_actual, tipo_cambio, precio_accion],
        system_instruction=(
            "Tienes herramientas para datos en tiempo real (precios de acciones, "
            "tipo de cambio, hora) y para cálculos. Cuando la pregunta requiera "
            "esos datos, USA las herramientas de inmediato en vez de decir que no "
            "tienes acceso o de pedir permiso. Nunca preguntes si debes consultar "
            "algo: hazlo directamente y entrega el resultado final. Encadena varias "
            "herramientas si hace falta."
        ),
    )
    resp = client.models.generate_content(
        model=MODELO,
        contents=objetivo,
        config=config,
    )

    return resp.text


if __name__ == "__main__":
    pregunta = (
        "Si compro 5 acciones de MSFT, ¿cuánto gastaría en pesos?"
    )
    print("\n=========== RESPUESTA FINAL ===========")
    print(agente(pregunta))