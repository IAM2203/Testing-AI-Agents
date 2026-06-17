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
    '''Evalúa aritmética de Python. Ej: 2**10, 100*1.05, sqrt(16).'''
    permitido = {k: getattr(math, k) for k in ("sqrt", "log", "exp", "pi", "e", "sin", "cos")}
    try:
        return str(eval(expresion, {"__builtins__": {}}, permitido))
    except Exception as err:
        return f"ERROR: {err}"


def hora_actual() -> str:
    '''Devuelve la fecha y hora actuales.'''
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def tipo_cambio() -> str:
    '''devuelve el valor del tipo de cambio de dolar a peso actual.'''
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
    '''Devuelve el precio de una acción e USD. Ej, AAPL, NVDA.'''
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
    config = types.GenerateContentConfig(
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
