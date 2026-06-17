import math
import os
import requests
import time
from datetime import datetime
from google import genai
from google.genai import types
from google.genai import errors

client = genai.Client()
MODELO = "gemini-2.5-flash-lite"
def calculadora(expresion: str) -> str:
    permitido = {k: getattr(math, k) for k in ("sqrt", "log", "exp", "pi", "e", "sin", "cos")}
    try:
        return str(eval(expresion, {"__builtins__": {}}, permitido))
    except Exception as err:
        return f"ERROR: {err}"


def hora_actual(_: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def tipo_cambio(_:str = "") -> str:
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

HERRAMIENTAS = {
    "calculadora": calculadora,
    "hora_actual": hora_actual,
    "tipo_cambio": tipo_cambio,
    "precio_accion": precio_accion,
}

SYSTEM = """Eres un agente que resuelve tareas razonando paso a paso.
Tienes estas herramientas disponibles:
- calculadora(expresion): evalúa aritmética de Python. Ej: 2**10, 100*1.05, sqrt(16).
- hora_actual(): devuelve la fecha y hora actuales.
- tipo_cambio(): devuelve el valor del tipo de cambio de dolar a peso actual.
- precio_accion(ticker): devuelve el precio de una acción e USD. Ej, AAPL, NVDA.

Responde SIEMPRE en uno de estos dos formatos EXACTOS:

(A) Si necesitas usar una herramienta:
Thought: <tu razonamiento breve>
Action: <nombre exacto de la herramienta>
Action Input: <el argumento, sin comillas ni texto extra>

(B) Si ya tienes la respuesta para el usuario:
Thought: <tu razonamiento breve>
Final Answer: <la respuesta>

Reglas:
- No inventes el resultado de una herramienta. Detente justo después de
  'Action Input:' y espera a recibir la 'Observation:'.
- Usa una sola herramienta por paso.
- Cuando tengas el resultado numérico final, NO uses una vuelta extra para
formatear. En la MISMA vuelta en que ya tienes el dato, da directamente el
Final Answer escribiendo las cifras de dinero con comas de miles y dos
decimales (ej: 18987.763 -> $18,987.76).
"""

def agente(objetivo: str, max_pasos: int = 6) -> str:
    scratchpad = f"Tarea: {objetivo}\n"

    for paso in range(1, max_pasos + 1):

        for intento in range(4):
            try:
                respuesta = client.models.generate_content(
                    model=MODELO,
                    contents=scratchpad,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        stop_sequences=["Observation:"],
                        temperature=0,
                    ),
                )
                break

            except errors.ServerError as e:
                if intento == 3:
                    raise
                espera = 2 ** intento
                print(f"Servidor ocupado ({e.code}). Reintento en {espera}s...")
                time.sleep(espera)

            except errors.ClientError as e:
                mensaje = str(e)
                if e.code == 429 and "PerDay" in mensaje:
                    raise RuntimeError(
                        "Agotaste la cuota DIARIA de este modelo. "
                        "Cambia MODELO a 'gemini-2.5-flash-lite' o espera al reinicio (medianoche PT)."
                    )
                elif e.code == 429 and intento < 3:
                    print("Cuota por minuto agotada (429). Espero 20s...")
                    time.sleep(20)
                else:
                    raise

        texto = respuesta.text.strip()
        print(f"\n--- Paso {paso} ---\n{texto}")
        scratchpad += texto + "\n"

        if "Final Answer:" in texto:
            return texto.split("Final Answer:", 1)[1].strip()

        if "Action:" in texto and "Action Input:" in texto:
            nombre = texto.split("Action:", 1)[1].split("\n", 1)[0].strip()
            arg = texto.split("Action Input:", 1)[1].strip().strip("`")

            herramienta = HERRAMIENTAS.get(nombre)
            observacion = herramienta(arg) if herramienta else f"No existe la herramienta '{nombre}'."

            print(f"Observation: {observacion}")
            scratchpad += f"Observation: {observacion}\n"

        else:
            scratchpad += "Observation: Formato inválido. Usa Action/Action Input o Final Answer.\n"

    return "Se alcanzó el máximo de pasos sin respuesta final."


if __name__ == "__main__":
    pregunta = (
        "Si compro 5 acciones de MSFT, ¿cuánto gastaría en pesos?"
    )
    print("\n=========== RESPUESTA FINAL ===========")
    print(agente(pregunta))
