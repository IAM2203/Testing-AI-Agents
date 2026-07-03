'''
Agente ReAct manual (sin framework) usando la API de Gemini directamente.
El "razonamiento" vive en un scratchpad de texto plano que se le reenvía
al modelo en cada paso; no hay memoria entre llamadas a agente().
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
MODELO = "gemini-2.5-flash-lite"


def calculadora(expresion: str) -> str:
    '''Evalúa aritmética de Python de forma restringida (sin builtins, solo
    funciones de math). Ej: 2**10, 100*1.05, sqrt(16).
    Retorna un string "ERROR: ..." si la expresión es inválida.'''
    permitido = {k: getattr(math, k) for k in ("sqrt", "log", "exp", "pi", "e", "sin", "cos")}
    try:
        return str(eval(expresion, {"__builtins__": {}}, permitido))
    except Exception as err:
        return f"ERROR: {err}"


def hora_actual(_: str = "") -> str:
    '''Devuelve la fecha y hora actuales.
    Recibe un argumento posicional sin usar solo para tener la misma firma
    que las demás herramientas: HERRAMIENTAS las llama todas como herramienta(arg).'''
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tipo_cambio(_: str = "") -> str:
    '''Devuelve el tipo de cambio USD/MXN actual vía Twelve Data.
    Retorna un string "ERROR: ..." si falta la API key o si la API falla.
    Recibe un argumento sin usar por la misma razón que hora_actual.'''
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


# Tabla de despacho: mapea el nombre de herramienta que el modelo escribe en
# "Action:" a la función real. Si el modelo inventa un nombre que no está
# aquí, agente() lo detecta con HERRAMIENTAS.get() y responde con un error
# en vez de fallar con un KeyError.
HERRAMIENTAS = {
    "calculadora": calculadora,
    "hora_actual": hora_actual,
    "tipo_cambio": tipo_cambio,
    "precio_accion": precio_accion,
}

# Prompt de sistema que define el protocolo ReAct: el modelo debe responder
# en texto con el formato exacto Thought/Action/Action Input o
# Thought/Final Answer. No hay function calling nativo aquí — todo el
# parsing del siguiente paso depende de que el modelo respete este formato
# al pie de la letra, por eso las reglas son tan explícitas.
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
    '''Corre el loop ReAct manual hasta max_pasos veces o hasta obtener un
    "Final Answer:". El "estado" del agente es únicamente el scratchpad de
    texto que se reenvía completo en cada llamada — no hay memoria entre
    llamadas distintas a agente(), a diferencia de la versión con LangGraph.
    Retorna el string de la respuesta final, o un mensaje de agotamiento de
    pasos si el modelo nunca llega a Final Answer.'''
    scratchpad = f"Tarea: {objetivo}\n"

    for paso in range(1, max_pasos + 1):

        # Reintentos con manejo distinto según el tipo de error de la API:
        # - ServerError (5xx): backoff exponencial, el servidor está ocupado
        #   y puede recuperarse solo.
        # - ClientError 429 con cuota DIARIA agotada: no tiene caso
        #   reintentar, hay que avisar y detener la ejecución.
        # - ClientError 429 con cuota POR MINUTO agotada: sí se recupera
        #   pronto, se espera un tiempo fijo (la ventana de rate limit).
        for intento in range(4):
            try:
                respuesta = client.models.generate_content(
                    model=MODELO,
                    contents=scratchpad,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        # Corta la generación antes de que el modelo alucine
                        # su propia "Observation:" — el resultado real de la
                        # herramienta se agrega después, fuera del modelo.
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
            # Parsing por texto plano: frágil por diseño, es el costo de no
            # usar function calling nativo. Si el modelo se desvía del
            # formato exacto, cae en el "else" de abajo en vez de crashear.
            nombre = texto.split("Action:", 1)[1].split("\n", 1)[0].strip()
            arg = texto.split("Action Input:", 1)[1].strip().strip("`")

            herramienta = HERRAMIENTAS.get(nombre)
            observacion = herramienta(arg) if herramienta else f"No existe la herramienta '{nombre}'."

            print(f"Observation: {observacion}")
            # La observación se agrega al scratchpad manualmente — así el
            # modelo la "ve" en la siguiente llamada sin haberla generado él.
            scratchpad += f"Observation: {observacion}\n"

        else:
            # Si el modelo no siguió ninguno de los dos formatos esperados,
            # se le avisa en el propio scratchpad en vez de fallar el loop.
            scratchpad += "Observation: Formato inválido. Usa Action/Action Input o Final Answer.\n"

    return "Se alcanzó el máximo de pasos sin respuesta final."


if __name__ == "__main__":
    pregunta = (
        "Si compro 5 acciones de MSFT, ¿cuánto gastaría en pesos?"
    )
    print("\n=========== RESPUESTA FINAL ===========")
    print(agente(pregunta))