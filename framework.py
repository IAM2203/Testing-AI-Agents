import math
import os
import requests
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, ToolMessage

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
    
#llm = ChatGoogleGenerativeAI(
#    model="gemini-2.5-flash",
#    google_api_key=os.environ["GEMINI_API_KEY"],
#)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

agente = create_agent(
    llm,
    tools=[calculadora, hora_actual, tipo_cambio, precio_accion],
    system_prompt=(
        "Usa las herramientas de inmediato; nunca pidas permiso ni digas que no "
        "tienes acceso. Llama UNA herramienta a la vez y ESPERA su resultado antes "
        "de la siguiente. La calculadora solo acepta números literales, nunca "
        "nombres de variables ni llamadas a funciones: primero obtén cada número "
        "con su herramienta, y solo entonces escribe la operación con esos números."
    ),
)

def correr_con_traza(pregunta: str):
    entrada = {"messages": [{"role": "user", "content": pregunta}]}

    # stream_mode="values" -> emite el estado COMPLETO tras cada paso;
    # miramos el mensaje más nuevo de cada estado.
    for estado in agente.stream(entrada, stream_mode="values"):
        msg = estado["messages"][-1]

        if isinstance(msg, AIMessage) and msg.tool_calls:
            if msg.content:                      # algunos modelos razonan en texto aquí
                print(f"  PENSAMIENTO -> {msg.content}")
            for tc in msg.tool_calls:
                print(f"  ACCIÓN      -> {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"  OBSERVACIÓN -> {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"\nRESPUESTA FINAL: {msg.content}")

if __name__ == "__main__":
    correr_con_traza("Si compro 5 acciones de MSFT, ¿cuánto gastaría en pesos?")
#    resp = agente.invoke({
#        "messages": [
#            {"role": "user", "content": "Si compro 5 acciones de MSFT, ¿cuánto gastaría en pesos?"}
#        ]
#    })
#    print(resp["messages"][-1].content)