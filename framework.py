'''
Agente financiero/matemático con LangGraph + Groq.
Memoria de corto plazo (checkpointer, por hilo) y largo plazo (store, entre conversaciones).
Requiere: GROQ_API_KEY, TWELVEDATA_API_KEY en variables de entorno.
'''

import math
import os
import requests
import uuid
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


def calculadora(expresion: str) -> str:
    '''Evalúa aritmética de Python de forma restringida (sin builtins, solofunciones de math). Ej: 2**10, 100*1.05, sqrt(16).
    Nunca aceptes expresiones que vengan directo del usuario sin pasar por el agente:
    eval() con builtins bloqueados reduce el riesgo pero no lo elimina.'''

    permitido = {k: getattr(math, k) for k in ("sqrt", "log", "exp", "pi", "e", "sin", "cos")}
    try:
        return str(eval(expresion, {"__builtins__": {}}, permitido))
    except Exception as err:
        return f"ERROR: {err}"


def hora_actual() -> str:
    '''Devuelve la fecha y hora actuales.'''
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def tipo_cambio() -> str:
    '''Devuelve el tipo de cambio USD/MXN actual vía Twelve Data. Retorna un string "ERROR: ..." 
    si falta la API key o si la API falla — el agente debe verificar ese prefijo antes de usar el valor como número.'''
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

@tool
def recordar(hecho: str, runtime: ToolRuntime) -> str:
    """Guarda un hecho duradero sobre el usuario (preferencias, datos personales)
    para recordarlo en TODAS las conversaciones futuras. Ej: 'prefiere respuestas en pesos mexicanos'."""
    runtime.store.put(("memorias", "usuario"), str(uuid.uuid4()), {"hecho": hecho})
    return f"Recordado: {hecho}"

@tool
def buscar_memoria(runtime: ToolRuntime) -> str:
    """Devuelve lo que se sabe del usuario de conversaciones anteriores. Úsala al
    inicio o cuando necesites contexto sobre quién es el usuario."""
    items = runtime.store.search(("memorias", "usuario"))
    if not items:
        return "No hay memorias guardadas todavía."
    return "Sé esto del usuario:\n" + "\n".join(f"- {it.value['hecho']}" for it in items)

def correr_con_traza(pregunta: str):
    '''Corre el agente en modo streaming y muestra pensamiento, acción y observación
    de cada paso. Útil para debug; el loop principal usa agente.invoke() sin traza.'''
    entrada = {"messages": [{"role": "user", "content": pregunta}]}

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

def extraer_y_guardar(mensaje_usuario: str, store):
    """Revisa el mensaje del usuario y guarda hechos duraderos en la memoria de largo plazo. Es una segunda pasada 
    por el LLM, separada de la tool 'recordar': así el agente no tiene que decidir en el momento si algo es 
    'memorizable', y no se le olvida."""
    instruccion = (
        "Extrae de este mensaje cualquier HECHO DURADERO sobre el usuario: su nombre, "
        "qué estudia, sus preferencias estables, datos personales que sigan siendo "
        "ciertos mañana. Ignora preguntas, cálculos o cosas pasajeras.\n"
        "Si NO hay ningún hecho duradero, responde EXACTAMENTE: NADA\n"
        "Si SÍ hay, responde solo el hecho en una frase corta, sin nada más.\n\n"
        f"Mensaje del usuario: {mensaje_usuario}"
    )
    resp = llm.invoke(instruccion)
    hecho = resp.content.strip()

    if hecho and hecho.upper() != "NADA":
        store.put(("memorias", "usuario"), str(uuid.uuid4()), {"hecho": hecho})
        print(f"[memoria] Guardé: {hecho}")

#llm = ChatGoogleGenerativeAI(
#    model="gemini-2.5-flash",
#    google_api_key=os.environ["GEMINI_API_KEY"],
#)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
    max_retries=3,
)

if __name__ == "__main__":

    # Dos bases de datos separadas: memoria.db (checkpointer) guarda el estado
    # de la conversación por hilo; memoria_lp.db (store) guarda hechos sobre el
    # usuario que persisten entre TODOS los hilos.
    with SqliteSaver.from_conn_string("memoria.db") as checkpointer, \
         SqliteStore.from_conn_string("memoria_lp.db") as store:

        store.setup()

        agente = create_agent(
            llm,
            tools=[calculadora, hora_actual, tipo_cambio, precio_accion, recordar, buscar_memoria],
            system_prompt=(
            "Usa las herramientas de inmediato; nunca pidas permiso ni digas que no "
            "tienes acceso. Llama UNA herramienta a la vez y ESPERA su resultado antes "
            "de la siguiente. La calculadora solo acepta números literales, nunca "
            "nombres de variables ni llamadas a funciones: primero obtén cada número "
            "con su herramienta, y solo entonces escribe la operación con esos números. "
            "Al inicio de una conversación o cuando necesites contexto sobre quién es el "
            "usuario, usa 'buscar_memoria'."
            ),
            checkpointer=checkpointer,
            store=store,
        )

        hilos_existentes = {
            c.config["configurable"]["thread_id"]
            for c in checkpointer.list(None)
        }
        if hilos_existentes:
            print("Conversaciones existentes:", ", ".join(hilos_existentes))
        else:
            print("No hay conversaciones previas todavía.")

        nombre_conv = input("¿Qué conversación quieres abrir? (Enter para 'default'): ").strip()
        if not nombre_conv:
            nombre_conv = "default"
        
        config = {"configurable": {"thread_id": nombre_conv}}

        print("================================================================")
        print("🤖 Agente Financiero y Matemático Iniciado.")
        print("💡 Escribe tu pregunta y presiona Enter.")
        print("🚪 Escribe 'salir', 'exit' o 'quit' para terminar la conversación.")
        print("================================================================\n")

        while True:
            pregunta = input("Tú: ")

            comando_salida = pregunta.strip().lower()
            if comando_salida in ["salir", "exit", "quit", "adios", "adiós"]:
                print("\nAgente: ¡Nos vemos! Apagando el sistema...")
                break

            if not pregunta.strip():
                continue

            try:
                respuesta = agente.invoke(
                    {"messages": [{"role": "user", "content": pregunta}]},
                    config,
                )
                print(f"Agente: {respuesta['messages'][-1].content}\n")

                extraer_y_guardar(pregunta, store)

            except Exception as e:
                print(f"\n Ocurrió un error: {e}\n")