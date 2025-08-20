import time
import psutil
import csv
import streamlit as st 
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from business_info import info 

# ====== GPU MONITORAMENTO ======
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_monitoring = True
except:
    gpu_monitoring = False
    st.warning("⚠ Monitoramento de GPU NVIDIA indisponível (biblioteca pynvml não encontrada).")

# ====== CONFIGURAÇÃO CSV ======
CSV_FILE = "metricas_chatbot.csv"
try:
    open(CSV_FILE, "x").write(
        "timestamp,latencia_s,tokens_entrada,tokens_saida,throughput_tps,"
        "cpu_percent,memoria_mb,gpu_percent,gpu_memoria_mb\n"
    )
except FileExistsError:
    pass

st.title("Asistente Virtual")

# ====== INICIALIZAR ESTADO ======
if "messages" not in st.session_state:
    st.session_state.messages = []

if "first_message" not in st.session_state:
    st.session_state.first_message = True

if "context" not in st.session_state:
    st.session_state.context = ""

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Mensaje de bienvenida
if st.session_state.first_message:
    welcome_message = "Hola, ¿cómo puedo ayudarte?"
    with st.chat_message("assistant"):
        st.markdown(welcome_message)
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    st.session_state.first_message = False

# Inicializar el modelo y el chain solo una vez
if "chain" not in st.session_state:
    template = """
    Answer the question below in Spanish.

    Here is business of the conversation:
    {business_info}

    {context}

    Question: {question}

    Answer:
    """
    model = OllamaLLM(model="llama3.1:8b")
    prompt = ChatPromptTemplate.from_template(template)
    st.session_state.chain = prompt | model

# ====== FUNÇÃO PARA COLETAR MÉTRICAS ======
def medir_metricas(user_input):
    # CPU e RAM antes
    cpu_inicio = psutil.cpu_percent(interval=None)
    mem_inicio = psutil.virtual_memory().used / (1024 * 1024)

    # GPU antes
    gpu_percent = None
    gpu_memoria = None
    if gpu_monitoring:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_percent = util.gpu
        gpu_memoria = mem.used / (1024 * 1024)

    tokens_entrada = len(user_input.split())

    # Tempo de resposta
    inicio = time.time()
    result = st.session_state.chain.invoke({
        "business_info": info,
        "context": st.session_state.context,
        "question": user_input
    })
    fim = time.time()

    # CPU e RAM depois
    cpu_fim = psutil.cpu_percent(interval=None)
    mem_fim = psutil.virtual_memory().used / (1024 * 1024)

    # GPU depois
    if gpu_monitoring:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_percent = (gpu_percent + util.gpu) / 2
        gpu_memoria = (gpu_memoria + (mem.used / (1024 * 1024))) / 2

    tokens_saida = len(result.split())
    latencia = fim - inicio
    throughput = tokens_saida / latencia if latencia > 0 else 0
    cpu_percent = (cpu_inicio + cpu_fim) / 2
    memoria_mb = (mem_inicio + mem_fim) / 2

    # Salvar no CSV
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            round(latencia, 3),
            tokens_entrada,
            tokens_saida,
            round(throughput, 2),
            round(cpu_percent, 1),
            round(memoria_mb, 1),
            round(gpu_percent, 1) if gpu_percent is not None else "",
            round(gpu_memoria, 1) if gpu_memoria is not None else ""
        ])

    return result, {
        "latencia_s": round(latencia, 3),
        "tokens_entrada": tokens_entrada,
        "tokens_saida": tokens_saida,
        "throughput_tps": round(throughput, 2),
        "cpu_percent": round(cpu_percent, 1),
        "memoria_mb": round(memoria_mb, 1),
        "gpu_percent": round(gpu_percent, 1) if gpu_percent is not None else "N/A",
        "gpu_memoria_mb": round(gpu_memoria, 1) if gpu_memoria is not None else "N/A"
    }

# ====== ENTRADA DO USUÁRIO ======
if user_input := st.chat_input("¿Cómo puedo ayudarte?"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Generando respuesta..."):
            result, metricas = medir_metricas(user_input)
            st.markdown(result)

            # Mostrar métricas
            st.markdown("### 📊 Métricas desta respuesta")
            st.json(metricas)

    st.session_state.messages.append({"role": "assistant", "content": result})

    # Actualizar contexto
    st.session_state.context += f"You: {user_input}\nBot: {result}\n"

