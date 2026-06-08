# ============================================================
#  Exam Doubt Solver — Ollama + LangChain + Streamlit
#  
#  HOW IT WORKS (simple explanation for interviews):
#  1. Streamlit   → builds the web UI (chat interface, sidebar)
#  2. LangChain   → manages prompt building + chat history
#  3. Ollama      → runs the AI model locally on your machine
#
#  Flow: User types question
#        → LangChain builds prompt (system + history + question)
#        → Ollama's LLM answers it
#        → Streamlit displays the answer
# ============================================================

import streamlit as st
from langchain_ollama import ChatOllama                          # Ollama LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Prompt builder
from langchain_core.messages import HumanMessage, AIMessage     # Message types

# ── Page Setup ───────────────────────────────────────────────
st.set_page_config(
    page_title="Exam Doubt Solver",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] * { color: #e0e0ff !important; }

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    margin-bottom: 8px;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(99, 102, 241, 0.2);
    border-color: rgba(99, 102, 241, 0.4);
}

[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(99, 102, 241, 0.5) !important;
    border-radius: 12px !important;
    color: #e0e0ff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
}

.stMarkdown, p, span, label { color: #c4c4e0 !important; }
h1, h2, h3 { color: #e0e0ff !important; }

code {
    background: rgba(0,0,0,0.4) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #a78bfa !important;
}
pre {
    background: rgba(0,0,0,0.5) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
}

.main-title {
    text-align: center;
    background: linear-gradient(135deg, #a78bfa, #6366f1, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: rgba(196,196,224,0.7) !important;
    font-size: 0.95rem;
    margin-bottom: 20px;
}

.subject-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px;
}

.info-box {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── Subject System Prompts ────────────────────────────────────
# Each subject has a custom system prompt so the AI behaves
# like a subject-specific tutor
SUBJECT_PROMPTS = {
    "General 📚": """You are a helpful and friendly exam doubt solver for CSE students.
Answer questions clearly. Use simple examples. For code questions, provide clean commented code.
Always keep explanations simple and exam-focused.""",

    "Data Structures & Algorithms 🧮": """You are a DSA tutor for CSE students.
Always mention Time Complexity and Space Complexity.
Use step-by-step explanations. Show Python or C++ code examples.
Draw ASCII diagrams for trees/graphs when helpful.""",

    "Operating Systems ⚙️": """You are an OS tutor for CSE students.
Explain process scheduling, memory management, deadlocks clearly.
Use real-world examples (like how Linux works).
Structure answers with headings for exam readiness.""",

    "DBMS 🗄️": """You are a DBMS tutor for CSE students.
Help with SQL, normalization, ER diagrams, transactions.
Always show SQL examples. Explain 1NF/2NF/3NF/BCNF with examples.
Keep answers exam-oriented.""",

    "Computer Networks 🌐": """You are a Computer Networks tutor for CSE students.
Explain OSI/TCP-IP models, protocols, subnetting clearly.
Show subnetting calculations step by step.
Use ASCII diagrams for network flows.""",

    "OOP & Java/C++ 💻": """You are an OOP tutor for CSE students.
Explain encapsulation, inheritance, polymorphism, abstraction with code.
Provide clean, well-commented Java or C++ examples.
Walk through logic step by step.""",

    "Mathematics 📐": """You are a math tutor for CSE students.
Help with Discrete Math, Linear Algebra, Probability, Calculus.
Show step-by-step solutions. Never skip intermediate steps.
Relate math to CS applications when possible.""",

    "Theory of Computation 🔄": """You are a TOC tutor for CSE students.
Help with DFA, NFA, Regular Expressions, CFG, PDA, Turing Machines.
Draw state diagrams using ASCII art.
Show step-by-step derivations for grammar problems.""",
}


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Exam Doubt Solver")
    st.markdown("---")

    # Model selector — Ollama supports multiple models
    st.markdown("### 🤖 AI Model")
    model_name = st.selectbox(
        "Choose Ollama model",
        ["llama3.2", "llama3.1", "mistral", "gemma2", "phi3"],
        help="Make sure you've pulled this model via: ollama pull <model>"
    )

    st.markdown("---")

    # Subject selector
    st.markdown("### 📖 Subject")
    subject = st.selectbox("Choose your subject", list(SUBJECT_PROMPTS.keys()))

    st.markdown("---")

    # Explanation level
    st.markdown("### 🎯 Explanation Level")
    level = st.select_slider(
        "Detail level",
        options=["Quick Answer", "Normal", "Deep Explanation"],
        value="Normal"
    )

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")

    # Help box
    st.markdown("""
    <div class='info-box'>
    🖥️ <b>Ollama Setup:</b><br>
    1. Install: <a href='https://ollama.com' style='color:#a78bfa'>ollama.com</a><br>
    2. Run: <code>ollama pull llama3.2</code><br>
    3. Ollama starts automatically!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; color:rgba(196,196,224,0.4); font-size:0.75rem; margin-top:16px'>
    100% Local • No API Key • Free Forever 🚀
    </div>
    """, unsafe_allow_html=True)


# ── Main UI ───────────────────────────────────────────────────
st.markdown("<div class='main-title'>🎓 Exam Doubt Solver</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by Ollama — 100% offline, no API key needed</div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center'><span class='subject-badge'>📌 {subject}</span><span class='subject-badge'>🤖 {model_name}</span><span class='subject-badge'>📊 {level}</span></div>", unsafe_allow_html=True)
st.markdown("")


# ── Session State (stores chat history) ──────────────────────
# This is how Streamlit remembers data between reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_subject" not in st.session_state:
    st.session_state.last_subject = subject

# Reset chat when subject changes
if st.session_state.last_subject != subject:
    st.session_state.chat_history = []
    st.session_state.last_subject = subject


# ── Core Functions ────────────────────────────────────────────

def get_llm(model: str):
    """
    Creates and returns the Ollama LLM.
    ChatOllama connects to your locally running Ollama server.
    """
    return ChatOllama(
        model=model,
        temperature=0.7,   # 0 = focused/deterministic, 1 = creative
        streaming=True     # Stream tokens one by one (like ChatGPT typing effect)
    )


def build_system_prompt(subject: str, level: str) -> str:
    """
    Combines the subject prompt with explanation level instruction.
    This tells the AI HOW to answer (short vs detailed).
    """
    base = SUBJECT_PROMPTS[subject]
    level_map = {
        "Quick Answer":      "\nIMPORTANT: Keep answers SHORT — 2-4 sentences max unless code is needed.",
        "Normal":            "\nGive balanced answers. Use bullet points for clarity.",
        "Deep Explanation":  "\nGive DETAILED explanations. Cover theory, examples, edge cases. Be thorough."
    }
    return base + level_map[level]


def get_response(question: str, history: list, subject: str, level: str, llm):
    """
    Builds the full prompt and gets a streaming response from Ollama.
    
    LangChain's ChatPromptTemplate combines:
    - System prompt (subject + level instructions)
    - Chat history (so AI remembers previous messages)
    - Current question
    """
    # Build the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", build_system_prompt(subject, level)),  # Role/behavior
        MessagesPlaceholder(variable_name="chat_history"),# Previous messages
        ("human", "{question}"),                          # Current question
    ])

    # Chain: prompt → llm (this is LangChain's LCEL syntax)
    chain = prompt | llm

    # Stream the response
    return chain.stream({
        "chat_history": history,
        "question": question,
    })


# ── Display Chat History ──────────────────────────────────────
if not st.session_state.chat_history:
    with st.chat_message("assistant"):
        st.markdown(f"""
        👋 **Hello! I'm your Exam Doubt Solver.**

        I'm running **locally via Ollama** — no internet or API key needed!

        Currently set to: **{subject}** | Level: **{level}** | Model: **{model_name}**

        Ask me anything — concepts, code, formulas, previous year questions! 🚀
        """)
else:
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)


# ── Chat Input ────────────────────────────────────────────────
user_input = st.chat_input("Ask your doubt here...")

if user_input:
    # Show user's message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and stream AI response
    with st.chat_message("assistant"):
        try:
            llm = get_llm(model_name)
            stream = get_response(
                user_input,
                st.session_state.chat_history,
                subject,
                level,
                llm
            )
            # st.write_stream handles the streaming display
            response_text = st.write_stream(stream)

        except Exception as e:
            error = str(e)
            if "connection" in error.lower() or "refused" in error.lower():
                response_text = "❌ **Ollama is not running!** Open a terminal and run: `ollama serve`"
            elif "not found" in error.lower() or "pull" in error.lower():
                response_text = f"❌ **Model '{model_name}' not found!** Run: `ollama pull {model_name}`"
            else:
                response_text = f"❌ **Error:** {error}"
            st.error(response_text)

    # Save both messages to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=str(response_text)))

    # Keep only last 20 messages to avoid memory overflow
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]