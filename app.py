import os
import time
from flask import Flask, render_template, request, redirect, session, url_for, Response
from pathlib import Path

# LangChain + Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

# ----------------------------
# Users
# ----------------------------
USERS = {"admin": "admin123", "user": "user123"}

# ----------------------------
# Globals (lazy loaded)
# ----------------------------
vectorstore = None
embeddings = None

# ----------------------------
# Load documents safely
# ----------------------------
def load_documents(folder="data"):
    texts = []
    folder_path = Path(folder)
    folder_path.mkdir(exist_ok=True)

    for file in folder_path.glob("*.txt"):
        content = file.read_text(encoding="utf-8").strip()
        if content:
            texts.append(content)

    return texts

def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def init_vectorstore():
    global vectorstore, embeddings

    if vectorstore is not None:
        return

    docs = load_documents()
    if not docs:
        print("⚠ No documents found in data/. Vector search disabled.")
        vectorstore = None
        return

    all_text = "\n\n".join(docs)
    chunks = chunk_text(all_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("✅ Vector store initialized")

# ----------------------------
# Ollama (CPU-safe)
# ----------------------------
llm = ChatOllama(
    model="phi",
    temperature=0.3,
    timeout=120,
    num_gpu=0
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI document assistant. Answer only from the provided context."),
    ("human", "Context:\n{context}\n\nConversation:\n{history}\n\nQuestion: {question}")
])

chain = prompt | llm

# ----------------------------
# AUTH ROUTES
# ----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if USERS.get(request.form["username"]) == request.form["password"]:
            session["user"] = request.form["username"]
            session["chat"] = []
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in USERS:
            return render_template("signup.html", error="Username already exists")
        USERS[username] = password
        return redirect(url_for("login"))
    return render_template("signup.html")

# ----------------------------
# Streaming Chat
# ----------------------------
@app.route("/stream", methods=["POST"])
def stream():
    if "user" not in session:
        return "Unauthorized", 401

    init_vectorstore()

    question = request.form.get("question", "").strip()
    if not question:
        return "Empty question", 400

    history = "\n".join(session.get("chat", []))

    context = ""
    if vectorstore:
        docs_found = vectorstore.similarity_search(question, k=3)
        context = "\n".join(d.page_content for d in docs_found)

    try:
        response = chain.invoke({
            "context": context,
            "question": question,
            "history": history
        })
        answer = response.content
    except Exception as e:
        answer = f"⚠ Error generating response: {str(e)}"

    # Save history
    session["chat"].append(f"User: {question}")
    session["chat"].append(f"AI: {answer}")

    def generate():
        for word in answer.split():
            yield word + " "
            time.sleep(0.02)

    return Response(generate(), mimetype="text/plain")

# ----------------------------
# UI
# ----------------------------
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", chat=session.get("chat", []))

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting AI Document Assistant...")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
