from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI   # ✅ only keep this import
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables from .env
load_dotenv()

# Get API keys with validation
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Check if API keys exist
if not PINECONE_API_KEY:
    print("❌ PINECONE_API_KEY not found in .env file")
    exit(1)
if not OPENROUTER_API_KEY:
    print("❌ OPENROUTER_API_KEY not found in .env file")
    exit(1)

print(f"✅ API Keys loaded successfully")

# Set Pinecone key in environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embeddings
embeddings = download_embeddings()

# Pinecone index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, 
    embedding=embeddings
)

# Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Configure ChatOpenAI to use OpenRouter
chat = ChatOpenAI(
    model="openai/gpt-4o-mini",                 # ✅ model
    temperature=0,
    api_key=OPENROUTER_API_KEY,                 # ✅ correct arg
    base_url="https://openrouter.ai/api/v1"     # ✅ correct arg
)

# Create chain
question_answer_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat_route():
    try:
        msg = request.form["msg"]
        print(f"📩 Received message: {msg}")
        response = rag_chain.invoke({"input": msg})
        print(f"✅ Response generated successfully")
        return str(response["answer"])
    except Exception as e:
        print(f"❌ Error in chat_route: {str(e)}")
        return f"Error: Unable to process your request. Please check the server logs."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)