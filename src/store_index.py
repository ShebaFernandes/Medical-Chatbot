from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 1️⃣ Load and prepare data
# ✅ Use a folder path, not a single file
from langchain.document_loaders import PyPDFLoader

# Path to your PDF
pdf_path = r"C:\Users\hp\OneDrive\Desktop\Medical Chatbot\Medical-Chatbot\src\data\Medical_book.pdf"

# Load PDF
loader = PyPDFLoader(pdf_path)
extracted_data = loader.load()

# Show results
print(len(extracted_data))   # Number of chunks
print(extracted_data[0])     # First chunk

extracted_data

filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

print(f"✅ Loaded {len(text_chunks)} text chunks.")
if len(text_chunks) > 0:
    print(f"🔎 Sample chunk: {text_chunks[0].page_content[:200]}...")

# 2️⃣ Load embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Test embedding dimension
sample_vector = embeddings.embed_query("Hello world")
print(f"✅ Sample embedding length: {len(sample_vector)} (should be 384)")

# 3️⃣ Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
print("✅ Connected to Pinecone index:", index_name)

# 4️⃣ Create or connect to vector store and add documents
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

if len(text_chunks) > 0:
    docsearch.add_documents(text_chunks)
    print("✅ Documents successfully added to Pinecone!")
else:
    print("⚠️ No documents to add. Please check your PDF folder or filter logic.")
