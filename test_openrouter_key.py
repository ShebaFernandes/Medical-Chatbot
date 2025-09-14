import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Get API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

print(f"API Key: {OPENROUTER_API_KEY[:15] if OPENROUTER_API_KEY else 'None'}...")

if not OPENROUTER_API_KEY:
    print("❌ No API key found!")
    exit()

# Test the key
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8080",
    "X-Title": "Medical Chatbot"
}

data = {
    "model": "openai/gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
}

try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}")
    
    if response.status_code == 200:
        print("✅ OpenRouter API key works!")
    else:
        print("❌ API key issue")
        
except Exception as e:
    print(f"❌ Network error: {e}")