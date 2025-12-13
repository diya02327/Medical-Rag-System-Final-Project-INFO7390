"""
Test OpenAI initialization - NO PROXIES
"""
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize WITHOUT any proxy arguments
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test
print("Testing OpenAI client...")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say 'OpenAI works!'"}],
    max_tokens=10
)

print(response.choices[0].message.content)
print("✅ OpenAI client works perfectly!")