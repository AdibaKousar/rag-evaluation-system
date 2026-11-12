
from openai import OpenAI

# Connect to local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Send a test question
response = client.chat.completions.create(
    model="tinyllama:latest",
    messages=[
        {
            "role": "user",
            "content": "Explain machine learning in two simple sentences."
        }
    ],
    temperature=0
)

# Display response
print("\nTINYLLAMA RESPONSE:")
print(response.choices[0].message.content)
