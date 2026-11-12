
from openai import OpenAI
from ragas.llms import llm_factory


# Step 1: Connect to Ollama
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# Step 2: Create RAGAS evaluator LLM
evaluator_llm = llm_factory(
    "tinyllama:latest",
    provider="openai",
    client=client
)

# Step 3: Verify initialization
print("RAGAS evaluator initialized successfully!")
print("Evaluator model: TinyLlama")
