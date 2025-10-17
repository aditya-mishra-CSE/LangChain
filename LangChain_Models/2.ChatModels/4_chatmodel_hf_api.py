# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India")
# print(os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))  # Should print your token
# print(result.content)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Get API key
api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not api_key:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env")

# Use HuggingFaceEndpoint (calls hosted API, no local PyTorch needed)
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small",  # lightweight hosted model
    task="text-generation",
    huggingfacehub_api_token=api_key
)

# Wrap as a Chat model
model = ChatHuggingFace(llm=llm)

# Ask a question
response = model.invoke("What is the capital of India?")
print("Answer:", response.content)
