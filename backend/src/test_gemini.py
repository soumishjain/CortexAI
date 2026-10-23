import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

if not api_key:
    raise ValueError("Google API key is not set. Please set the GOOGLE_API_KEY environment variable.")

model = ChatGoogleGenerativeAI(
    model = model_name,
    google_api_key=api_key,
    temperature=0
)

response = model.invoke(
    "Introduce yourself as the AI engine of CortexAI"
)

print(response.content)