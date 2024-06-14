import os
from dotenv import load_dotenv, dotenv_values

load_dotenv()

gg_api_key = os.getenv("GOOGLE_API_KEY")
gg_model_name = "gemini-pro"
embedding_model_file = "models/all-MiniLM-L6-v2-f16.gguf"
file_path = "data/test.txt"