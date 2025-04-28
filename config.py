from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY') 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')