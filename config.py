from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY') 