"""
Configuration file for the Streamlit app
Update the API_URL with your deployed FastAPI endpoint
"""
import os

# Default API URL - UPDATE THIS with your actual Render API URL
API_URL = "https://your-api-service-name.onrender.com/predict"

# For local development
if not os.environ.get("RENDER"):
    API_URL = "http://127.0.0.1:8000/predict"