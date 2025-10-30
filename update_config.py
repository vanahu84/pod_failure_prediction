"""
Script to update the API URL in config.py after deployment
Run this script with your actual API URL
"""
import sys

def update_api_url(api_url):
    """Update the API URL in config.py"""
    config_content = f'''"""
Configuration file for the Streamlit app
Update the API_URL with your deployed FastAPI endpoint
"""
import os

# Default API URL - UPDATE THIS with your actual Render API URL
API_URL = "{api_url}"

# For local development
if not os.environ.get("RENDER"):
    API_URL = "http://127.0.0.1:8000/predict"'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print(f"✅ Updated config.py with API URL: {api_url}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_config.py <your-api-url>")
        print("Example: python update_config.py https://pod-failure-api.onrender.com/predict")
        sys.exit(1)
    
    api_url = sys.argv[1]
    if not api_url.startswith('http'):
        print("❌ API URL must start with http:// or https://")
        sys.exit(1)
    
    update_api_url(api_url)