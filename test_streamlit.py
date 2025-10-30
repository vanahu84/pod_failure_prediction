"""
Test script to run Streamlit app locally
"""
import subprocess
import sys
import time
import webbrowser

def main():
    print("Starting Streamlit app...")
    print("Make sure your FastAPI server is running on http://127.0.0.1:8000")
    print("Opening browser in 3 seconds...")
    
    time.sleep(3)
    
    # Open browser
    webbrowser.open('http://localhost:8501')
    
    # Start Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501"
    ])

if __name__ == "__main__":
    main()