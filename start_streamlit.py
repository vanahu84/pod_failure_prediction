"""
Startup script for Streamlit on Render.com
"""
import os
import subprocess
import sys

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false"
    ]
    subprocess.run(cmd)