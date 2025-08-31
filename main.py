# main.py - Superior Entry point for Render deployment

import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import the SUPERIOR Flask app
from backend.flask_app_superior import app

# This allows Render to find the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting SUPERIOR Medical Dictation v4.0")
    print("📋 Based on analysis of superior v2 app")
    print("🏥 Features: WebM detection, hallucination detection, quality control")
    app.run(host="0.0.0.0", port=port, debug=False)

