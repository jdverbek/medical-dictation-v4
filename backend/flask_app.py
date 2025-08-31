# backend/flask_app.py - Ultra-simple Flask app for guaranteed deployment

"""
Medical Dictation v4.0 - Flask Version
Minimal dependencies, maximum compatibility
"""

from flask import Flask, request, jsonify, send_file, render_template
import os
import uuid
import json
import io
from datetime import datetime
import openai
import anthropic

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Flask app
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize AI clients (older API style)
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

anthropic_client = None
if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# In-memory storage
audio_storage = {}
transcription_results = {}

@app.route('/', methods=['GET'])
def index():
    """Main web interface"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "4.0.0-flask",
        "services": {
            "openai": "configured" if OPENAI_API_KEY else "not configured",
            "anthropic": "configured" if anthropic_client else "not configured"
        }
    })

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "Medical Dictation v4.0 Flask API", 
        "status": "running",
        "endpoints": ["/health", "/api/transcribe", "/api/audio/<session_id>"]
    })

def transcribe_audio_sync(audio_data):
    """Synchronous transcription using Whisper (older API)"""
    if not OPENAI_API_KEY:
        return "OpenAI API not configured"
    
    try:
        # Save audio to temporary file for older API
        temp_filename = f"/tmp/audio_{uuid.uuid4()}.webm"
        with open(temp_filename, 'wb') as f:
            f.write(audio_data)
        
        # Use older OpenAI API
        with open(temp_filename, 'rb') as audio_file:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language="nl",
                temperature=0.0,
                prompt="Medische dictatie in het Nederlands. Let op medische terminologie."
            )
        
        # Clean up temp file
        os.remove(temp_filename)
        
        return response.text
    except Exception as e:
        print(f"Transcription failed: {e}")
        return f"Transcription error: {str(e)}"

def improve_transcript_sync(transcript, patient_id, report_type):
    """Improve transcript using GPT-4 (older API)"""
    if not OPENAI_API_KEY:
        return transcript
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": f"""Je bent een medische transcriptie expert. Verbeter de transcriptie voor een {report_type} rapport:

1. Corrigeer verkeerd gespelde medicijnnamen (bijv. sedocar -> cedocard)
2. Verbeter medische terminologie
3. Zorg voor logische consistentie
4. Behoud alle originele informatie

Geef alleen de verbeterde transcriptie terug, geen uitleg."""
            }, {
                "role": "user",
                "content": f"Patiënt ID: {patient_id}\nTranscriptie: {transcript}"
            }],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT improvement failed: {e}")
        return transcript

def generate_medical_report_sync(transcript, patient_id, report_type):
    """Generate structured medical report (older API)"""
    if not OPENAI_API_KEY:
        return f"MEDISCH RAPPORT\n\nPatiënt ID: {patient_id}\nType: {report_type}\n\nTranscriptie:\n{transcript}"
    
    templates = {
        "TTE": "Structureer als TTE rapport met linker ventrikel, rechter ventrikel, atria, kleppen en conclusie",
        "TEE": "Structureer als TEE rapport met gedetailleerde klep analyse",
        "ECG": "Structureer als ECG rapport met ritme, intervallen, as en morfologie",
        "Holter": "Structureer als Holter rapport met basisritme en aritmieën",
        "Consult": "Structureer als consultrapport met anamnese, onderzoek en beleid"
    }
    
    template = templates.get(report_type, templates["Consult"])
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": f"""Je bent een ervaren cardioloog. Schrijf een gestructureerd medisch rapport.

{template}

Gebruik alleen informatie uit de transcriptie. Schrijf in professioneel Nederlands."""
            }, {
                "role": "user",
                "content": f"Patiënt ID: {patient_id}\nTranscriptie: {transcript}"
            }],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Report generation failed: {e}")
        return f"MEDISCH RAPPORT\n\nPatiënt ID: {patient_id}\nType: {report_type}\n\nTranscriptie:\n{transcript}"

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio_endpoint():
    """Main transcription endpoint with AI intelligence"""
    
    try:
        # Get form data
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        patient_id = request.form.get('patient_id', 'Unknown')
        report_type = request.form.get('report_type', 'TTE')
        
        # Validate audio
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Read audio data
        audio_data = audio_file.read()
        session_id = str(uuid.uuid4())
        
        # Store audio for playback
        audio_storage[session_id] = audio_data
        
        # Step 1: Initial transcription
        print(f"Starting transcription for session {session_id}")
        transcript = transcribe_audio_sync(audio_data)
        
        # Step 2: Improve with GPT-4 (multi-agent simulation)
        print(f"Improving transcript for session {session_id}")
        improved_transcript = improve_transcript_sync(transcript, patient_id, report_type)
        
        # Step 3: Generate medical report
        print(f"Generating report for session {session_id}")
        report = generate_medical_report_sync(improved_transcript, patient_id, report_type)
        
        # Store result
        result = {
            'transcript': improved_transcript,
            'report': report,
            'session_id': session_id,
            'confidence': 0.9,
            'processing_metadata': {
                'agents_used': ['transcriber', 'gpt_improver', 'report_generator'],
                'iterations': 1,
                'improvements_made': 1 if improved_transcript != transcript else 0
            }
        }
        
        transcription_results[session_id] = result
        
        return jsonify({
            "success": True,
            "transcript": result['transcript'],
            "report": result['report'],
            "audio_url": f"/api/audio/{session_id}",
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/api/audio/<session_id>', methods=['GET'])
def get_audio_playback(session_id):
    """Retrieve audio for playback"""
    
    audio_data = audio_storage.get(session_id)
    if not audio_data:
        return jsonify({"error": "Audio not found"}), 404
    
    return send_file(
        io.BytesIO(audio_data),
        mimetype="audio/webm",
        as_attachment=False,
        download_name=f"recording_{session_id}.webm"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)

