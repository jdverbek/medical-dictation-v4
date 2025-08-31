"""
Superior Medical Dictation v4.0 - Based on v2 Analysis
Implements all superior features from the working v2 app
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import os
import io
import datetime
import sqlite3
import hashlib
import secrets
import time
import logging
from functools import wraps
from superior_transcription import SuperiorMedicalTranscription

app = Flask(__name__, template_folder='templates')

# Configure session with secure settings (from v2)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=2)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize superior transcription system
transcription_system = SuperiorMedicalTranscription()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting storage (simple in-memory for demo)
rate_limit_storage = {}

def rate_limit(max_requests=20, window=300):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simple rate limiting implementation
            client_ip = request.remote_addr
            current_time = time.time()
            
            if client_ip not in rate_limit_storage:
                rate_limit_storage[client_ip] = []
            
            # Clean old requests
            rate_limit_storage[client_ip] = [
                req_time for req_time in rate_limit_storage[client_ip]
                if current_time - req_time < window
            ]
            
            # Check rate limit
            if len(rate_limit_storage[client_ip]) >= max_requests:
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Add current request
            rate_limit_storage[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.after_request
def add_security_headers(response):
    """Add security headers (from v2)"""
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self';"
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Database initialization (simplified)
def init_db():
    """Initialize database"""
    try:
        conn = sqlite3.connect('medical_app.db')
        cursor = conn.cursor()
        
        # Create transcription history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcription_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                verslag_type TEXT NOT NULL,
                original_transcript TEXT,
                structured_report TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")

# Initialize database on startup
init_db()

@app.route('/')
def index():
    """Main interface"""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
@rate_limit(max_requests=20, window=300)
def transcribe():
    """Superior transcription endpoint based on v2"""
    try:
        # Get form data
        verslag_type = request.form.get('verslag_type', 'TTE')
        patient_id = request.form.get('patient_id', '').strip()
        
        # DEBUG: Log what template was selected
        print(f"🔍 DEBUG: Template selected = '{verslag_type}'")
        print(f"🔍 DEBUG: Patient ID = '{patient_id}'")
        
        # Check if audio file is present
        if 'audio' not in request.files:
            return render_template('index.html', 
                                 error="⚠️ Geen bestand geselecteerd.",
                                 verslag_type=verslag_type)
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return render_template('index.html', 
                                 error="⚠️ Geen bestand geselecteerd.",
                                 verslag_type=verslag_type)
        
        # Transcribe audio using superior system
        transcription_result = transcription_system.transcribe_audio(audio_file)
        
        if not transcription_result['success']:
            return render_template('index.html', 
                                 error=transcription_result['error'],
                                 verslag_type=verslag_type)
        
        corrected_transcript = transcription_result['transcript']
        
        # Generate report based on type
        print(f"🔍 DEBUG: About to generate report for type: '{verslag_type}'")
        
        if verslag_type == 'TTE':
            print("🔍 DEBUG: Generating TTE report...")
            structured_report = transcription_system.generate_tte_report(
                corrected_transcript, patient_id
            )
        elif verslag_type == 'SPOEDCONSULT':
            print("🔍 DEBUG: Generating SPOEDCONSULT report...")
            structured_report = transcription_system.generate_spoedconsult_report(
                corrected_transcript, patient_id
            )
        else:
            print(f"🔍 DEBUG: Unknown type '{verslag_type}', defaulting to TTE...")
            # Default TTE
            structured_report = transcription_system.generate_tte_report(
                corrected_transcript, patient_id
            )
        
        print(f"🔍 DEBUG: Generated report preview: {structured_report[:100]}...")
        
        # Store in database
        try:
            conn = sqlite3.connect('medical_app.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transcription_history 
                (patient_id, verslag_type, original_transcript, structured_report)
                VALUES (?, ?, ?, ?)
            ''', (patient_id, verslag_type, corrected_transcript, structured_report))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database error: {e}")
        
        # Return results
        return render_template('index.html',
                             transcript=corrected_transcript,
                             report=structured_report,
                             verslag_type=verslag_type,
                             patient_id=patient_id,
                             success=True)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return render_template('index.html', 
                             error=f"Fout bij verwerking: {str(e)}",
                             verslag_type=verslag_type)

@app.route('/api/transcribe', methods=['POST'])
@rate_limit(max_requests=20, window=300)
def api_transcribe():
    """API endpoint for transcription"""
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        verslag_type = request.form.get('verslag_type', 'TTE')
        patient_id = request.form.get('patient_id', '')
        
        # DEBUG: Log what template was selected in API
        print(f"🔍 API DEBUG: Template selected = '{verslag_type}'")
        print(f"🔍 API DEBUG: Patient ID = '{patient_id}'")
        print(f"🔍 API DEBUG: Audio filename = '{audio_file.filename}'")
        
        # Transcribe audio
        transcription_result = transcription_system.transcribe_audio(audio_file)
        
        if not transcription_result['success']:
            return jsonify({
                'success': False, 
                'error': transcription_result['error']
            }), 400
        
        transcript = transcription_result['transcript']
        
        # Generate report
        print(f"🔍 API DEBUG: About to generate report for type: '{verslag_type}'")
        
        if verslag_type == 'TTE':
            print("🔍 API DEBUG: Generating TTE report...")
            report = transcription_system.generate_tte_report(transcript, patient_id)
        elif verslag_type == 'SPOEDCONSULT':
            print("🔍 API DEBUG: Generating SPOEDCONSULT report...")
            report = transcription_system.generate_spoedconsult_report(transcript, patient_id)
        else:
            print(f"🔍 API DEBUG: Unknown type '{verslag_type}', defaulting to TTE...")
            report = transcription_system.generate_tte_report(transcript, patient_id)
        
        print(f"🔍 API DEBUG: Generated report preview: {report[:100]}...")
        
        return jsonify({
            'success': True,
            'transcript': transcript,
            'report': report,
            'patient_id': patient_id,
            'verslag_type': verslag_type
        })
        
    except Exception as e:
        logger.error(f"API transcription error: {e}")
        return jsonify({
            'success': False, 
            'error': f'Transcription failed: {str(e)}'
        }), 500

@app.route('/history')
def history():
    """View transcription history"""
    try:
        conn = sqlite3.connect('medical_app.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, patient_id, verslag_type, created_at, 
                   substr(original_transcript, 1, 100) as preview
            FROM transcription_history 
            ORDER BY created_at DESC 
            LIMIT 50
        ''')
        history_records = cursor.fetchall()
        conn.close()
        
        return render_template('history.html', records=history_records)
    except Exception as e:
        logger.error(f"History error: {e}")
        return render_template('history.html', records=[], error=str(e))

@app.route('/view/<int:record_id>')
def view_record(record_id):
    """View specific transcription record"""
    try:
        conn = sqlite3.connect('medical_app.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT patient_id, verslag_type, original_transcript, 
                   structured_report, created_at
            FROM transcription_history 
            WHERE id = ?
        ''', (record_id,))
        record = cursor.fetchone()
        conn.close()
        
        if record:
            return render_template('view_record.html', 
                                 patient_id=record[0],
                                 verslag_type=record[1],
                                 transcript=record[2],
                                 report=record[3],
                                 created_at=record[4])
        else:
            return "Record not found", 404
            
    except Exception as e:
        logger.error(f"View record error: {e}")
        return f"Error: {str(e)}", 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '4.0-superior',
        'features': [
            'WebM audio detection',
            'Hallucination detection', 
            'Quality control review',
            'Dutch medical terminology',
            'Context-aware drug correction',
            'Rate limiting',
            'Security headers'
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting Superior Medical Dictation v4.0")
    print("📋 Features: WebM detection, hallucination detection, quality control")
    print("🏥 Medical: Dutch terminology, context-aware drugs, safety rules")
    print("🔒 Security: Rate limiting, headers, input validation")
    app.run(host='0.0.0.0', port=5000, debug=False)

