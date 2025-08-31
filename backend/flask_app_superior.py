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

def extract_treatment_from_transcript(transcript):
    """Extract treatment plan from transcript"""
    import re
    
    # Look for treatment-related keywords
    treatment_patterns = [
        r'behandeling[:\s]+(.*?)(?=\n|$)',
        r'medicatie[:\s]+(.*?)(?=\n|$)',
        r'therapie[:\s]+(.*?)(?=\n|$)',
        r'voorschrift[:\s]+(.*?)(?=\n|$)',
        r'dosering[:\s]+(.*?)(?=\n|$)',
        r'plan[:\s]+(.*?)(?=\n|$)'
    ]
    
    treatments = []
    for pattern in treatment_patterns:
        matches = re.findall(pattern, transcript, re.IGNORECASE | re.MULTILINE)
        treatments.extend(matches)
    
    if treatments:
        return '; '.join([t.strip() for t in treatments if t.strip()])
    else:
        # If no specific treatment found, extract sentences with drug names
        drug_patterns = [
            r'[A-Z][a-z]+(?:ol|pril|ine|ide|ium|card|xtra|pine)\s+\d+(?:\.\d+)?\s*mg',
            r'arixtra|cedocard|metoprolol|lisinopril|atorvastatine'
        ]
        
        sentences = transcript.split('.')
        treatment_sentences = []
        
        for sentence in sentences:
            for drug_pattern in drug_patterns:
                if re.search(drug_pattern, sentence, re.IGNORECASE):
                    treatment_sentences.append(sentence.strip())
                    break
        
        return '; '.join(treatment_sentences) if treatment_sentences else "Geen specifieke behandeling gedetecteerd"

def compare_treatments(dictated, ai_recommended):
    """Compare dictated treatment with AI recommendations"""
    differences = []
    
    if not dictated or dictated == "Geen specifieke behandeling gedetecteerd":
        differences.append("Geen behandeling gedicteerd - alleen AI aanbevelingen beschikbaar")
        return differences
    
    if not ai_recommended or ai_recommended == "Geen AI aanbevelingen beschikbaar":
        differences.append("AI aanbevelingen niet beschikbaar")
        return differences
    
    # Simple comparison - in real implementation this would be more sophisticated
    dictated_lower = dictated.lower()
    ai_lower = ai_recommended.lower()
    
    # Check for drug differences
    if 'arixtra' in dictated_lower and 'arixtra' not in ai_lower:
        differences.append("Je dicteerde Arixtra, AI beveelt dit niet aan")
    elif 'arixtra' not in dictated_lower and 'arixtra' in ai_lower:
        differences.append("AI beveelt Arixtra aan, maar je dicteerde dit niet")
    
    if 'cedocard' in dictated_lower and 'cedocard' not in ai_lower:
        differences.append("Je dicteerde Cedocard, AI beveelt dit niet aan")
    elif 'cedocard' not in dictated_lower and 'cedocard' in ai_lower:
        differences.append("AI beveelt Cedocard aan, maar je dicteerde dit niet")
    
    # Check for dosage differences
    import re
    dictated_doses = re.findall(r'\d+(?:\.\d+)?\s*mg', dictated_lower)
    ai_doses = re.findall(r'\d+(?:\.\d+)?\s*mg', ai_lower)
    
    if dictated_doses != ai_doses:
        differences.append(f"Dosering verschil: jij {dictated_doses}, AI {ai_doses}")
    
    if not differences:
        differences.append("Behandelingen zijn grotendeels vergelijkbaar")
    
    return differences

# Initialize transcription systemstem
transcription_system = SuperiorMedicalTranscription()

# Initialize medical expert agents system (OpenAI only for reliability)
try:
    from medical_expert_agents_fixed import MedicalExpertAgents
    medical_experts = MedicalExpertAgents()
    EXPERTS_AVAILABLE = True
    print("🤖 Medical Expert Agents (OpenAI Only) initialized successfully!")
except Exception as e:
    print(f"⚠️ Medical Expert Agents not available: {e}")
    medical_experts = None
    EXPERTS_AVAILABLE = False

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
    """Main interface with enhanced template"""
    return render_template('enhanced_index.html')

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
        
        # Transcribe audio with report type
        transcription_result = transcription_service.transcribe_audio(audio_file, verslag_type)        
        if not transcription_result['success']:
            return jsonify({
                'success': False, 
                'error': transcription_result['error']
            }), 400
        
        transcript = transcription_result['transcript']
        
        # 🚀 RUN 3 EXPERT MEDICAL AGENTS (if available)
        expert_analysis = {}
        improved_transcript = transcript
        
        if EXPERTS_AVAILABLE and medical_experts:
            try:
                print(f"🤖 API DEBUG: Starting 3 Expert Medical Agents analysis...")
                expert_analysis = medical_experts.orchestrate_medical_analysis(
                    transcript=transcript,
                    patient_context=f"Patient ID: {patient_id}, Report Type: {verslag_type}"
                )
                
                # Use improved transcript from Agent 1
                improved_transcript = expert_analysis.get('agent_1_quality_control', {}).get('improved_transcript', transcript)
                print(f"🤖 API DEBUG: Expert analysis completed successfully!")
            except Exception as e:
                print(f"⚠️ API DEBUG: Expert analysis failed: {e}")
                expert_analysis = {}
        else:
            print(f"⚠️ API DEBUG: Expert agents not available, using basic processing")
        
        # Generate report
        print(f"🔍 API DEBUG: About to generate report for type: '{verslag_type}'")
        
        if verslag_type == 'TTE':
            print("🔍 API DEBUG: Generating TTE report...")
            report = transcription_system.generate_tte_report(improved_transcript, patient_id)
        elif verslag_type == 'SPOEDCONSULT':
            print("🔍 API DEBUG: Generating SPOEDCONSULT report...")
            report = transcription_system.generate_spoedconsult_report(improved_transcript, patient_id)
        else:
            print(f"🔍 API DEBUG: Unknown type '{verslag_type}', defaulting to TTE...")
            report = transcription_system.generate_tte_report(improved_transcript, patient_id)
        
        print(f"🔍 API DEBUG: Generated report preview: {report[:100]}...")
        
        # Extract treatment from transcript for comparison
        dictated_treatment = extract_treatment_from_transcript(improved_transcript)
        
        # Get AI treatment recommendations
        ai_treatment = ""
        treatment_differences = []
        
        if EXPERTS_AVAILABLE and medical_experts and expert_analysis:
            try:
                # Extract AI treatment recommendations from Agent 3
                agent_3_data = expert_analysis.get('agent_3_treatment_protocol', {})
                treatment_plan = agent_3_data.get('treatment_plan', {})
                
                # Format AI treatment recommendations with ESC 2024 citations and concrete details
                ai_medications = treatment_plan.get('medications', [])
                ai_actions = treatment_plan.get('immediate_actions', [])
                ai_monitoring = treatment_plan.get('monitoring', [])
                esc_citations = agent_3_data.get('esc_2024_citations', [])
                quality_indicators = agent_3_data.get('quality_indicators', {})
                clinical_pathway = agent_3_data.get('clinical_pathway', {})
                
                ai_treatment_parts = []
                if ai_actions:
                    ai_treatment_parts.append(f"🚨 Directe acties: {'; '.join(ai_actions)}")
                
                if ai_medications:
                    med_strings = []
                    for med in ai_medications:
                        if isinstance(med, dict):
                            med_str = f"{med.get('name', 'Unknown')} {med.get('dose', '')} {med.get('frequency', '')}"
                            target = med.get('target_value', '')
                            esc_class = med.get('esc_class', '')
                            esc_evidence = med.get('esc_evidence', '')
                            
                            if target:
                                med_str += f" (target: {target})"
                            if esc_class and esc_evidence:
                                med_str += f" [ESC Class {esc_class}, Level {esc_evidence}]"
                            
                            med_strings.append(med_str.strip())
                        else:
                            med_strings.append(str(med))
                    ai_treatment_parts.append(f"💊 Medicatie: {'; '.join(med_strings)}")
                
                if ai_monitoring:
                    ai_treatment_parts.append(f"📊 Monitoring: {'; '.join(ai_monitoring)}")
                
                # Add clinical pathway for concrete timing
                if clinical_pathway:
                    pathway_parts = []
                    if clinical_pathway.get('day_1'):
                        pathway_parts.append(f"Dag 1: {clinical_pathway['day_1']}")
                    if clinical_pathway.get('day_2_7'):
                        pathway_parts.append(f"Week 1: {clinical_pathway['day_2_7']}")
                    if pathway_parts:
                        ai_treatment_parts.append(f"⏰ Planning: {'; '.join(pathway_parts)}")
                
                # Add ESC 2024 compliance information
                if esc_citations:
                    ai_treatment_parts.append(f"📚 ESC 2024: {'; '.join(esc_citations[:2])}")  # Show first 2 citations
                
                if quality_indicators:
                    guideline_adherence = quality_indicators.get('guideline_adherence', 'unknown')
                    evidence_strength = quality_indicators.get('evidence_strength', 'unknown')
                    target_achievement = quality_indicators.get('target_achievement', 'unknown')
                    ai_treatment_parts.append(f"✅ Kwaliteit: {guideline_adherence}, Evidence: {evidence_strength}, Targets: {target_achievement}")
                
                ai_treatment = ' | '.join(ai_treatment_parts) if ai_treatment_parts else "Geen specifieke AI aanbevelingen"
                
                # Compare treatments
                if dictated_treatment and ai_treatment:
                    treatment_differences = compare_treatments(dictated_treatment, ai_treatment)
                
            except Exception as e:
                print(f"⚠️ Treatment comparison failed: {e}")
                ai_treatment = "AI aanbevelingen niet beschikbaar"
                treatment_differences = ["Vergelijking niet mogelijk"]
        
        return jsonify({
            'success': True,
            'transcript': improved_transcript,  # Show improved transcript
            'raw_transcript': transcript,  # Keep original for debugging
            'report': report,
            'patient_id': patient_id,
            'verslag_type': verslag_type,
            'treatment_comparison': {
                'dictated_treatment': dictated_treatment,
                'ai_treatment': ai_treatment,
                'differences': treatment_differences
            },
            'expert_analysis': {
                'quality_score': expert_analysis.get('agent_1_quality_control', {}).get('quality_score', 0),
                'primary_diagnosis': expert_analysis.get('agent_2_diagnostic_expert', {}).get('primary_diagnosis', {}),
                'treatment_plan': expert_analysis.get('agent_3_treatment_protocol', {}).get('treatment_plan', {}),
                'safety_alerts': expert_analysis.get('agent_1_quality_control', {}).get('safety_alerts', []),
                'urgency_level': expert_analysis.get('agent_2_diagnostic_expert', {}).get('urgency_level', 'unknown'),
                'corrections_made': len(expert_analysis.get('agent_1_quality_control', {}).get('corrections', [])),
                'agents_used': expert_analysis.get('orchestration_summary', {}).get('agents_used', 3),
                'esc_2024_citations': expert_analysis.get('agent_3_treatment_protocol', {}).get('esc_2024_citations', []),
                'esc_guideline_class': expert_analysis.get('agent_3_treatment_protocol', {}).get('esc_guideline_class', 'Unknown'),
                'evidence_level': expert_analysis.get('agent_3_treatment_protocol', {}).get('evidence_level', 'Unknown'),
                'quality_indicators': expert_analysis.get('agent_3_treatment_protocol', {}).get('quality_indicators', {})
            }
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

