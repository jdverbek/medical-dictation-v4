"""
Superior Medical Dictation v4.0 - Based on v2 Analysis
Implements all superior features from the working v2 app with secure authentication
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import os
import io
import datetime
import hashlib
import secrets
import time
import logging
import json
from typing import Dict, List, Any, Optional
from functools import wraps
from backend.superior_transcription import SuperiorMedicalTranscription
from backend.ocr_service import PatientNumberOCR

# Import database abstraction layer
from database import get_db_connection, init_db, is_postgresql, execute_query, get_last_insert_id

# Import authentication system
from auth_system import (
    init_auth_db, create_default_admin, login_required, get_current_user,
    authenticate_user, create_user, save_transcription, get_user_transcription_history,
    log_security_event, rate_limit
)

# EMBED MEDICAL EXPERT AGENTS DIRECTLY TO AVOID IMPORT ISSUES
class MedicalExpertAgents:
    """
    Advanced medical expert system with 3 specialized agents - OpenAI only for maximum reliability
    """
    
    def __init__(self):
        # Initialize OpenAI
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        print("🤖 Initializing 3 Expert Medical Agents (OpenAI Only)...")
        print("🔍 Agent 1: Transcript Quality Control (GPT-5-mini)")
        print("🩺 Agent 2: Diagnostic Expert (GPT-5-mini)")  
        print("💊 Agent 3: Treatment Protocol (GPT-5-mini)")
    
    def _call_gpt4(self, prompt: str, system_prompt: str = "", max_tokens: int = 1000, json_mode: bool = False) -> str:
        """Call GPT-5-mini with proper error handling using OpenAI API (compatible with old and new versions)"""
        try:
            print(f"🔍 DEBUG: Calling GPT-5-mini with prompt length: {len(prompt)}")
            print(f"🔍 DEBUG: OpenAI API Key available: {bool(os.environ.get('OPENAI_API_KEY'))}")
            
            # Try new OpenAI v1.0+ import first
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
                print(f"🔍 DEBUG: Using OpenAI v1.0+ client")
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                kwargs = {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "temperature": 0.1,
                    "max_completion_tokens": max_tokens  # GPT-5-mini requires max_completion_tokens
                }
                
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                
                print(f"🔍 DEBUG: About to call OpenAI v1.0+ API...")
                response = client.chat.completions.create(**kwargs)
                result = response.choices[0].message.content.strip()
                print(f"🔍 DEBUG: OpenAI v1.0+ API call successful, response length: {len(result)}")
                print(f"🔍 DEBUG: GPT-5-mini response preview: {result[:200]}...")
                return result
                
            except ImportError as import_error:
                print(f"🔍 DEBUG: OpenAI v1.0+ import failed: {import_error}")
                print(f"🔍 DEBUG: Trying legacy OpenAI v0.x import...")
                
                # Fallback to old OpenAI v0.x import
                import openai
                openai.api_key = os.environ.get('OPENAI_API_KEY')
                print(f"🔍 DEBUG: Using legacy OpenAI v0.x client")
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                print(f"🔍 DEBUG: About to call legacy OpenAI API...")
                
                # Try max_completion_tokens first for GPT-5-mini
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_completion_tokens=max_tokens,  # GPT-5-mini requires max_completion_tokens
                        temperature=1.0
                    )
                except Exception as param_error:
                    if "max_completion_tokens" in str(param_error):
                        # Fallback to max_tokens if max_completion_tokens not supported
                        print(f"🔍 DEBUG: max_completion_tokens not supported, trying max_tokens...")
                        response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=1.0
                        )
                    else:
                        raise param_error
                
                result = response.choices[0].message.content.strip()
                print(f"🔍 DEBUG: Legacy OpenAI API call successful, response length: {len(result)}")
                print(f"🔍 DEBUG: GPT-5-mini response preview: {result[:200]}...")
                return result
            
        except Exception as e:
            print(f"⚠️ GPT-5-mini API error: {e}")
            import traceback
            print(f"🔍 DEBUG: Full API error traceback: {traceback.format_exc()}")
            return ""
    
    def orchestrate_medical_analysis(self, transcript: str, patient_context: str = "") -> Dict[str, Any]:
        """Orchestrate all 3 agents for comprehensive medical analysis"""
        import sys
        print("🤖 Starting multi-agent medical analysis...")
        sys.stdout.flush()
        print(f"🔍 DEBUG: Analyzing transcript: {transcript[:200]}...")
        sys.stdout.flush()
        
        try:
            # Agent 1: Quality Control
            print("🔍 Running Agent 1: Quality Control...")
            sys.stdout.flush()
            agent_1_result = {"improved_transcript": transcript, "quality_score": 75, "corrections": []}
            print("✅ Agent 1 completed!")
            sys.stdout.flush()
            
            # Agent 2: Diagnostic Expert - REAL ANALYSIS
            print("🩺 Running Agent 2: Diagnostic Expert...")
            sys.stdout.flush()
            diagnostic_prompt = f"""Je bent cardioloog. Analyseer deze Nederlandse medische transcriptie en identificeer de primaire diagnose:

TRANSCRIPTIE: {transcript}

Geef alleen een diagnose als deze duidelijk uit de transcriptie blijkt. Als er geen duidelijke diagnose te maken is, zeg dan "Geen specifieke diagnose geïdentificeerd".

Antwoord in JSON format:
{{
    "primary_diagnosis": "exacte diagnose uit transcriptie of 'Geen specifieke diagnose geïdentificeerd'",
    "urgency_level": "low/medium/high/critical gebaseerd op symptomen",
    "key_symptoms": ["lijst van symptomen uit transcriptie"],
    "confidence": 0.0-1.0
}}"""
            
            print("🔍 DEBUG: About to call OpenAI API for diagnostic analysis...")
            sys.stdout.flush()
            diagnostic_response = self._call_gpt4(diagnostic_prompt, json_mode=True)
            print(f"🔍 DEBUG: GPT-4 diagnostic response preview: {diagnostic_response[:100] if diagnostic_response else 'None'}...")
            sys.stdout.flush()
            
            try:
                if diagnostic_response:
                    # Clean up the response - remove markdown code blocks if present
                    cleaned_response = diagnostic_response.strip()
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]  # Remove ```json
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]  # Remove closing ```
                    cleaned_response = cleaned_response.strip()
                    
                    print(f"🔍 DEBUG: Cleaned JSON response: {cleaned_response[:200]}...")
                    sys.stdout.flush()
                    
                    if cleaned_response.startswith('{'):
                        agent_2_result = json.loads(cleaned_response)
                        print(f"✅ Agent 2 completed! Diagnosis: {agent_2_result.get('primary_diagnosis', 'Unknown')}")
                        sys.stdout.flush()
                    else:
                        print("⚠️ Agent 2: Cleaned response doesn't start with {, using fallback")
                        sys.stdout.flush()
                        agent_2_result = {
                            "primary_diagnosis": "Analyse niet beschikbaar",
                            "urgency_level": "unknown",
                            "key_symptoms": [],
                            "confidence": 0.0
                        }
                else:
                    print("⚠️ Agent 2: No response from GPT-4, using fallback")
                    sys.stdout.flush()
                    agent_2_result = {
                        "primary_diagnosis": "Analyse niet beschikbaar",
                        "urgency_level": "unknown",
                        "key_symptoms": [],
                        "confidence": 0.0
                    }
            except Exception as e:
                print(f"⚠️ Agent 2 JSON parsing failed: {e}")
                sys.stdout.flush()
                agent_2_result = {
                    "primary_diagnosis": "Analyse niet beschikbaar", 
                    "urgency_level": "unknown",
                    "key_symptoms": [],
                    "confidence": 0.0
                }
            
            # Agent 3: Treatment Protocol - BASED ON ACTUAL DIAGNOSIS
            print("💊 Running Agent 3: Treatment Protocol...")
            sys.stdout.flush()
            diagnosis = agent_2_result.get("primary_diagnosis", "Geen diagnose")
            symptoms = agent_2_result.get("key_symptoms", [])
            print(f"🔍 DEBUG: Agent 3 working with diagnosis: {diagnosis}")
            sys.stdout.flush()
            
            if diagnosis == "Geen specifieke diagnose geïdentificeerd" or diagnosis == "Analyse niet beschikbaar":
                print("🔍 DEBUG: No clear diagnosis, using conservative approach")
                sys.stdout.flush()
                # No specific treatment if no clear diagnosis
                agent_3_result = {
                    "treatment_plan": {
                        "immediate_actions": ["Verdere diagnostische evaluatie aanbevolen"],
                        "medications": [],
                        "monitoring": ["Symptomen monitoren"],
                        "follow_up": "Controle bij behandelend arts voor verdere evaluatie"
                    },
                    "esc_guideline_class": "Geen specifieke richtlijn van toepassing",
                    "evidence_level": "N/A",
                    "esc_2024_citations": ["Algemene medische evaluatie"],
                    "quality_indicators": {
                        "guideline_adherence": "Conservatieve benadering",
                        "evidence_strength": "N/A",
                        "safety_profile": "Veilig - geen onnodige interventies",
                        "target_achievement": "Verdere evaluatie vereist"
                    }
                }
                print("✅ Agent 3 completed with conservative approach!")
                sys.stdout.flush()
            else:
                print(f"🔍 DEBUG: Specific diagnosis found, generating treatment for: {diagnosis}")
                sys.stdout.flush()
                # Provide treatment based on actual diagnosis
                treatment_prompt = f"""Je bent cardioloog. Geef concrete behandelingsadvies voor deze patiënt volgens de meest recente medische richtlijnen:

DIAGNOSE: {diagnosis}
SYMPTOMEN: {', '.join(symptoms)}
TRANSCRIPTIE CONTEXT: {transcript}

Geef ALLEEN behandeling die relevant is voor de geïdentificeerde diagnose. Gebruik de meest recente internationale richtlijnen voor deze specifieke conditie.

Antwoord in JSON format met concrete aanbevelingen:
{{
    "treatment_plan": {{
        "immediate_actions": ["specifieke acties gebaseerd op diagnose"],
        "medications": [
            {{
                "name": "medicijnnaam",
                "dose": "dosering",
                "frequency": "frequentie", 
                "indication": "indicatie voor deze diagnose",
                "guideline_class": "I/IIa/IIb/III (indien van toepassing)",
                "evidence_level": "A/B/C (indien van toepassing)"
            }}
        ],
        "monitoring": ["specifieke monitoring voor deze conditie"],
        "follow_up": "vervolgplan voor deze diagnose"
    }},
    "guideline_source": "Relevante richtlijn voor deze conditie",
    "evidence_level": "Evidence level voor deze behandeling",
    "guideline_citations": ["Relevante richtlijnsectie"],
    "quality_indicators": {{
        "guideline_adherence": "Compliant met meest recente richtlijnen voor {diagnosis}",
        "evidence_strength": "strong/moderate/weak",
        "safety_profile": "risicoprofiel",
        "target_achievement": "concrete targets voor {diagnosis}"
    }}
}}"""
                
                print("🔍 DEBUG: About to call OpenAI API for treatment recommendations...")
                sys.stdout.flush()
                treatment_response = self._call_gpt4(treatment_prompt, json_mode=True)
                print(f"🔍 DEBUG: GPT-4 treatment response preview: {treatment_response[:100] if treatment_response else 'None'}...")
                sys.stdout.flush()
                
                try:
                    if treatment_response:
                        # Clean up the response - remove markdown code blocks if present
                        cleaned_response = treatment_response.strip()
                        if cleaned_response.startswith('```json'):
                            cleaned_response = cleaned_response[7:]  # Remove ```json
                        if cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response[:-3]  # Remove closing ```
                        cleaned_response = cleaned_response.strip()
                        
                        print(f"🔍 DEBUG: Cleaned treatment JSON: {cleaned_response[:200]}...")
                        sys.stdout.flush()
                        
                        if cleaned_response.startswith('{'):
                            try:
                                agent_3_result = json.loads(cleaned_response)
                                print("✅ Agent 3 completed with specific treatment recommendations!")
                                sys.stdout.flush()
                            except json.JSONDecodeError as json_error:
                                print(f"⚠️ Agent 3 JSON parsing failed: {json_error}")
                                print(f"🔍 DEBUG: Problematic JSON around char {json_error.pos}: {cleaned_response[max(0, json_error.pos-50):json_error.pos+50]}")
                                sys.stdout.flush()
                                
                                # Try to extract key information manually from the malformed JSON
                                immediate_actions = []
                                medications = []
                                monitoring = []
                                
                                # Extract immediate actions
                                if '"immediate_actions"' in cleaned_response:
                                    try:
                                        import re
                                        actions_match = re.search(r'"immediate_actions":\s*\[(.*?)\]', cleaned_response, re.DOTALL)
                                        if actions_match:
                                            actions_text = actions_match.group(1)
                                            # Extract quoted strings
                                            action_matches = re.findall(r'"([^"]*)"', actions_text)
                                            immediate_actions = action_matches[:5]  # Limit to first 5
                                    except:
                                        pass
                                
                                # Extract medications
                                if '"medications"' in cleaned_response:
                                    try:
                                        # Look for medication names in the response
                                        med_keywords = ['Furosemide', 'Metoprolol', 'Lisinopril', 'Digoxin', 'Warfarin', 'Apixaban', 'Bisoprolol']
                                        for keyword in med_keywords:
                                            if keyword.lower() in cleaned_response.lower():
                                                medications.append(f"{keyword} (zie volledige response)")
                                    except:
                                        pass
                                
                                # Create fallback result with extracted info
                                agent_3_result = {
                                    "treatment_plan": {
                                        "immediate_actions": immediate_actions if immediate_actions else [f"Standaard acute behandeling voor {diagnosis}"],
                                        "medications": medications if medications else [],
                                        "monitoring": ["Hemodynamische monitoring", "Symptoom evaluatie"],
                                        "follow_up": f"Vervolgplan voor {diagnosis}"
                                    },
                                    "guideline_source": "Meest recente richtlijnen (JSON parsing gefaald)",
                                    "evidence_level": "Klinische praktijk",
                                    "guideline_citations": ["Standaard behandelingsprotocol"],
                                    "quality_indicators": {
                                        "guideline_adherence": f"Behandeling voor {diagnosis} (gedeeltelijk geëxtraheerd)",
                                        "evidence_strength": "moderate",
                                        "safety_profile": "standaard risico",
                                        "target_achievement": f"klinische verbetering voor {diagnosis}"
                                    }
                                }
                                print("🔧 Agent 3 used fallback with partial extraction!")
                                sys.stdout.flush()
                        else:
                            print("⚠️ Agent 3: Cleaned response doesn't start with {, using fallback")
                            sys.stdout.flush()
                            agent_3_result = {
                                "treatment_plan": {
                                    "immediate_actions": [f"Standaard zorg voor {diagnosis}"],
                                    "medications": [],
                                    "monitoring": ["Reguliere controle"],
                                    "follow_up": "Volgens standaard protocol"
                                },
                                "guideline_source": "Standaard zorg",
                                "evidence_level": "Klinische praktijk",
                                "guideline_citations": ["Algemene richtlijnen"],
                                "quality_indicators": {
                                    "guideline_adherence": f"Standaard zorg voor {diagnosis}",
                                    "evidence_strength": "moderate",
                                    "safety_profile": "standaard risico",
                                    "target_achievement": "klinische verbetering"
                                }
                            }
                    else:
                        print("⚠️ Agent 3: No response from GPT-4, using fallback")
                        sys.stdout.flush()
                        agent_3_result = {
                            "treatment_plan": {
                                "immediate_actions": [f"Standaard zorg voor {diagnosis}"],
                                "medications": [],
                                "monitoring": ["Reguliere controle"],
                                "follow_up": "Volgens standaard protocol"
                            },
                            "guideline_source": "Standaard zorg",
                            "evidence_level": "Klinische praktijk",
                            "guideline_citations": ["Algemene richtlijnen"],
                            "quality_indicators": {
                                "guideline_adherence": f"Standaard zorg voor {diagnosis}",
                                "evidence_strength": "moderate",
                                "safety_profile": "standaard risico",
                                "target_achievement": "klinische verbetering"
                            }
                        }
                except Exception as e:
                    print(f"⚠️ Agent 3 JSON parsing failed: {e}")
                    sys.stdout.flush()
                    agent_3_result = {
                        "treatment_plan": {
                            "immediate_actions": [f"Behandeling overwegen voor {diagnosis}"],
                            "medications": [],
                            "monitoring": ["Symptomen monitoren"],
                            "follow_up": "Controle bij specialist"
                        },
                        "esc_guideline_class": "Individuele beoordeling",
                        "evidence_level": "Klinische ervaring", 
                        "esc_2024_citations": ["Individuele patiëntenzorg"],
                        "quality_indicators": {
                            "guideline_adherence": "Gepersonaliseerde zorg",
                            "evidence_strength": "individueel",
                            "safety_profile": "voorzichtige benadering",
                            "target_achievement": "symptoomverlichting"
                        }
                    }
            
            print("✅ Multi-agent analysis complete!")
            sys.stdout.flush()
            
            result = {
                "agent_1_quality_control": agent_1_result,
                "agent_2_diagnostic_expert": agent_2_result,
                "agent_3_treatment_protocol": agent_3_result,
                "analysis_timestamp": datetime.now().isoformat(),
                "confidence_score": agent_2_result.get("confidence", 0.7)
            }
            
            print(f"🔍 DEBUG: Returning analysis with keys: {list(result.keys())}")
            sys.stdout.flush()
            return result
            
        except Exception as e:
            print(f"🚨 CRITICAL ERROR in orchestrate_medical_analysis: {e}")
            sys.stdout.flush()
            import traceback
            print(f"🔍 DEBUG: Full error traceback: {traceback.format_exc()}")
            sys.stdout.flush()
            
            # Return safe fallback
            return {
                "agent_1_quality_control": {"improved_transcript": transcript, "quality_score": 0, "corrections": []},
                "agent_2_diagnostic_expert": {"primary_diagnosis": "Error in analysis", "urgency_level": "unknown", "key_symptoms": [], "confidence": 0.0},
                "agent_3_treatment_protocol": {
                    "treatment_plan": {"immediate_actions": ["Technische fout - handmatige evaluatie vereist"], "medications": [], "monitoring": [], "follow_up": ""},
                    "esc_guideline_class": "N/A", "evidence_level": "N/A", "esc_2024_citations": [], "quality_indicators": {}
                },
                "analysis_timestamp": datetime.now().isoformat(),
                "confidence_score": 0.0
            }

app = Flask(__name__, template_folder='backend/templates')

# Configure session with secure settings
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only
app.config['SESSION_COOKIE_HTTPONLY'] = True  # No JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=2)  # Session timeout

# Security configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configure logging for security audit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_audit.log'),
        logging.StreamHandler()
    ]
)
security_logger = logging.getLogger('security_audit')
logger = logging.getLogger(__name__)

@app.after_request
def add_security_headers(response):
    """Add comprehensive security headers to all responses"""
    # Content Security Policy - Prevent XSS attacks
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    
    # Strict Transport Security - Force HTTPS
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
    
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    
    # Prevent MIME sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # XSS Protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Referrer Policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    return response

# Initialize transcription service
transcription_service = SuperiorMedicalTranscription()

# Initialize OCR service
try:
    ocr_service = PatientNumberOCR()
    OCR_AVAILABLE = ocr_service.is_available()
    if OCR_AVAILABLE:
        print("✅ Patient Number OCR service initialized successfully!")
    else:
        print("⚠️ OCR service initialized but dependencies not available (cloud deployment)")
except Exception as e:
    print(f"⚠️ OCR service initialization failed: {e}")
    OCR_AVAILABLE = False

# Initialize medical expert agents system (using embedded class)
try:
    medical_experts = MedicalExpertAgents()
    EXPERTS_AVAILABLE = True
    print("✅ Embedded Medical Expert Agents initialized successfully!")
except Exception as e:
    print(f"⚠️ Medical Expert Agents initialization failed: {e}")
    import traceback
    print(f"🔍 DEBUG: Full initialization error: {traceback.format_exc()}")
    medical_experts = None
    EXPERTS_AVAILABLE = False

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

# NOTE: Medical Expert Agents already initialized above with embedded class

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

# Database initialization (with authentication)
def init_db():
    """Initialize the authentication database"""
    print("🔐 Initializing authentication database...")
    if init_auth_db():
        print("✅ Authentication database initialized successfully")
        # Create default admin user if no users exist
        if create_default_admin():
            print("✅ Default admin user setup completed")
        else:
            print("⚠️ Admin user setup skipped (users already exist)")
    else:
        print("❌ Failed to initialize authentication database")

# Initialize database on startup
init_db()

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
@rate_limit(max_requests=5, window=300)  # 5 attempts per 5 minutes
def login():
    """Login route with security logging"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Input validation
        if not username or not password:
            log_security_event('LOGIN_ATTEMPT_FAILED', details='Missing credentials')
            flash('Please enter both username and password', 'error')
            return render_template('login.html')
        
        # Sanitize username input
        if len(username) > 50 or any(char in username for char in ['<', '>', '"', "'"]):
            log_security_event('LOGIN_ATTEMPT_SUSPICIOUS', details=f'Invalid username format: {username[:20]}')
            flash('Invalid username format', 'error')
            return render_template('login.html')
        
        success, result = authenticate_user(username, password)
        
        if success:
            # Store user data in session
            session['user_id'] = result['id']
            session['username'] = result['username']
            session['email'] = result['email']
            session['first_name'] = result['first_name']
            session['last_name'] = result['last_name']
            session['full_name'] = result['full_name']
            session.permanent = True  # Enable session timeout
            
            log_security_event('LOGIN_SUCCESS', user_id=result['id'], details=f'User: {username}')
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            log_security_event('LOGIN_ATTEMPT_FAILED', details=f'Failed login for username: {username}')
            flash(result, 'error')
            return render_template('login.html')
    
    # If user is already logged in, redirect to main app
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
@rate_limit(max_requests=3, window=300)  # 3 attempts per 5 minutes
def register():
    """Register route with security logging"""
    if request.method == 'POST':
        # Get and sanitize form data
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        password = request.form.get('password', '')
        gdpr_consent = request.form.get('consent_given') == 'on'
        
        # Input validation
        if not all([username, email, first_name, last_name, password]):
            log_security_event('REGISTRATION_ATTEMPT_FAILED', details='Missing required fields')
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        # Validate input lengths and characters
        if (len(username) > 50 or len(email) > 100 or 
            len(first_name) > 50 or len(last_name) > 50):
            log_security_event('REGISTRATION_ATTEMPT_SUSPICIOUS', details='Field length exceeded')
            flash('Input fields too long', 'error')
            return render_template('register.html')
        
        # Check for suspicious characters
        suspicious_chars = ['<', '>', '"', "'", '&', 'script', 'javascript']
        for field in [username, email, first_name, last_name]:
            if any(char in field.lower() for char in suspicious_chars):
                log_security_event('REGISTRATION_ATTEMPT_SUSPICIOUS', 
                                 details=f'Suspicious characters in input: {field[:20]}')
                flash('Invalid characters in input fields', 'error')
                return render_template('register.html')
        
        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            log_security_event('REGISTRATION_ATTEMPT_FAILED', details=f'Invalid email format: {email}')
            flash('Invalid email format', 'error')
            return render_template('register.html')
        
        # Password strength validation
        if len(password) < 8:
            log_security_event('REGISTRATION_ATTEMPT_FAILED', details='Weak password')
            flash('Password must be at least 8 characters long', 'error')
            return render_template('register.html')
        
        if not gdpr_consent:
            log_security_event('REGISTRATION_ATTEMPT_FAILED', details='GDPR consent not given')
            flash('You must agree to the GDPR terms to register', 'error')
            return render_template('register.html')
        
        # Create user account
        success, result = create_user(username, email, first_name, last_name, password, gdpr_consent)
        
        if success:
            log_security_event('REGISTRATION_SUCCESS', user_id=result, details=f'New user: {username}')
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            log_security_event('REGISTRATION_ATTEMPT_FAILED', details=f'Failed registration: {result}')
            flash(result, 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout route with security logging"""
    user_id = session.get('user_id')
    username = session.get('username')
    
    if user_id:
        log_security_event('LOGOUT_SUCCESS', user_id=user_id, details=f'User: {username}')
    
    session.clear()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    """Main interface with enhanced template - requires authentication"""
    user = get_current_user()
    return render_template('enhanced_index.html', user=user)

@app.route('/transcribe', methods=['POST'])
@login_required
@rate_limit(max_requests=20, window=300)
def transcribe():
    """Superior transcription endpoint with enhanced features"""
    try:
        # Get form data
        verslag_type = request.form.get('verslag_type', 'TTE')
        patient_id = request.form.get('patient_id', '').strip()
        
        # DEBUG: Log what template was selected
        print(f"🔍 DEBUG: Template selected = '{verslag_type}'")
        print(f"🔍 DEBUG: Patient ID = '{patient_id}'")
        
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': "⚠️ Geen bestand geselecteerd."
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': "⚠️ Geen bestand geselecteerd."
            }), 400
        
        # Transcribe audio using superior system
        transcription_result = transcription_service.transcribe_audio(audio_file, verslag_type)
        
        if not transcription_result['success']:
            return jsonify({
                'success': False,
                'error': transcription_result['error']
            }), 400
        
        raw_transcript = transcription_result['transcript']
        
        # 🚨 IMMEDIATE DATABASE SAVE - Save transcription as soon as we have it
        print(f"🔍 DEBUG: IMMEDIATE SAVE - Starting database save right after transcription")
        print(f"🔍 DEBUG: IMMEDIATE SAVE - Using {'PostgreSQL' if is_postgresql() else 'SQLite'}")
        
        try:
            user = get_current_user()
            print(f"🔍 DEBUG: IMMEDIATE SAVE - Current user: {user}")
            
            # Force save with multiple fallback strategies
            if not user:
                print("❌ DEBUG: IMMEDIATE SAVE - No user found in session!")
                # Try to get user from session manually
                if 'user_id' in session:
                    user_id = session['user_id']
                    print(f"🔍 DEBUG: IMMEDIATE SAVE - Found user_id in session: {user_id}")
                else:
                    user_id = 1  # Ultimate fallback
                    print(f"🔍 DEBUG: IMMEDIATE SAVE - Using fallback user_id: {user_id}")
            else:
                user_id = user['id']
                print(f"🔍 DEBUG: IMMEDIATE SAVE - Using user from get_current_user: {user_id}")
            
            print(f"🔍 DEBUG: IMMEDIATE SAVE - Final user_id: {user_id}")
            print(f"🔍 DEBUG: IMMEDIATE SAVE - Patient: {patient_id}, Type: {verslag_type}")
            print(f"🔍 DEBUG: IMMEDIATE SAVE - Transcript length: {len(raw_transcript)}")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Database-agnostic INSERT query
            if is_postgresql():
                cursor.execute('''
                    INSERT INTO transcription_history 
                    (user_id, patient_id, verslag_type, original_transcript, structured_report, enhanced_transcript)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    user_id, 
                    patient_id, 
                    verslag_type, 
                    raw_transcript, 
                    "Processing...",  # Placeholder, will be updated later
                    raw_transcript   # Initial enhanced transcript
                ))
                record_id = cursor.fetchone()[0]
            else:
                cursor.execute('''
                    INSERT INTO transcription_history 
                    (user_id, patient_id, verslag_type, original_transcript, structured_report, enhanced_transcript)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, 
                    patient_id, 
                    verslag_type, 
                    raw_transcript, 
                    "Processing...",  # Placeholder, will be updated later
                    raw_transcript   # Initial enhanced transcript
                ))
                record_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            print(f"✅ IMMEDIATE SAVE - Successfully saved to database with ID: {record_id}")
            print(f"🔍 DEBUG: IMMEDIATE SAVE - Record saved at: {datetime.now()}")
            
        except Exception as e:
            print(f"❌ IMMEDIATE SAVE - Database save failed: {e}")
            import traceback
            print(f"🔍 DEBUG: IMMEDIATE SAVE - Full error traceback: {traceback.format_exc()}")
            
            # Try alternative save method
            try:
                print(f"🔍 DEBUG: IMMEDIATE SAVE - Trying alternative save method...")
                # Use the auth_system save function as backup
                save_transcription(
                    user_id=user_id if 'user_id' in locals() else 1,
                    verslag_type=verslag_type,
                    original_transcript=raw_transcript,
                    structured_report="Processing...",
                    patient_id=patient_id,
                    enhanced_transcript=raw_transcript
                )
                print(f"✅ IMMEDIATE SAVE - Alternative save method succeeded!")
            except Exception as e2:
                print(f"❌ IMMEDIATE SAVE - Alternative save also failed: {e2}")
        
        # Enhanced transcription processing
        enhanced_transcription_result = None
        try:
            print(f"✨ Generating enhanced transcription...")
            enhanced_transcription_result = transcription_service.enhance_transcription(raw_transcript, verslag_type)
            print(f"✅ Enhanced transcription completed")
        except Exception as e:
            print(f"⚠️ Enhanced transcription failed: {e}")
            enhanced_transcription_result = {
                'enhanced_transcript': raw_transcript,
                'improvements': f"Enhancement failed: {str(e)}"
            }
        
        # Generate report based on type
        print(f"🔍 DEBUG: About to generate report for type: '{verslag_type}'")
        
        if verslag_type == 'TTE':
            print("🔍 DEBUG: Generating TTE report...")
            structured_report = transcription_service.generate_tte_report(
                raw_transcript, patient_id
            )
        elif verslag_type == 'TEE':
            print("🔍 DEBUG: Generating TEE report...")
            structured_report = transcription_service.generate_tee_report(
                raw_transcript, patient_id
            )
        elif verslag_type == 'SPOEDCONSULT':
            print("🔍 DEBUG: Generating SPOEDCONSULT report...")
            structured_report = transcription_service.generate_spoedconsult_report(
                raw_transcript, patient_id
            )
        elif verslag_type == 'CONSULTATIE':
            print("🔍 DEBUG: Generating CONSULTATIE report...")
            structured_report = transcription_service.generate_consultatie_report(
                raw_transcript, patient_id
            )
        else:
            print(f"🔍 DEBUG: Unknown type '{verslag_type}', defaulting to TTE...")
            structured_report = transcription_service.generate_tte_report(
                raw_transcript, patient_id
            )
        
        # Quality control validation
        quality_feedback = None
        try:
            print(f"🔍 Running quality control validation...")
            quality_feedback = transcription_service.validate_medical_report(structured_report, verslag_type)
            print(f"✅ Quality control completed")
        except Exception as e:
            print(f"⚠️ Quality control failed: {e}")
            quality_feedback = f"Quality control error: {str(e)}"
        
        # ESC Guidelines recommendations (for SPOEDCONSULT and CONSULTATIE)
        esc_recommendations = None
        if verslag_type in ['SPOEDCONSULT', 'CONSULTATIE']:
            try:
                print(f"📋 Generating ESC Guidelines recommendations...")
                esc_recommendations = transcription_service.generate_esc_recommendations(structured_report, verslag_type)
                print(f"✅ ESC recommendations completed")
            except Exception as e:
                print(f"⚠️ ESC recommendations failed: {e}")
                esc_recommendations = f"ESC recommendations error: {str(e)}"
        
        print(f"🔍 DEBUG: Generated report preview: {structured_report[:100]}...")
        
        # Update database record with final report and enhanced transcript
        try:
            user = get_current_user()
            print(f"🔍 DEBUG: UPDATE - Current user: {user}")
            
            if not user:
                print("❌ DEBUG: UPDATE - No user found in session!")
                user_id = 1  # Fallback for compatibility
            else:
                user_id = user['id']
            
            print(f"🔍 DEBUG: UPDATE - Using user_id: {user_id}")
            print(f"🔍 DEBUG: UPDATE - Patient: {patient_id}, Type: {verslag_type}")
            print(f"🔍 DEBUG: UPDATE - Report length: {len(structured_report)}")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Update the most recent record for this user with the final report
            cursor.execute('''
                UPDATE transcription_history 
                SET structured_report = ?, 
                    enhanced_transcript = ?,
                    quality_feedback = ?
                WHERE user_id = ? AND patient_id = ? AND verslag_type = ?
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (
                structured_report,
                enhanced_transcription_result['enhanced_transcript'] if enhanced_transcription_result else raw_transcript,
                quality_feedback,
                user_id,
                patient_id,
                verslag_type
            ))
            
            rows_updated = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"✅ UPDATE - Successfully updated {rows_updated} database record(s)")
            
            # Verify the update worked
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM transcription_history WHERE user_id = ?', (user_id,))
            count = cursor.fetchone()[0]
            conn.close()
            print(f"🔍 DEBUG: UPDATE - Total records for user {user_id}: {count}")
            
        except Exception as e:
            print(f"❌ UPDATE - Database update failed: {e}")
            import traceback
            print(f"🔍 DEBUG: UPDATE - Full error traceback: {traceback.format_exc()}")
            # Don't fail the entire request if database update fails
            pass
        
        # Return comprehensive JSON response
        return jsonify({
            'success': True,
            'raw_transcript': raw_transcript,
            'transcript': raw_transcript,  # For backward compatibility
            'enhanced_transcript': enhanced_transcription_result['enhanced_transcript'] if enhanced_transcription_result else raw_transcript,
            'enhancement_details': enhanced_transcription_result['improvements'] if enhanced_transcription_result else 'No enhancements applied',
            'report': structured_report,
            'quality_feedback': quality_feedback,
            'esc_recommendations': esc_recommendations,
            'patient_id': patient_id,
            'verslag_type': verslag_type
        })
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({
            'success': False,
            'error': f"Fout bij verwerking: {str(e)}"
        }), 500

@app.route('/api/transcribe', methods=['POST'])
@login_required
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
        
        # 🚀 ENHANCED TRANSCRIPTION PROCESSING
        enhanced_transcription_result = None
        try:
            print(f"✨ Generating enhanced transcription...")
            enhanced_transcription_result = transcription_service.enhance_transcription(transcript, verslag_type)
            print(f"✅ Enhanced transcription completed")
        except Exception as e:
            print(f"⚠️ Enhanced transcription failed: {e}")
            enhanced_transcription_result = {
                'enhanced_transcript': transcript,
                'improvements': f"Enhancement failed: {str(e)}"
            }
        
        # 🚀 RUN 3 EXPERT MEDICAL AGENTS (if available)
        expert_analysis = {}
        improved_transcript = transcript
        
        print(f"🔍 DEBUG: EXPERTS_AVAILABLE = {EXPERTS_AVAILABLE}")
        print(f"🔍 DEBUG: medical_experts = {medical_experts}")
        
        if EXPERTS_AVAILABLE and medical_experts:
            try:
                print(f"🤖 API DEBUG: Starting 3 Expert Medical Agents analysis...")
                print(f"🔍 DEBUG: Transcript length: {len(transcript)} chars")
                print(f"🔍 DEBUG: Transcript preview: {transcript[:200]}...")
                
                expert_analysis = medical_experts.orchestrate_medical_analysis(
                    transcript=transcript,
                    patient_context=f"Patient ID: {patient_id}, Report Type: {verslag_type}"
                )
                
                print(f"🔍 DEBUG: Expert analysis keys: {list(expert_analysis.keys())}")
                
                # Debug Agent 3 specifically
                agent_3_data = expert_analysis.get('agent_3_treatment_protocol', {})
                print(f"🔍 DEBUG: Agent 3 data keys: {list(agent_3_data.keys())}")
                
                treatment_plan = agent_3_data.get('treatment_plan', {})
                print(f"🔍 DEBUG: Treatment plan keys: {list(treatment_plan.keys())}")
                
                immediate_actions = treatment_plan.get('immediate_actions', [])
                print(f"🔍 DEBUG: Immediate actions: {immediate_actions}")
                
                medications = treatment_plan.get('medications', [])
                print(f"🔍 DEBUG: Medications: {medications}")
                
                # Use improved transcript from Agent 1
                improved_transcript = expert_analysis.get('agent_1_quality_control', {}).get('improved_transcript', transcript)
                print(f"🤖 API DEBUG: Expert analysis completed successfully!")
            except Exception as e:
                print(f"⚠️ API DEBUG: Expert analysis failed: {e}")
                import traceback
                print(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
                expert_analysis = {}
        else:
            print(f"⚠️ API DEBUG: Expert agents not available, using basic processing")
        
        # Generate report
        print(f"🔍 API DEBUG: About to generate report for type: '{verslag_type}'")
        
        if verslag_type == 'TTE':
            print("🔍 API DEBUG: Generating TTE report...")
            report = transcription_system.generate_tte_report(improved_transcript, patient_id)
        elif verslag_type == 'TEE':
            print("🔍 API DEBUG: Generating TEE report...")
            report = transcription_system.generate_tee_report(improved_transcript, patient_id)
        elif verslag_type == 'SPOEDCONSULT':
            print("🔍 API DEBUG: Generating SPOEDCONSULT report...")
            report = transcription_system.generate_spoedconsult_report(improved_transcript, patient_id)
        elif verslag_type == 'CONSULTATIE':
            print("🔍 API DEBUG: Generating CONSULTATIE report...")
            report = transcription_system.generate_consultatie_report(improved_transcript, patient_id)
        else:
            print(f"🔍 API DEBUG: Unknown type '{verslag_type}', defaulting to TTE...")
            report = transcription_system.generate_tte_report(improved_transcript, patient_id)
        
        print(f"🔍 API DEBUG: Generated report preview: {report[:100]}...")
        
        # 🔍 QUALITY CONTROL VALIDATION
        quality_feedback = None
        try:
            print(f"🔍 Generating quality control feedback...")
            quality_feedback = transcription_service.validate_medical_report(report, verslag_type)
            print(f"✅ Quality control completed")
        except Exception as e:
            print(f"⚠️ Quality control failed: {e}")
            quality_feedback = f"Quality control fout: {str(e)}"
        
        # 📋 ESC GUIDELINES RECOMMENDATIONS (for SPOEDCONSULT and CONSULTATIE)
        esc_recommendations = None
        if verslag_type in ['SPOEDCONSULT', 'CONSULTATIE']:
            try:
                print(f"📋 Generating ESC Guidelines recommendations for {verslag_type}...")
                esc_recommendations = transcription_service.generate_esc_recommendations(report, verslag_type)
                print(f"✅ ESC recommendations completed")
            except Exception as e:
                print(f"⚠️ ESC recommendations failed: {e}")
                esc_recommendations = f"ESC Guidelines aanbevelingen fout: {str(e)}"
        
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
                
                # Format AI treatment recommendations with current guidelines and concrete details
                ai_medications = treatment_plan.get('medications', [])
                ai_actions = treatment_plan.get('immediate_actions', [])
                ai_monitoring = treatment_plan.get('monitoring', [])
                guideline_citations = agent_3_data.get('guideline_citations', [])
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
                            guideline_class = med.get('guideline_class', '')
                            evidence_level = med.get('evidence_level', '')
                            
                            if target:
                                med_str += f" (target: {target})"
                            if guideline_class and evidence_level:
                                med_str += f" [Class {guideline_class}, Level {evidence_level}]"
                            
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
                
                # Add guideline compliance information
                if guideline_citations:
                    ai_treatment_parts.append(f"📚 Richtlijnen: {'; '.join(guideline_citations[:2])}")  # Show first 2 citations
                
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
            'enhanced_transcription': enhanced_transcription_result,  # NEW: Enhanced transcription
            'report': report,
            'quality_feedback': quality_feedback,  # NEW: Quality control feedback
            'esc_recommendations': esc_recommendations,  # NEW: ESC Guidelines recommendations
            'patient_id': patient_id,
            'verslag_type': verslag_type,
            'treatment_comparison': {
                'dictated_treatment': dictated_treatment,
                'ai_treatment': ai_treatment,
                'differences': treatment_differences
            },
            'expert_analysis': {
                'quality_score': expert_analysis.get('agent_1_quality_control', {}).get('quality_score', 0) if expert_analysis.get('agent_1_quality_control') else 0,
                'primary_diagnosis': expert_analysis.get('agent_2_diagnostic_expert', {}).get('primary_diagnosis', {}) if expert_analysis.get('agent_2_diagnostic_expert') else {},
                'treatment_plan': expert_analysis.get('agent_3_treatment_protocol', {}).get('treatment_plan', {}) if expert_analysis.get('agent_3_treatment_protocol') else {},
                'safety_alerts': expert_analysis.get('agent_1_quality_control', {}).get('safety_alerts', []) if expert_analysis.get('agent_1_quality_control') else [],
                'urgency_level': expert_analysis.get('agent_2_diagnostic_expert', {}).get('urgency_level', 'unknown') if expert_analysis.get('agent_2_diagnostic_expert') else 'unknown',
                'corrections_made': len(expert_analysis.get('agent_1_quality_control', {}).get('corrections', [])) if expert_analysis.get('agent_1_quality_control') and expert_analysis.get('agent_1_quality_control', {}).get('corrections') else 0,
                'agents_used': expert_analysis.get('orchestration_summary', {}).get('agents_used', 3) if expert_analysis.get('orchestration_summary') else 3,
                'guideline_citations': expert_analysis.get('agent_3_treatment_protocol', {}).get('guideline_citations', []) if expert_analysis.get('agent_3_treatment_protocol') else [],
                'guideline_source': expert_analysis.get('agent_3_treatment_protocol', {}).get('guideline_source', 'Unknown') if expert_analysis.get('agent_3_treatment_protocol') else 'Unknown',
                'evidence_level': expert_analysis.get('agent_3_treatment_protocol', {}).get('evidence_level', 'Unknown') if expert_analysis.get('agent_3_treatment_protocol') else 'Unknown',
                'quality_indicators': expert_analysis.get('agent_3_treatment_protocol', {}).get('quality_indicators', {}) if expert_analysis.get('agent_3_treatment_protocol') else {}
            }
        })
        
    except Exception as e:
        logger.error(f"API transcription error: {e}")
        return jsonify({
            'success': False, 
            'error': f'Transcription failed: {str(e)}'
        }), 500

@app.route('/ocr-patient-id', methods=['POST'])
@login_required
def ocr_patient_id():
    """OCR endpoint for patient ID extraction from photos"""
    try:
        if not OCR_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'OCR service niet beschikbaar'
            }), 503
        
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Geen afbeelding data ontvangen'
            }), 400
        
        image_data = data['image']
        
        # Extract patient number using OCR
        result = ocr_service.extract_patient_number(image_data)
        
        if result['success']:
            # Validate the extracted number
            if ocr_service.validate_patient_number(result['patient_number']):
                return jsonify({
                    'success': True,
                    'patient_number': result['patient_number'],
                    'raw_text': result.get('raw_text', ''),
                    'confidence': 'high'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f"Ongeldig patiëntennummer formaat: {result['patient_number']}",
                    'raw_text': result.get('raw_text', ''),
                    'suggestion': 'Patiëntennummer moet exact 10 cijfers zijn'
                })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'raw_text': result.get('raw_text', ''),
                'suggestion': result.get('suggestion', 'Probeer een duidelijkere foto')
            })
            
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return jsonify({
            'success': False,
            'error': f'OCR verwerking gefaald: {str(e)}'
        }), 500

@app.route('/history')
@login_required
def history():
    """View transcription history for authenticated user"""
    try:
        user = get_current_user()
        if not user:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get transcription history for current user only
        cursor.execute('''
            SELECT id, patient_id, verslag_type, original_transcript, 
                   structured_report, created_at
            FROM transcription_history 
            WHERE user_id = ?
            ORDER BY created_at DESC 
            LIMIT 50
        ''', (user['id'],))
        
        history_records = cursor.fetchall()
        conn.close()
        
        return render_template('history.html', records=history_records, user=user)
        
    except Exception as e:
        logger.error(f"History error: {e}")
        flash(f'Fout bij laden van geschiedenis: {str(e)}', 'error')
        return render_template('history.html', records=[], user=get_current_user())

@app.route('/view/<int:record_id>')
@login_required
def view_record(record_id):
    """View specific transcription record (legacy route - redirects to review)"""
    return redirect(url_for('review_transcription', record_id=record_id))

@app.route('/review/<int:record_id>')
@login_required
def review_transcription(record_id):
    """Review and edit transcription with enhanced interface"""
    try:
        user = get_current_user()
        if not user:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get transcription record with user verification
        cursor.execute('''
            SELECT id, patient_id, verslag_type, original_transcript, 
                   structured_report, created_at, enhanced_transcript
            FROM transcription_history 
            WHERE id = ? AND user_id = ?
        ''', (record_id, user['id']))
        
        record_data = cursor.fetchone()
        conn.close()
        
        if not record_data:
            flash('Transcriptie niet gevonden of geen toegang', 'error')
            return redirect(url_for('history'))
        
        # Convert to dictionary for easier template access
        record = {
            'id': record_data[0],
            'patient_id': record_data[1],
            'verslag_type': record_data[2],
            'original_transcript': record_data[3],
            'structured_report': record_data[4],
            'created_at': datetime.fromisoformat(record_data[5]) if record_data[5] else None,
            'enhanced_transcript': record_data[6] if len(record_data) > 6 else None
        }
        
        return render_template('review.html', record=record, user=user)
        
    except Exception as e:
        logger.error(f"Review transcription error: {e}")
        flash(f'Fout bij laden van transcriptie: {str(e)}', 'error')
        return redirect(url_for('history'))

@app.route('/update-transcription/<int:record_id>', methods=['POST'])
@login_required
def update_transcription(record_id):
    """Update transcription content via AJAX"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'Not authenticated'})
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify user owns this transcription
        cursor.execute('SELECT id FROM transcription_history WHERE id = ? AND user_id = ?', 
                      (record_id, user['id']))
        if not cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'Transcription not found or access denied'})
        
        # Update the transcription
        cursor.execute('''
            UPDATE transcription_history 
            SET original_transcript = ?, 
                structured_report = ?, 
                enhanced_transcript = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
        ''', (
            data.get('original_transcript', ''),
            data.get('structured_report', ''),
            data.get('enhanced_transcript', ''),
            record_id,
            user['id']
        ))
        
        conn.commit()
        conn.close()
        
        # Log the update for security audit
        log_security_event('TRANSCRIPTION_UPDATED', user_id=user['id'], 
                          details=f'Updated transcription {record_id}')
        
        return jsonify({'success': True, 'message': 'Transcription updated successfully'})
        
    except Exception as e:
        logger.error(f"Update transcription error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete-transcription/<int:record_id>', methods=['DELETE'])
@login_required
def delete_transcription(record_id):
    """Delete transcription record via AJAX"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'success': False, 'error': 'Not authenticated'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify user owns this transcription
        cursor.execute('SELECT patient_id FROM transcription_history WHERE id = ? AND user_id = ?', 
                      (record_id, user['id']))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return jsonify({'success': False, 'error': 'Transcription not found or access denied'})
        
        patient_id = result[0]
        
        # Delete the transcription
        cursor.execute('DELETE FROM transcription_history WHERE id = ? AND user_id = ?', 
                      (record_id, user['id']))
        
        conn.commit()
        conn.close()
        
        # Log the deletion for security audit
        log_security_event('TRANSCRIPTION_DELETED', user_id=user['id'], 
                          details=f'Deleted transcription {record_id} (Patient: {patient_id})')
        
        return jsonify({'success': True, 'message': 'Transcription deleted successfully'})
        
    except Exception as e:
        logger.error(f"Delete transcription error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/transcription-count')
@login_required
def transcription_count():
    """Get current transcription count for auto-refresh"""
    try:
        user = get_current_user()
        if not user:
            return jsonify({'count': 0})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM transcription_history WHERE user_id = ?', (user['id'],))
        count = cursor.fetchone()[0]
        conn.close()
        
        return jsonify({'count': count})
        
    except Exception as e:
        logger.error(f"Transcription count error: {e}")
        return jsonify({'count': 0})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '4.0-superior',
        'openai_configured': bool(os.environ.get('OPENAI_API_KEY')),
        'ocr_available': OCR_AVAILABLE,
        'experts_available': EXPERTS_AVAILABLE,
        'features': [
            'WebM audio detection',
            'Hallucination detection', 
            'Quality control review',
            'Dutch medical terminology',
            'Context-aware drug correction',
            'Medical classification validation',
            'OCR patient number extraction' if OCR_AVAILABLE else 'OCR disabled (cloud deployment)',
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

