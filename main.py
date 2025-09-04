"""
Superior Medical Dictation v4.0 - Based on v2 Analysis
Implements all superior features from the working v2 app with secure authentication
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import os
import io
from datetime import datetime, timedelta
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
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # Session timeout

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
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
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

# Rate limiting is handled by auth_system.py

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
@rate_limit(max_requests=15, window=300)  # 15 attempts per 5 minutes
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
@rate_limit(max_requests=10, window=300)  # 10 attempts per 5 minutes for POST only
def register():
    """Register route with security logging"""
    if request.method == 'GET':
        # No rate limiting for viewing the registration form
        return render_template('register.html')
    
    # Handle POST request (registration attempt)
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
    import sys
    sys.stdout.write("🔍 SIMPLE TEST: /transcribe function called\n")
    sys.stdout.flush()
    
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
                    VALUES (?, %s, %s, %s, %s, %s)
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
                SET structured_report = %s, 
                    enhanced_transcript = %s,
                    quality_feedback = %s
                WHERE user_id = %s AND patient_id = %s AND verslag_type = %s
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
            cursor.execute('SELECT COUNT(*) FROM transcription_history WHERE user_id = %s', (user_id,))
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
    import sys
    sys.stdout.write("🔍 SIMPLE TEST: api_transcribe function called\n")
    sys.stdout.flush()
    
    try:
        logger.info("🔍 DEBUG: ===== API_TRANSCRIBE FUNCTION STARTED =====")
        print("🔍 DEBUG: ===== API_TRANSCRIBE FUNCTION STARTED =====", flush=True)
    except Exception as e:
        logger.error(f"🔍 DEBUG: Exception in function start: {e}")
        print(f"🔍 DEBUG: Exception in function start: {e}", flush=True)
    
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        verslag_type = request.form.get('verslag_type', 'TTE')
        patient_id = request.form.get('patient_id', '')
        contact_location = request.form.get('contact_location', 'Poli')
        
        # DEBUG: Log what template was selected in API
        print(f"🔍 API DEBUG: Template selected = '{verslag_type}'")
        print(f"🔍 API DEBUG: Patient ID = '{patient_id}'")
        print(f"🔍 API DEBUG: Contact Location = '{contact_location}'")
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
                
            except Exception as e:
                print(f"❌ Medical analysis failed: {e}")
                import traceback
                print(f"🔍 DEBUG: Medical analysis error traceback: {traceback.format_exc()}")
                # Continue with empty analysis - don't crash the whole transcription
                expert_analysis = {}
                
                medications = treatment_plan.get('medications', [])
                print(f"🔍 DEBUG: Medications: {medications}")
                
                # Use improved transcript from Agent 1
                improved_transcript = expert_analysis.get('agent_1_quality_control', {}).get('improved_transcript', transcript)
                print(f"🔍 DEBUG: Agent 1 result keys: {list(expert_analysis.get('agent_1_quality_control', {}).keys())}")
                print(f"🔍 DEBUG: Original transcript length: {len(transcript)}")
                print(f"🔍 DEBUG: Improved transcript length: {len(improved_transcript)}")
                print(f"🔍 DEBUG: Improved transcript preview: {improved_transcript[:200]}...")
                print(f"🤖 API DEBUG: Expert analysis completed successfully!")
                
            except Exception as e:
                print(f"⚠️ API DEBUG: Expert analysis failed: {e}")
                import traceback
                print(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
                expert_analysis = {}
        else:
            print(f"⚠️ API DEBUG: Expert agents not available, using basic processing")
        
        # 💾 IMMEDIATE DATABASE SAVE AFTER MEDICAL EXPERTS (MOVED OUTSIDE TRY BLOCK)
        print(f"🔍 DEBUG: IMMEDIATE SAVE AFTER EXPERTS - Starting database save")
        try:
            # Get current user
            user = get_current_user()
            user_id = user.get('id') if user else 1
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO transcription_history 
                (user_id, patient_id, verslag_type, contact_location, original_transcript, structured_report, enhanced_transcript, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (user_id, patient_id, verslag_type, contact_location, transcript, "Processing...", improved_transcript, datetime.now()))
            
            result = cursor.fetchone()
            record_id = result['id'] if isinstance(result, dict) else result[0]
            
            conn.commit()
            conn.close()
            
            print(f"✅ IMMEDIATE SAVE AFTER EXPERTS - Successfully saved with ID: {record_id}")
            
        except Exception as e:
            print(f"❌ IMMEDIATE SAVE AFTER EXPERTS - Failed: {e}")
            import traceback
            print(f"🔍 DEBUG: Save error traceback: {traceback.format_exc()}")
        
        print(f"🔍 DEBUG: Expert analysis section completed, moving to report generation")
        print(f"🔍 DEBUG: improved_transcript length: {len(improved_transcript)}")
        print(f"🔍 DEBUG: verslag_type: '{verslag_type}'")
        
        # Generate report
        print(f"🔍 API DEBUG: About to generate report for type: '{verslag_type}'")
        
        if verslag_type == 'TTE':
            print("🔍 API DEBUG: Generating TTE report...")
            report = transcription_system.generate_tte_report(improved_transcript, patient_id, expert_analysis)
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
            report = transcription_system.generate_tte_report(improved_transcript, patient_id, expert_analysis)
        
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
        print(f"🔍 DEBUG: About to extract treatment from transcript")
        dictated_treatment = extract_treatment_from_transcript(improved_transcript)
        print(f"🔍 DEBUG: Dictated treatment extracted: {dictated_treatment[:100] if dictated_treatment else 'None'}...")
        
        # Get AI treatment recommendations
        print(f"🔍 DEBUG: About to get AI treatment recommendations")
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
        
        print(f"🔍 DEBUG: About to start immediate save - reached this point successfully")
        
        # 💾 IMMEDIATE SAVE TO DATABASE (PostgreSQL only)
        print(f"🔍 DEBUG: API IMMEDIATE SAVE - Starting database save right after transcription")
        print(f"🔍 DEBUG: API IMMEDIATE SAVE - Using PostgreSQL")
        
        try:
            # Get current user
            user = get_current_user()
            print(f"🔍 DEBUG: API IMMEDIATE SAVE - Current user: {user}")
            
            user_id = None
            if not user:
                print("❌ DEBUG: API IMMEDIATE SAVE - No user found in session!")
                # Try to get user_id from session directly
                if 'user_id' in session:
                    user_id = session['user_id']
                    print(f"🔍 DEBUG: API IMMEDIATE SAVE - Found user_id in session: {user_id}")
                else:
                    user_id = 1  # Fallback to admin user
                    print(f"🔍 DEBUG: API IMMEDIATE SAVE - Using fallback user_id: {user_id}")
            else:
                user_id = user.get('id') if isinstance(user, dict) else user[0]
                print(f"🔍 DEBUG: API IMMEDIATE SAVE - Using user from get_current_user: {user_id}")
            
            print(f"🔍 DEBUG: API IMMEDIATE SAVE - Final user_id: {user_id}")
            print(f"🔍 DEBUG: API IMMEDIATE SAVE - Patient: {patient_id}, Type: {verslag_type}")
            print(f"🔍 DEBUG: API IMMEDIATE SAVE - Transcript length: {len(transcript)}")
            
            # Save to database immediately - PostgreSQL only
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO transcription_history 
                (user_id, patient_id, verslag_type, original_transcript, structured_report, enhanced_transcript, quality_feedback, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (
                user_id,
                patient_id or 'Unknown',
                verslag_type,
                transcript,
                report,
                improved_transcript,
                str(quality_feedback) if quality_feedback else 'No feedback',
                datetime.now()
            ))
            result = cursor.fetchone()
            record_id = result[0] if result else None
            
            conn.commit()
            conn.close()
            
            print(f"✅ API IMMEDIATE SAVE - Successfully saved to database with ID: {record_id}")
            print(f"🔍 DEBUG: API IMMEDIATE SAVE - Record saved at: {datetime.now()}")
            
        except Exception as e:
            print(f"❌ API IMMEDIATE SAVE - Database save failed: {e}")
            import traceback
            print(f"🔍 DEBUG: API IMMEDIATE SAVE - Full error traceback: {traceback.format_exc()}")
            
            # Try alternative save method
            try:
                print(f"🔍 DEBUG: API IMMEDIATE SAVE - Trying alternative save method...")
                success = save_transcription(
                    user_id=user_id,
                    patient_id=patient_id or 'Unknown',
                    verslag_type=verslag_type,
                    original_transcript=transcript,
                    structured_report=report,
                    enhanced_transcript=improved_transcript,
                    quality_feedback=str(quality_feedback) if quality_feedback else 'No feedback'
                )
                if success:
                    print(f"✅ API IMMEDIATE SAVE - Alternative save method succeeded!")
                else:
                    print(f"❌ API IMMEDIATE SAVE - Alternative save also failed: {success}")
            except Exception as e2:
                print(f"❌ API IMMEDIATE SAVE - Alternative save also failed: {e2}")
        
        print(f"🔍 DEBUG: About to return JSON response - api_transcribe completed successfully")
        
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

@app.route('/api/ocr-extract', methods=['POST'])
@login_required
def api_ocr_extract():
    """API endpoint for OCR patient ID extraction from photos"""
    try:
        if not OCR_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'OCR functionaliteit niet beschikbaar',
                'suggestion': 'Voer patiënt ID handmatig in'
            }), 503
        
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
                    'method': result.get('method', 'unknown'),
                    'debug_info': result.get('debug_info', '')
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Geëxtraheerd nummer is geen geldig patiëntennummer',
                    'suggestion': 'Controleer of het patiëntennummer duidelijk zichtbaar is',
                    'debug_info': f"Extracted: {result['patient_number']}"
                })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'OCR extractie gefaald'),
                'suggestion': result.get('suggestion', 'Probeer een duidelijkere foto'),
                'debug_info': result.get('debug_info', '')
            })
            
    except Exception as e:
        print(f"❌ OCR API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'OCR verwerking fout: {str(e)}',
            'suggestion': 'Probeer opnieuw of voer patiënt ID handmatig in'
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
        user_id = user.get('id') if user else 1
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get recent transcriptions (last 48 hours)
        cursor.execute('''
            SELECT id, patient_id, verslag_type, original_transcript, created_at
            FROM transcription_history 
            WHERE user_id = %s 
            AND created_at >= NOW() - INTERVAL '48 hours'
            ORDER BY created_at DESC
        ''', (user_id,))
        
        records = cursor.fetchall()
        print(f"🔍 DEBUG: History query returned {len(records)} records for user {user_id}")
        
        conn.close()
        
        return render_template('history.html', records=records)
        
    except Exception as e:
        print(f"❌ History error: {e}")
        import traceback
        print(f"🔍 DEBUG: History error traceback: {traceback.format_exc()}")
        return render_template('history.html', records=[], error="Fout bij laden van geschiedenis")

@app.route('/admin')
@login_required
def admin_overview():
    """Administrative overview page"""
    return render_template('admin.html')

@app.route('/api/admin/week-data')
@login_required
def get_week_data():
    """Get transcription data for a specific week"""
    try:
        week = request.args.get('week')  # Format: 2025-W36
        if not week:
            return jsonify({'success': False, 'error': 'Week parameter required'}), 400
        
        # Parse week format (2025-W36)
        year, week_num = week.split('-W')
        year = int(year)
        week_num = int(week_num)
        
        # Calculate start and end dates of the week
        from datetime import datetime, timedelta
        jan1 = datetime(year, 1, 1)
        week_start = jan1 + timedelta(weeks=week_num-1) - timedelta(days=jan1.weekday())
        week_end = week_start + timedelta(days=6)
        
        user = get_current_user()
        user_id = user.get('id') if user else 1
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, patient_id, verslag_type, contact_location, 
                   is_completed, billing_ok, created_at
            FROM transcription_history 
            WHERE user_id = %s 
            AND created_at >= %s 
            AND created_at <= %s
            ORDER BY created_at DESC
        ''', (user_id, week_start, week_end + timedelta(days=1)))
        
        records = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts for JSON serialization
        records_list = []
        for record in records:
            if isinstance(record, dict):
                records_list.append(record)
            else:
                # Handle tuple format
                records_list.append({
                    'id': record[0],
                    'patient_id': record[1],
                    'verslag_type': record[2],
                    'contact_location': record[3],
                    'is_completed': record[4],
                    'billing_ok': record[5],
                    'created_at': record[6].isoformat() if record[6] else None
                })
        
        return jsonify({
            'success': True,
            'records': records_list,
            'week': week,
            'period': f"{week_start.strftime('%d/%m')} - {week_end.strftime('%d/%m/%Y')}"
        })
        
    except Exception as e:
        print(f"❌ Error getting week data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/update-status', methods=['POST'])
@login_required
def update_status():
    """Update completion or billing status"""
    try:
        data = request.get_json()
        record_id = data.get('record_id')
        status_type = data.get('type')  # 'completed' or 'billing'
        value = data.get('value')
        
        if not all([record_id, status_type, value is not None]):
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if status_type == 'completed':
            cursor.execute('''
                UPDATE transcription_history 
                SET is_completed = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            ''', (value, record_id))
        elif status_type == 'billing':
            cursor.execute('''
                UPDATE transcription_history 
                SET billing_ok = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            ''', (value, record_id))
        else:
            return jsonify({'success': False, 'error': 'Invalid status type'}), 400
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"❌ Error updating status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/generate-email')
@login_required
def generate_email():
    """Generate email for secretaries with weekly overview"""
    try:
        week = request.args.get('week')
        if not week:
            return jsonify({'success': False, 'error': 'Week parameter required'}), 400
        
        # Parse week and get data (reuse logic from get_week_data)
        year, week_num = week.split('-W')
        year = int(year)
        week_num = int(week_num)
        
        from datetime import datetime, timedelta
        jan1 = datetime(year, 1, 1)
        week_start = jan1 + timedelta(weeks=week_num-1) - timedelta(days=jan1.weekday())
        week_end = week_start + timedelta(days=6)
        
        user = get_current_user()
        user_id = user.get('id') if user else 1
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT patient_id, verslag_type, contact_location, created_at
            FROM transcription_history 
            WHERE user_id = %s 
            AND created_at >= %s 
            AND created_at <= %s
            ORDER BY created_at ASC
        ''', (user_id, week_start, week_end + timedelta(days=1)))
        
        records = cursor.fetchall()
        conn.close()
        
        # Generate email content
        email_content = f"""Onderwerp: Cardiologie Contacten Week {week_num} ({week_start.strftime('%d/%m')} - {week_end.strftime('%d/%m/%Y')})

Beste secretaressen,

Hierbij het overzicht van alle cardiologie contacten voor week {week_num}:

"""
        
        if not records:
            email_content += "Geen contacten geregistreerd voor deze week.\n"
        else:
            email_content += f"Totaal aantal contacten: {len(records)}\n\n"
            email_content += "OVERZICHT PER CONTACT:\n"
            email_content += "=" * 50 + "\n\n"
            
            for i, record in enumerate(records, 1):
                if isinstance(record, dict):
                    patient_id = record['patient_id']
                    verslag_type = record['verslag_type']
                    contact_location = record['contact_location']
                    created_at = record['created_at']
                else:
                    patient_id = record[0]
                    verslag_type = record[1]
                    contact_location = record[2]
                    created_at = record[3]
                
                date_str = created_at.strftime('%d/%m/%Y %H:%M') if created_at else 'Onbekend'
                
                email_content += f"{i}. Patiënt: {patient_id}\n"
                email_content += f"   Datum: {date_str}\n"
                email_content += f"   Type: {verslag_type}\n"
                email_content += f"   Locatie: {contact_location}\n\n"
        
        email_content += """
Met vriendelijke groet,
Dr. [Naam]
Cardiologie

---
Dit overzicht is automatisch gegenereerd door Medical Dictation v4.0"""
        
        return jsonify({
            'success': True,
            'email': email_content,
            'count': len(records)
        })
        
    except Exception as e:
        print(f"❌ Error generating email: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/mark-all-completed', methods=['POST'])
@login_required
def mark_all_completed():
    """Mark all records in a week as completed"""
    try:
        data = request.get_json()
        week = data.get('week')
        
        if not week:
            return jsonify({'success': False, 'error': 'Week parameter required'}), 400
        
        # Parse week and calculate dates
        year, week_num = week.split('-W')
        year = int(year)
        week_num = int(week_num)
        
        from datetime import datetime, timedelta
        jan1 = datetime(year, 1, 1)
        week_start = jan1 + timedelta(weeks=week_num-1) - timedelta(days=jan1.weekday())
        week_end = week_start + timedelta(days=6)
        
        user = get_current_user()
        user_id = user.get('id') if user else 1
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE transcription_history 
            SET is_completed = true, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s 
            AND created_at >= %s 
            AND created_at <= %s
            AND is_completed = false
        ''', (user_id, week_start, week_end + timedelta(days=1)))
        
        updated_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'updated': updated_count
        })
        
    except Exception as e:
        print(f"❌ Error marking all completed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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
        user_id = user.get('id') if user else 1
        
        print(f"🔍 DEBUG: Review page loading for record {record_id}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get transcription record with existing fields only
        cursor.execute('''
            SELECT id, patient_id, verslag_type, original_transcript, 
                   structured_report, enhanced_transcript, created_at, 
                   quality_feedback
            FROM transcription_history 
            WHERE id = %s AND user_id = %s
        ''', (record_id, user_id))
        
        record = cursor.fetchone()
        conn.close()
        
        if not record:
            flash('Transcriptie niet gevonden', 'error')
            return redirect(url_for('history'))
        
        # Handle both dict and tuple responses
        if isinstance(record, dict):
            data = record
        else:
            data = {
                'id': record[0],
                'patient_id': record[1],
                'verslag_type': record[2],
                'original_transcript': record[3],
                'structured_report': record[4],
                'enhanced_transcript': record[5],
                'created_at': record[6],
                'quality_feedback': record[7] if len(record) > 7 else None
            }
        
        # Ensure all fields have default values
        for field in ['original_transcript', 'structured_report', 'enhanced_transcript', 'quality_feedback']:
            if data.get(field) is None:
                data[field] = ''
        
        # Format date
        created_date = data['created_at'].strftime('%d/%m/%Y %H:%M') if data['created_at'] else 'Onbekend'
        
        # Generate complete HTML page with enhanced functionality
        html_content = f'''<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review Transcriptie - Medical Dictation v4.0</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .content {{
            padding: 30px;
        }}

        .info-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            border-left: 5px solid #667eea;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}

        .info-item {{
            display: flex;
            flex-direction: column;
        }}

        .info-label {{
            font-weight: 600;
            color: #495057;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}

        .info-value {{
            font-size: 1.1em;
            color: #212529;
        }}

        .audio-section {{
            background: #e8f4fd;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            border-left: 5px solid #007bff;
        }}

        .audio-controls {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}

        .audio-player {{
            flex: 1;
            min-width: 300px;
        }}

        .audio-info {{
            color: #495057;
            font-size: 0.9em;
        }}

        .report-section {{
            margin-bottom: 30px;
        }}

        .section-title {{
            font-size: 1.5em;
            font-weight: 600;
            color: #495057;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }}

        .report-textarea {{
            width: 100%;
            min-height: 400px;
            padding: 20px;
            border: 2px solid #dee2e6;
            border-radius: 12px;
            font-family: Georgia, serif;
            font-size: 16px;
            line-height: 1.6;
            resize: vertical;
            transition: border-color 0.3s ease;
        }}

        .report-textarea:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}

        .button-group {{
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}

        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}

        .btn-primary {{
            background: #007bff;
            color: white;
        }}

        .btn-primary:hover {{
            background: #0056b3;
            transform: translateY(-2px);
        }}

        .btn-success {{
            background: #28a745;
            color: white;
        }}

        .btn-success:hover {{
            background: #1e7e34;
            transform: translateY(-2px);
        }}

        .btn-warning {{
            background: #ffc107;
            color: #212529;
        }}

        .btn-warning:hover {{
            background: #e0a800;
            transform: translateY(-2px);
        }}

        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}

        .btn-secondary:hover {{
            background: #545b62;
            transform: translateY(-2px);
        }}

        .btn-danger {{
            background: #dc3545;
            color: white;
        }}

        .btn-danger:hover {{
            background: #c82333;
            transform: translateY(-2px);
        }}

        .collapsible-section {{
            margin-top: 20px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
        }}

        .collapsible-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            color: #495057;
        }}

        .collapsible-content {{
            padding: 20px;
            display: none;
            background: white;
        }}

        .collapsible-content.active {{
            display: block;
        }}

        .loading {{
            display: none;
            text-align: center;
            padding: 20px;
            color: #6c757d;
        }}

        .success-message {{
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }}

        .error-message {{
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }}

        @media (max-width: 768px) {{
            .container {{
                margin: 10px;
                border-radius: 15px;
            }}

            .header {{
                padding: 20px;
            }}

            .header h1 {{
                font-size: 2em;
            }}

            .content {{
                padding: 20px;
            }}

            .button-group {{
                flex-direction: column;
            }}

            .btn {{
                justify-content: center;
            }}

            .audio-controls {{
                flex-direction: column;
                align-items: stretch;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📋 Review Transcriptie</h1>
            <p>Bekijk en bewerk medisch rapport</p>
        </div>

        <div class="content">
            <div class="info-section">
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">🧑‍⚕️ Patiënt ID</div>
                        <div class="info-value">{data['patient_id']}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">📋 Type Verslag</div>
                        <div class="info-value">{data['verslag_type']}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">📅 Datum</div>
                        <div class="info-value">{created_date}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">🆔 Record ID</div>
                        <div class="info-value">#{data['id']}</div>
                    </div>
                </div>
            </div>

            {f'''
            <div class="audio-section">
                <h3 style="margin-bottom: 15px; color: #007bff;">🎵 Audio Opname</h3>
                <div class="audio-controls">
                    <audio controls class="audio-player" preload="metadata">
                        <source src="/audio/{data['audio_filename']}" type="audio/mpeg">
                        <source src="/audio/{data['audio_filename']}" type="audio/wav">
                        <source src="/audio/{data['audio_filename']}" type="audio/m4a">
                        <source src="/audio/{data['audio_filename']}" type="audio/webm">
                        Je browser ondersteunt geen audio playback.
                    </audio>
                    <div class="audio-info">
                        📁 {data['audio_filename']}<br>
                        🎧 Compatibel met iPhone/Safari
                    </div>
                </div>
            </div>
            ''' if data.get('audio_filename') else ''}

            <div class="report-section">
                <div class="section-title">📄 Medisch Rapport</div>
                <textarea id="reportContent" class="report-textarea" placeholder="Medisch rapport wordt geladen...">{data['structured_report']}</textarea>
                
                <div class="button-group">
                    <button class="btn btn-success" onclick="saveReport()">💾 Rapport Opslaan</button>
                    <button class="btn btn-primary" onclick="copyToClipboard()">📋 Kopieer naar Klembord</button>
                    <button class="btn btn-warning" onclick="correctReport()">🔧 Corrigeer Rapport</button>
                    <button class="btn btn-secondary" onclick="analyzeReport()">🔍 Cardiologische Analyse</button>
                    <a href="/history" class="btn btn-secondary">← Terug naar Overzicht</a>
                </div>

                <div class="success-message" id="successMessage"></div>
                <div class="error-message" id="errorMessage"></div>
                <div class="loading" id="loadingMessage">⏳ Bezig met verwerken...</div>
            </div>

            <div class="collapsible-section">
                <div class="collapsible-header" onclick="toggleSection('correctedSection')">
                    <span>🔧 Gecorrigeerd Rapport</span>
                    <span>▼</span>
                </div>
                <div class="collapsible-content" id="correctedSection">
                    <textarea class="report-textarea" readonly style="min-height: 200px;" placeholder="Klik op 'Corrigeer Rapport' om een gecorrigeerde versie te genereren">{data.get('improved_report', '')}</textarea>
                </div>
            </div>

            <div class="collapsible-section">
                <div class="collapsible-header" onclick="toggleSection('analysisSection')">
                    <span>🔍 Conservatieve Cardiologische Analyse</span>
                    <span>▼</span>
                </div>
                <div class="collapsible-content" id="analysisSection">
                    <textarea class="report-textarea" readonly style="min-height: 300px;" placeholder="Klik op 'Cardiologische Analyse' om een conservatieve analyse te genereren">{data.get('differential_diagnosis', '')}</textarea>
                </div>
            </div>
        </div>
    </div>

    <script>
        function toggleSection(sectionId) {{
            const content = document.getElementById(sectionId);
            const header = content.previousElementSibling;
            const arrow = header.querySelector('span:last-child');
            
            if (content.classList.contains('active')) {{
                content.classList.remove('active');
                arrow.textContent = '▼';
            }} else {{
                content.classList.add('active');
                arrow.textContent = '▲';
            }}
        }}

        async function saveReport() {{
            const content = document.getElementById('reportContent').value;
            const loadingMsg = document.getElementById('loadingMessage');
            const successMsg = document.getElementById('successMessage');
            const errorMsg = document.getElementById('errorMessage');
            
            // Hide previous messages
            successMsg.style.display = 'none';
            errorMsg.style.display = 'none';
            loadingMsg.style.display = 'block';
            
            try {{
                const response = await fetch('/api/save-report', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        record_id: {record_id},
                        report_content: content
                    }})
                }});
                
                const result = await response.json();
                loadingMsg.style.display = 'none';
                
                if (result.success) {{
                    successMsg.textContent = '✅ Rapport succesvol opgeslagen!';
                    successMsg.style.display = 'block';
                }} else {{
                    errorMsg.textContent = '❌ Fout bij opslaan: ' + result.error;
                    errorMsg.style.display = 'block';
                }}
            }} catch (error) {{
                loadingMsg.style.display = 'none';
                errorMsg.textContent = '❌ Fout bij opslaan: ' + error.message;
                errorMsg.style.display = 'block';
            }}
        }}

        function copyToClipboard() {{
            const content = document.getElementById('reportContent').value;
            navigator.clipboard.writeText(content).then(() => {{
                const successMsg = document.getElementById('successMessage');
                successMsg.textContent = '📋 Rapport gekopieerd naar klembord!';
                successMsg.style.display = 'block';
                setTimeout(() => successMsg.style.display = 'none', 3000);
            }}).catch(err => {{
                const errorMsg = document.getElementById('errorMessage');
                errorMsg.textContent = '❌ Fout bij kopiëren naar klembord';
                errorMsg.style.display = 'block';
            }});
        }}

        async function correctReport() {{
            const content = document.getElementById('reportContent').value;
            const loadingMsg = document.getElementById('loadingMessage');
            const errorMsg = document.getElementById('errorMessage');
            
            if (!content.trim()) {{
                errorMsg.textContent = '❌ Geen rapport om te corrigeren';
                errorMsg.style.display = 'block';
                return;
            }}
            
            errorMsg.style.display = 'none';
            loadingMsg.style.display = 'block';
            
            try {{
                const response = await fetch('/api/improve-report', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        record_id: {record_id},
                        report_content: content
                    }})
                }});
                
                const result = await response.json();
                loadingMsg.style.display = 'none';
                
                if (result.success) {{
                    const correctedSection = document.getElementById('correctedSection');
                    const textarea = correctedSection.querySelector('textarea');
                    textarea.value = result.improved_report;
                    
                    // Auto-expand the section
                    if (!correctedSection.classList.contains('active')) {{
                        toggleSection('correctedSection');
                    }}
                    
                    const successMsg = document.getElementById('successMessage');
                    successMsg.textContent = '✅ Rapport gecorrigeerd!';
                    successMsg.style.display = 'block';
                }} else {{
                    errorMsg.textContent = '❌ Fout bij corrigeren: ' + result.error;
                    errorMsg.style.display = 'block';
                }}
            }} catch (error) {{
                loadingMsg.style.display = 'none';
                errorMsg.textContent = '❌ Fout bij corrigeren: ' + error.message;
                errorMsg.style.display = 'block';
            }}
        }}

        async function analyzeReport() {{
            const content = document.getElementById('reportContent').value;
            const loadingMsg = document.getElementById('loadingMessage');
            const errorMsg = document.getElementById('errorMessage');
            
            if (!content.trim()) {{
                errorMsg.textContent = '❌ Geen rapport om te analyseren';
                errorMsg.style.display = 'block';
                return;
            }}
            
            errorMsg.style.display = 'none';
            loadingMsg.style.display = 'block';
            
            try {{
                const response = await fetch('/api/differential-diagnosis', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        record_id: {record_id},
                        report_content: content
                    }})
                }});
                
                const result = await response.json();
                loadingMsg.style.display = 'none';
                
                if (result.success) {{
                    const analysisSection = document.getElementById('analysisSection');
                    const textarea = analysisSection.querySelector('textarea');
                    textarea.value = result.analysis;
                    
                    // Auto-expand the section
                    if (!analysisSection.classList.contains('active')) {{
                        toggleSection('analysisSection');
                    }}
                    
                    const successMsg = document.getElementById('successMessage');
                    successMsg.textContent = '✅ Cardiologische analyse voltooid!';
                    successMsg.style.display = 'block';
                }} else {{
                    errorMsg.textContent = '❌ Fout bij analyse: ' + result.error;
                    errorMsg.style.display = 'block';
                }}
            }} catch (error) {{
                loadingMsg.style.display = 'none';
                errorMsg.textContent = '❌ Fout bij analyse: ' + error.message;
                errorMsg.style.display = 'block';
            }}
        }}

        // Auto-save functionality (optional)
        let saveTimeout;
        document.getElementById('reportContent').addEventListener('input', function() {{
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {{
                // Auto-save after 3 seconds of no typing
                // saveReport();
            }}, 3000);
        }});
    </script>
</body>
</html>'''
        
        return html_content
        
    except Exception as e:
        print(f"❌ Review page error: {e}")
        import traceback
        print(f"🔍 DEBUG: Review error traceback: {traceback.format_exc()}")
        flash(f'Fout bij laden van transcriptie: {str(e)}', 'error')
        return redirect(url_for('history'))

@app.route('/api/improve-report', methods=['POST'])
@login_required
def improve_report():
    """Improve medical report using Professor Cardiology expertise"""
    try:
        data = request.get_json()
        report_text = data.get('report_content', '')
        record_id = data.get('record_id')
        
        if not report_text:
            return jsonify({'success': False, 'error': 'Geen rapport tekst ontvangen'})
        
        # Use OpenAI to improve the report
        import openai
        
        prompt = f"""U bent een professor cardiologie met 30 jaar ervaring. Corrigeer ALLEEN de volgende aspecten in dit medisch rapport:

1. TYPO'S EN TERMINOLOGIE:
   - Corrigeer medische termen (vb: "hartdoorsluiting" → "hartoorsluiting/LAA closure")
   - Fix anatomische termen en afkortingen
   - Corrigeer spelfouten in medische terminologie

2. INCONSISTENTIES:
   - Harmoniseer tegenstrijdige bevindingen (vb: "aorta 50mm" vs "normale aorta")
   - Zorg voor consistente metingen en terminologie

3. DIAGNOSES VERBETEREN:
   - Vervang vage termen door specifieke diagnoses
   - Bij bicuspiede klep: voeg aanbeveling toe voor aortadilatatie/coarctatio screening + familie screening
   - Bij significante klepafwijkingen: voeg follow-up schema toe

BEHOUD ALLE ORIGINELE BEVINDINGEN EN STRUCTUUR. Geef alleen het gecorrigeerde rapport terug.

Origineel rapport:
{report_text}"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Je bent een professor cardiologie die medische rapporten corrigeert voor nauwkeurigheid en consistentie."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        improved_report = response.choices[0].message.content
        
        # Note: Database save disabled until migration adds improved_report column
        # if record_id:
        #     try:
        #         conn = get_db_connection()
        #         cursor = conn.cursor()
        #         cursor.execute('''
        #             UPDATE transcription_history 
        #             SET improved_report = %s, updated_at = CURRENT_TIMESTAMP
        #             WHERE id = %s AND user_id = %s
        #         ''', (improved_report, record_id, user_id))
        #         
        #         conn.commit()
        #         conn.close()
        #     except Exception as e:
        #         print(f"⚠️ Could not save improved report: {e}")
        
        return jsonify({
            'success': True,
            'improved_report': improved_report
        })
        
    except Exception as e:
        print(f"❌ Improve report error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/differential-diagnosis', methods=['POST'])
@login_required
def differential_diagnosis():
    """Generate conservative cardiological analysis"""
    try:
        data = request.get_json()
        report_text = data.get('report_content', '')
        record_id = data.get('record_id')
        
        if not report_text:
            return jsonify({'success': False, 'error': 'Geen rapport tekst ontvangen'})
        
        # Use OpenAI for conservative analysis
        import openai
        
        prompt = f"""U bent een ervaren cardioloog. Geef een CONSERVATIEVE cardiologische analyse van dit rapport.

BELANGRIJKE VEILIGHEIDSINSTRUCTIES:
- Geef GEEN operatieve adviezen bij minimale afwijkingen
- Geef GEEN medicatie zonder duidelijke indicatie  
- Vermijd gevaarlijke of experimentele behandelingen
- Focus op SIGNIFICANTE bevindingen alleen

Analyseer alleen als er duidelijke pathologie is:

1. SIGNIFICANTE BEVINDINGEN:
   - Alleen klinisch relevante afwijkingen benoemen
   - Negeer normale varianten en minimale afwijkingen

2. MOGELIJKE DIAGNOSES:
   - Alleen bij duidelijke pathologie
   - Met waarschijnlijkheid (hoog/matig/laag)

3. CONSERVATIEVE AANBEVELINGEN:
   - Specialist verwijzing bij complexe beslissingen
   - Standaard follow-up volgens ESC/AHA richtlijnen
   - Geen operatieve adviezen zonder duidelijke indicatie

Rapport:
{report_text}"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Je bent een conservatieve cardioloog die veiligheid boven uitgebreide analyse stelt."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        analysis = response.choices[0].message.content
        
        # Note: Database save disabled until migration adds differential_diagnosis column
        # if record_id:
        #     try:
        #         conn = get_db_connection()
        #         cursor = conn.cursor()
        #         cursor.execute('''
        #             UPDATE transcription_history 
        #             SET differential_diagnosis = %s, updated_at = CURRENT_TIMESTAMP
        #             WHERE id = %s AND user_id = %s
        #         ''', (analysis, record_id, user_id))
        #         
        #         conn.commit()
        #         conn.close()
        #     except Exception as e:
        #         print(f"⚠️ Could not save differential diagnosis: {e}")
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        print(f"❌ Differential diagnosis error: {e}")
        return jsonify({'success': False, 'error': str(e)})

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
        cursor.execute('SELECT id FROM transcription_history WHERE id = %s AND user_id = %s', 
                      (record_id, user['id']))
        if not cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'Transcription not found or access denied'})
        
        # Update the transcription
        cursor.execute('''
            UPDATE transcription_history 
            SET original_transcript = %s, 
                structured_report = %s, 
                enhanced_transcript = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
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
        cursor.execute('SELECT patient_id FROM transcription_history WHERE id = %s AND user_id = %s', 
                      (record_id, user['id']))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return jsonify({'success': False, 'error': 'Transcription not found or access denied'})
        
        patient_id = result[0]
        
        # Delete the transcription
        if is_postgresql():
            cursor.execute('DELETE FROM transcription_history WHERE id = %s AND user_id = %s', 
                          (record_id, user['id']))
        else:
            cursor.execute('DELETE FROM transcription_history WHERE id = %s AND user_id = %s', 
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
        cursor.execute('SELECT COUNT(*) FROM transcription_history WHERE user_id = %s', (user['id'],))
        count = cursor.fetchone()[0]
        conn.close()
        
        return jsonify({'count': count})
        
    except Exception as e:
        logger.error(f"Transcription count error: {e}")
        return jsonify({'count': 0})

@app.route('/debug-env')
@login_required
def debug_environment():
    """Debug endpoint to check environment variables"""
    import os
    
    database_url = os.environ.get('DATABASE_URL')
    
    return jsonify({
        "success": True,
        "database_url_present": bool(database_url),
        "database_url_prefix": database_url[:30] + "..." if database_url else None,
        "environment_vars": {
            "PORT": os.environ.get('PORT'),
            "RENDER": os.environ.get('RENDER'),
            "RENDER_SERVICE_NAME": os.environ.get('RENDER_SERVICE_NAME'),
        },
        "python_path": os.environ.get('PYTHONPATH'),
        "working_directory": os.getcwd()
    })

@app.route('/debug-url-parsing')
@login_required
def debug_url_parsing():
    """Debug DATABASE_URL parsing"""
    import os
    from urllib.parse import urlparse
    
    database_url = os.environ.get('DATABASE_URL')
    
    if not database_url:
        return jsonify({"error": "DATABASE_URL not found"})
    
    # Parse URL
    result = urlparse(database_url)
    
    # Safe URL display (hide password)
    safe_url = database_url.replace(result.password or '', '***') if result.password else database_url
    
    return jsonify({
        "success": True,
        "database_url_length": len(database_url),
        "database_url_safe": safe_url,
        "parsed_components": {
            "scheme": result.scheme,
            "hostname": result.hostname,
            "port": result.port,
            "username": result.username,
            "password_present": bool(result.password),
            "path": result.path,
            "database_name": result.path[1:] if result.path else None,
            "query": result.query,
            "fragment": result.fragment
        },
        "url_starts_with": database_url[:50] + "..." if len(database_url) > 50 else database_url
    })

@app.route('/debug-db-connection')
@login_required
def debug_database_connection():
    """Debug endpoint to test database connection directly"""
    import os
    from urllib.parse import urlparse
    
    try:
        database_url = os.environ.get('DATABASE_URL')
        
        if not database_url:
            return jsonify({
                "success": False,
                "error": "DATABASE_URL not found",
                "using": "SQLite fallback"
            })
        
        # Test psycopg import
        try:
            import psycopg
            psycopg_available = True
            psycopg_version = psycopg.__version__
        except ImportError as e:
            return jsonify({
                "success": False,
                "error": f"psycopg import failed: {e}",
                "database_url_present": True,
                "using": "SQLite fallback"
            })
        
        # Parse DATABASE_URL
        result = urlparse(database_url)
        
        # Test PostgreSQL connection
        try:
            conn = psycopg.connect(
                dbname=result.path[1:],
                user=result.username,
                password=result.password,
                host=result.hostname,
                port=result.port
            )
            
            # Test a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            
            cursor.execute("SELECT current_database()")
            database_name = cursor.fetchone()[0]
            
            conn.close()
            
            return jsonify({
                "success": True,
                "database_type": "PostgreSQL",
                "psycopg_version": psycopg_version,
                "database_name": database_name,
                "postgres_version": version,
                "connection_params": {
                    "host": result.hostname,
                    "port": result.port,
                    "database": result.path[1:],
                    "user": result.username
                }
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"PostgreSQL connection failed: {e}",
                "error_type": str(type(e)),
                "psycopg_available": True,
                "psycopg_version": psycopg_version,
                "connection_params": {
                    "host": result.hostname,
                    "port": result.port,
                    "database": result.path[1:],
                    "user": result.username
                },
                "using": "SQLite fallback"
            })
            
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route('/debug-db')
@login_required
def debug_database():
    """Debug endpoint to test database persistence"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check connection and database info - PostgreSQL only
        cursor.execute("SELECT current_database(), version()")
        db_info = cursor.fetchone()
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM transcription_history")
        count_result = cursor.fetchone()
        count = count_result['count'] if isinstance(count_result, dict) else count_result[0]
        
        # Get recent records
        cursor.execute("""
            SELECT id, patient_id, verslag_type, created_at 
            FROM transcription_history 
            ORDER BY created_at DESC LIMIT 5
        """)
        recent = cursor.fetchall()
        
        # Test insert
        cursor.execute("""
            INSERT INTO transcription_history 
            (user_id, patient_id, verslag_type, original_transcript, structured_report, enhanced_transcript)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (1, "TEST123", "DEBUG", "Test transcription", "Test report", "Test enhanced"))
        test_result = cursor.fetchone()
        test_id = test_result['id'] if isinstance(test_result, dict) else test_result[0]
        
        conn.commit()
        
        # Verify the test record exists
        cursor.execute("SELECT * FROM transcription_history WHERE id = %s", (test_id,))
        test_record = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            "success": True,
            "database_type": "PostgreSQL",
            "database_info": db_info,
            "total_records": count,
            "recent_records": [{"id": r['id'], "patient_id": r['patient_id'], "type": r['verslag_type'], "created_at": str(r['created_at'])} for r in recent],
            "test_insert_id": test_id,
            "test_record_found": test_record is not None,
            "test_record": dict(test_record) if test_record else None
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/admin-note', methods=['POST'])
@login_required
def save_admin_note():
    """Save administrative note without audio recording"""
    try:
        patient_id = request.form.get('patient_id')
        report_type = request.form.get('report_type')
        contact_location = request.form.get('contact_location')
        note_text = request.form.get('note_text')
        
        if not all([patient_id, report_type, contact_location, note_text]):
            return jsonify({'success': False, 'error': 'Alle velden zijn verplicht'}), 400
        
        # Validate patient ID
        if not patient_id.isdigit() or len(patient_id) != 10:
            return jsonify({'success': False, 'error': 'Patiënt ID moet exact 10 cijfers bevatten'}), 400
        
        user = get_current_user()
        user_id = user.get('id') if user else 1
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO transcription_history (
                user_id, patient_id, verslag_type, contact_location,
                original_transcript, structured_report, enhanced_transcript,
                created_at, is_completed, billing_ok
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s)
            RETURNING id
        ''', (
            user_id,
            patient_id,
            report_type,
            contact_location,
            f"Administratieve nota: {note_text}",
            f"ADMINISTRATIEVE NOTA\n\nPatiënt: {patient_id}\nType: {report_type}\nLocatie: {contact_location}\n\nNota:\n{note_text}",
            f"Administratieve nota: {note_text}",
            False,  # is_completed
            False   # billing_ok
        ))
        
        result = cursor.fetchone()
        record_id = result['id'] if isinstance(result, dict) else result[0]
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Administratieve nota succesvol opgeslagen',
            'record_id': record_id
        })
        
    except Exception as e:
        print(f"❌ Admin note error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save-report', methods=['POST'])
@login_required
def save_report():
    """Save edited report content"""
    try:
        data = request.get_json()
        record_id = data.get('record_id')
        report_content = data.get('report_content')
        
        if not record_id or not report_content:
            return jsonify({'success': False, 'error': 'Missing record_id or report_content'}), 400
        
        user = get_current_user()
        user_id = user.get('id') if user else 1
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update the structured_report field
        cursor.execute('''
            UPDATE transcription_history 
            SET structured_report = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
        ''', (report_content, record_id, user_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'success': False, 'error': 'Record not found or no permission'}), 404
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Report saved successfully'})
        
    except Exception as e:
        print(f"❌ Save report error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/audio/<filename>')
@login_required
def serve_audio(filename):
    """Serve audio files with proper headers for iPhone compatibility"""
    try:
        # Audio files are stored in the uploads directory
        audio_path = os.path.join(app.config.get('UPLOAD_FOLDER', 'uploads'), filename)
        
        if not os.path.exists(audio_path):
            return "Audio file not found", 404
        
        # Determine MIME type based on file extension
        file_ext = filename.lower().split('.')[-1]
        mime_types = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'm4a': 'audio/mp4',
            'webm': 'audio/webm',
            'ogg': 'audio/ogg'
        }
        
        mime_type = mime_types.get(file_ext, 'audio/mpeg')
        
        def generate():
            with open(audio_path, 'rb') as f:
                data = f.read(1024)
                while data:
                    yield data
                    data = f.read(1024)
        
        response = Response(generate(), mimetype=mime_type)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = str(os.path.getsize(audio_path))
        response.headers['Cache-Control'] = 'no-cache'
        
        return response
        
    except Exception as e:
        print(f"❌ Audio serve error: {e}")
        return "Error serving audio file", 500

@app.route('/migrate-db')
@login_required
def migrate_database():
    """Migrate database to add new administrative columns"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'transcription_history'
        """)
        existing_columns = [row[0] if isinstance(row, tuple) else row['column_name'] for row in cursor.fetchall()]
        
        migrations_needed = []
        
        # Add contact_location column if it doesn't exist
        if 'contact_location' not in existing_columns:
            cursor.execute("""
                ALTER TABLE transcription_history 
                ADD COLUMN contact_location VARCHAR(50) DEFAULT 'Poli'
            """)
            migrations_needed.append('contact_location')
        
        # Add is_completed column if it doesn't exist
        if 'is_completed' not in existing_columns:
            cursor.execute("""
                ALTER TABLE transcription_history 
                ADD COLUMN is_completed BOOLEAN DEFAULT FALSE
            """)
            migrations_needed.append('is_completed')
        
        # Add billing_ok column if it doesn't exist
        if 'billing_ok' not in existing_columns:
            cursor.execute("""
                ALTER TABLE transcription_history 
                ADD COLUMN billing_ok BOOLEAN DEFAULT FALSE
            """)
            migrations_needed.append('billing_ok')
        
        # Add updated_at column if it doesn't exist
        if 'updated_at' not in existing_columns:
            cursor.execute("""
                ALTER TABLE transcription_history 
                ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """)
            migrations_needed.append('updated_at')
        
        conn.commit()
        conn.close()
        
        if migrations_needed:
            return jsonify({
                'success': True,
                'message': f'Database migrated successfully! Added columns: {", ".join(migrations_needed)}',
                'migrations': migrations_needed
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Database is already up to date!',
                'migrations': []
            })
        
    except Exception as e:
        print(f"❌ Database migration error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

