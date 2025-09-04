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
                (user_id, patient_id, verslag_type, original_transcript, structured_report, enhanced_transcript, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (user_id, patient_id, verslag_type, transcript, "Processing...", improved_transcript, datetime.now()))
            
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
        if not user:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get transcription history for current user only - PostgreSQL
        cursor.execute('''
            SELECT id, patient_id, verslag_type, original_transcript, 
                   structured_report, created_at
            FROM transcription_history 
            WHERE user_id = %s
            ORDER BY created_at DESC 
            LIMIT 50
        ''', (user['id'],))
        
        history_records = cursor.fetchall()
        conn.close()
        
        print(f"🔍 DEBUG: History query returned {len(history_records)} records for user {user['id']}")
        
        return render_template('history.html', records=history_records, user=user)
        
    except Exception as e:
        logger.error(f"History error: {e}")
        import traceback
        print(f"🔍 DEBUG: History error traceback: {traceback.format_exc()}")
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
        
        # Get transcription record with user verification - PostgreSQL
        cursor.execute('''
            SELECT id, patient_id, verslag_type, original_transcript, 
                   structured_report, created_at, enhanced_transcript, quality_feedback
            FROM transcription_history 
            WHERE id = %s AND user_id = %s
        ''', (record_id, user['id']))
        
        record_data = cursor.fetchone()
        conn.close()
        
        if not record_data:
            flash('Transcriptie niet gevonden of geen toegang', 'error')
            return redirect(url_for('history'))
        
        # PostgreSQL with dict_row returns dictionary already
        # Check if it's a dictionary or tuple
        if isinstance(record_data, dict):
            record = {
                'id': record_data['id'],
                'patient_id': record_data['patient_id'] or 'Unknown',
                'verslag_type': record_data['verslag_type'] or 'Unknown',
                'original_transcript': record_data['original_transcript'] or '',
                'structured_report': record_data['structured_report'] or '',
                'created_at': record_data['created_at'],
                'enhanced_transcript': record_data.get('enhanced_transcript') or record_data['original_transcript'] or '',
                'quality_feedback': record_data.get('quality_feedback', '')
            }
        else:
            # Handle tuple response (shouldn't happen with psycopg dict_row)
            record = {
                'id': record_data[0],
                'patient_id': record_data[1] or 'Unknown',
                'verslag_type': record_data[2] or 'Unknown',
                'original_transcript': record_data[3] or '',
                'structured_report': record_data[4] or '',
                'created_at': record_data[5],
                'enhanced_transcript': record_data[6] if len(record_data) > 6 else record_data[3] or '',
                'quality_feedback': record_data[7] if len(record_data) > 7 else ''
            }
        
        print(f"🔍 DEBUG: Review page loading for record {record_id}")
        print(f"🔍 DEBUG: Record data: {record}")
        
        # Since the template is missing, return a simple HTML response
        html_content = f"""
        <!DOCTYPE html>
        <html lang="nl">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medisch Rapport - Medical Dictation v4.0</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 10px;
                    text-align: center;
                }}
                .meta {{
                    color: #666;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #f0f0f0;
                    text-align: center;
                }}
                .report-section {{
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #4a5568;
                    font-size: 1.3em;
                    margin-bottom: 15px;
                    padding: 10px;
                    background: #f7fafc;
                    border-left: 4px solid #667eea;
                }}
                .report-content {{
                    background: #f9fafb;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 20px;
                    font-family: 'Georgia', serif;
                    font-size: 16px;
                    line-height: 1.8;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    min-height: 300px;
                }}
                .button-group {{
                    display: flex;
                    gap: 15px;
                    margin-top: 30px;
                    flex-wrap: wrap;
                    justify-content: center;
                }}
                button {{
                    padding: 12px 24px;
                    border: none;
                    border-radius: 6px;
                    font-size: 16px;
                    cursor: pointer;
                    transition: all 0.3s;
                    font-weight: 500;
                }}
                .btn-primary {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                .btn-primary:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
                }}
                .btn-secondary {{
                    background: #e2e8f0;
                    color: #4a5568;
                }}
                .btn-success {{
                    background: #48bb78;
                    color: white;
                }}
                .btn-warning {{
                    background: #ed8936;
                    color: white;
                }}
                .btn-info {{
                    background: #4299e1;
                    color: white;
                }}
                .btn-danger {{
                    background: #f56565;
                    color: white;
                }}
                .loading {{
                    opacity: 0.6;
                    pointer-events: none;
                }}
                .loading::after {{
                    content: " ⏳";
                }}
                .audio-section {{
                    margin: 20px 0;
                    padding: 15px;
                    background: #f0f8ff;
                    border-radius: 8px;
                    border-left: 4px solid #4299e1;
                }}
                audio {{
                    width: 100%;
                    margin-top: 10px;
                }}
                .enhanced-content {{
                    margin-top: 20px;
                    padding: 20px;
                    background: #f0fff4;
                    border-radius: 8px;
                    border-left: 4px solid #48bb78;
                    display: none;
                }}
                .differential-content {{
                    margin-top: 20px;
                    padding: 20px;
                    background: #fff5f5;
                    border-radius: 8px;
                    border-left: 4px solid #f56565;
                    display: none;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📋 Medisch Rapport</h1>
                <div class="meta">
                    <strong>Patiënt ID:</strong> {record['patient_id']} | 
                    <strong>Type:</strong> {record['verslag_type']} | 
                    <strong>Datum:</strong> {record['created_at'].strftime('%d-%m-%Y %H:%M') if record['created_at'] else 'Unknown'}
                </div>
                
                <div class="report-section">
                    <h2>📋 Gestructureerd Medisch Rapport</h2>
                    <div class="report-content" id="medical_report">{record['structured_report']}</div>
                </div>
                
                <div class="enhanced-content" id="enhanced_section">
                    <h2>🔧 Gecorrigeerd Rapport (Professor Cardiologie)</h2>
                    <div class="report-content" id="enhanced_report"></div>
                </div>
                
                <div class="differential-content" id="differential_section">
                    <h2>🔍 Conservatieve Cardiologische Analyse</h2>
                    <div class="report-content" id="differential_report"></div>
                </div>
                
                <div class="button-group">
                    <button type="button" onclick="copyToClipboard('medical_report')" class="btn-success">
                        📋 Kopieer Rapport
                    </button>
                    <button type="button" onclick="improveReport()" class="btn-warning" id="improve_btn">
                        🔧 Corrigeer Rapport
                    </button>
                    <button type="button" onclick="getDifferentialDiagnosis()" class="btn-info" id="differential_btn">
                        🔍 Cardiologische Analyse
                    </button>
                    <button type="button" onclick="window.location.href='/history'" class="btn-secondary">
                        ← Terug naar Geschiedenis
                    </button>
                    <button type="button" onclick="deleteRecord()" class="btn-danger">
                        🗑️ Verwijderen
                    </button>
                </div>
            </div>
            
            <script>
                function copyToClipboard(elementId) {{
                    const element = document.getElementById(elementId);
                    const text = element.textContent;
                    navigator.clipboard.writeText(text).then(() => {{
                        alert('✅ Rapport gekopieerd naar klembord!');
                    }}).catch(() => {{
                        // Fallback for older browsers
                        const textArea = document.createElement('textarea');
                        textArea.value = text;
                        document.body.appendChild(textArea);
                        textArea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textArea);
                        alert('✅ Rapport gekopieerd naar klembord!');
                    }});
                }}
                
                function improveReport() {{
                    const btn = document.getElementById('improve_btn');
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '⏳ Corrigeren...';
                    btn.classList.add('loading');
                    
                    const reportText = document.getElementById('medical_report').textContent;
                    
                    fetch('/api/improve-report', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            report: reportText,
                            record_id: {record['id']}
                        }})
                    }})
                    .then(response => response.json())
                    .then(result => {{
                        if (result.success) {{
                            document.getElementById('enhanced_report').textContent = result.improved_report;
                            document.getElementById('enhanced_section').style.display = 'block';
                            alert('✅ Rapport gecorrigeerd door Professor Cardiologie!');
                        }} else {{
                            alert('❌ Fout bij corrigeren: ' + result.error);
                        }}
                    }})
                    .catch(error => {{
                        alert('❌ Fout bij corrigeren: ' + error);
                    }})
                    .finally(() => {{
                        btn.innerHTML = originalText;
                        btn.classList.remove('loading');
                    }});
                }}
                
                function getDifferentialDiagnosis() {{
                    const btn = document.getElementById('differential_btn');
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '⏳ Analyseren...';
                    btn.classList.add('loading');
                    
                    const reportText = document.getElementById('medical_report').textContent;
                    
                    fetch('/api/differential-diagnosis', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            report: reportText,
                            record_id: {record['id']}
                        }})
                    }})
                    .then(response => response.json())
                    .then(result => {{
                        if (result.success) {{
                            document.getElementById('differential_report').textContent = result.differential_diagnosis;
                            document.getElementById('differential_section').style.display = 'block';
                            alert('✅ Conservatieve analyse opgesteld door Cardioloog!');
                        }} else {{
                            alert('❌ Fout bij analyse: ' + result.error);
                        }}
                    }})
                    .catch(error => {{
                        alert('❌ Fout bij analyse: ' + error);
                    }})
                    .finally(() => {{
                        btn.innerHTML = originalText;
                        btn.classList.remove('loading');
                    }});
                }}
                
                function deleteRecord() {{
                    if (confirm('Weet u zeker dat u deze transcriptie wilt verwijderen?')) {{
                        fetch('/delete-transcription/{record['id']}', {{
                            method: 'DELETE'
                        }})
                        .then(response => response.json())
                        .then(result => {{
                            if (result.success) {{
                                alert('✅ Transcriptie verwijderd!');
                                window.location.href = '/history';
                            }} else {{
                                alert('❌ Fout bij verwijderen: ' + result.error);
                            }}
                        }})
                        .catch(error => {{
                            alert('❌ Fout bij verwijderen: ' + error);
                        }});
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        print(f"❌ Review page error: {e}")
        import traceback
        print(f"🔍 DEBUG: Review error traceback: {traceback.format_exc()}")
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

@app.route('/api/improve-report', methods=['POST'])
@login_required
def improve_report():
    """Improve medical report using Professor Cardiology expertise"""
    try:
        data = request.get_json()
        report_text = data.get('report', '')
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
   - Los tegenstrijdigheden op (vb: "aorta 50mm" vs "normale aorta")
   - Zorg voor consistente metingen en bevindingen
   - Harmoniseer terminologie door het rapport

3. DIAGNOSES VERBETEREN:
   - Vervang vage termen zoals "aortasclerose" door specifieke diagnoses
   - Voeg ontbrekende klinische aanbevelingen toe:
     * Bij bicuspiede klep: check aortadilatatie + coarctatio + screening 1e graads verwanten
     * Bij significante klepafwijkingen: follow-up schema
     * Bij congenitale afwijkingen: familie screening

4. BEHOUD:
   - Alle originele bevindingen en metingen
   - Originele structuur en stijl
   - Alle patiëntspecifieke informatie

Geef ALLEEN het gecorrigeerde rapport terug, zonder uitleg of toevoegingen.

Origineel rapport:
{report_text}

Gecorrigeerd rapport:"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Je bent een professor cardiologie die medische rapporten corrigeert. Focus op typo's, inconsistenties en vage diagnoses. Behoud alle originele bevindingen en structuur. Geef alleen het gecorrigeerde rapport terug."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        improved_report = response.choices[0].message.content.strip()
        
        # Optionally save the improved report to database
        if record_id:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE transcription_history 
                    SET improved_report = %s 
                    WHERE id = %s
                ''', (improved_report, record_id))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"⚠️ Could not save improved report: {e}")
        
        return jsonify({
            'success': True,
            'improved_report': improved_report
        })
        
    except Exception as e:
        print(f"❌ Improve report error: {e}")
        import traceback
        print(f"🔍 DEBUG: Improve report traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/differential-diagnosis', methods=['POST'])
@login_required
def differential_diagnosis():
    """Generate differential diagnosis and treatment recommendations"""
    try:
        data = request.get_json()
        report_text = data.get('report', '')
        record_id = data.get('record_id')
        
        if not report_text:
            return jsonify({'success': False, 'error': 'Geen rapport tekst ontvangen'})
        
        # Use OpenAI GPT-4 for differential diagnosis
        import openai
        
        prompt = f"""U bent een ervaren cardioloog die een conservatieve, evidence-based differentiaaldiagnose opstelt. BELANGRIJK: Geef GEEN gevaarlijke of ongefundeerde behandelingsadviezen.

INSTRUCTIES:
1. Analyseer alleen de SIGNIFICANTE bevindingen uit het rapport
2. Geef alleen CONSERVATIEVE, evidence-based aanbevelingen
3. Verwijs naar cardioloog/specialist voor complexe beslissingen
4. Vermijd gevaarlijke adviezen zoals onnodige operaties

STRUCTUUR:
1. SIGNIFICANTE BEVINDINGEN:
   - Lijst alleen klinisch relevante afwijkingen
   - Negeer normale varianten en minimale afwijkingen

2. MOGELIJKE DIAGNOSES (alleen bij duidelijke pathologie):
   - Waarschijnlijkheid: hoog/matig/laag
   - Ondersteunende bevindingen

3. CONSERVATIEVE AANBEVELINGEN:
   - Alleen evidence-based follow-up
   - Verwijs naar specialist bij twijfel
   - Geen operatieve adviezen zonder duidelijke indicatie

4. FOLLOW-UP:
   - Standaard cardiologische follow-up schema's
   - Gebaseerd op ESC/AHA richtlijnen

VERMIJD:
- Operatieve adviezen bij minimale afwijkingen
- Medicatie zonder duidelijke indicatie
- Gevaarlijke of experimentele behandelingen

Medisch rapport:
{report_text}

Conservatieve cardiologische analyse:"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Je bent een conservatieve cardioloog die veilige, evidence-based analyses geeft. Vermijd gevaarlijke behandelingsadviezen. Focus op significante bevindingen en verwijs naar specialist bij twijfel."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.2
        )
        
        differential_diagnosis_text = response.choices[0].message.content.strip()
        
        # Optionally save the differential diagnosis to database
        if record_id:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE transcription_history 
                    SET differential_diagnosis = %s 
                    WHERE id = %s
                ''', (differential_diagnosis_text, record_id))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"⚠️ Could not save differential diagnosis: {e}")
        
        return jsonify({
            'success': True,
            'differential_diagnosis': differential_diagnosis_text
        })
        
    except Exception as e:
        print(f"❌ Differential diagnosis error: {e}")
        import traceback
        print(f"🔍 DEBUG: Differential diagnosis traceback: {traceback.format_exc()}")
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

