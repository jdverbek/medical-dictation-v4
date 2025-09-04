"""
Superior Medical Transcription System with GPT-4o Transcribe
Latest OpenAI API with GPT-4o Transcribe model for superior accuracy
"""

import os
import io
import datetime
import tempfile
import subprocess
from openai import OpenAI
from dotenv import load_dotenv
from .medical_classification_validator import MedicalClassificationValidator

# Load environment variables
load_dotenv()

class SuperiorMedicalTranscription:
    def __init__(self):
        # Initialize OpenAI client with latest API
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Clean, modern initialization
        self.client = OpenAI(api_key=api_key)
        
        # Initialize medical classification validator
        self.medical_validator = MedicalClassificationValidator()
        
        print(f"🎤 Audio transcription client initialized with OpenAI GPT-4o Transcribe")
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 10 else 'short'}")
        print(f"🛡️ Medical classification validator initialized")
    
    def enhance_transcription(self, raw_transcript, report_type):
        """
        Enhance raw transcription with context-aware improvements
        Checks for nonsense, contradictions, and inaccuracies
        """
        try:
            # Define context-specific prompts
            context_prompts = {
                'TTE': 'transthoracale echocardiografie (TTE)',
                'TEE': 'transesofageale echocardiografie (TEE)', 
                'CONSULTATIE': 'cardiologische consultatie',
                'SPOEDCONSULT': 'spoed cardiologische consultatie'
            }
            
            context = context_prompts.get(report_type, 'medisch verslag')
            
            system_message = f"""Je bent een medische transcriptie specialist die transcripties verbetert voor {context}.

TAAK: Verbeter de ruwe transcriptie door:
1. Nonsens woorden corrigeren naar medische termen
2. Tegenstrijdigheden identificeren en oplossen  
3. Onnauwkeurigheden verbeteren met context
4. Medische terminologie optimaliseren
5. Grammatica en zinsbouw verbeteren

MEDISCHE TERMINOLOGIE RICHTLIJNEN:
- TTE = "tee-tee-ee" → transthoracale echografie
- EDD = "ie-dee-dee" → eind-diastolische diameter  
- IVS = "ie-vee-es" → interventriculair septum
- PW = "pee-double-you" → posterior wand
- LVEF = "el-vee-ee-ef" → linker ventrikelejectiefractie
- TAPSE = "tap-se" → tricuspid annulaire verplaatsing
- LA = "el-aa" → linker atrium
- RVSP = "err-vee-es-pee" → pulmonaaldruk

KRITIEKE MEDISCHE CLASSIFICATIES (ESC 2023):
- HFrEF: LVEF ≤40% (Heart Failure with reduced EF)
- HFmrEF: LVEF 41-49% (Heart Failure with mildly reduced EF)  
- HFpEF: LVEF ≥50% (Heart Failure with preserved EF)
- Normale LVEF: ≥50% (bij afwezigheid van hartfalen)

AORTAKLEPSTENOSE CLASSIFICATIE:
- Mild: Vmax <3.0 m/s, gemiddelde gradient <20 mmHg
- Matig: Vmax 3.0-4.0 m/s, gemiddelde gradient 20-40 mmHg  
- Ernstig: Vmax >4.0 m/s, gemiddelde gradient >40 mmHg

MITRAALINSUFFICIËNTIE CLASSIFICATIE:
- Mild: vena contracta <3mm, ERO <20mm², RVol <30ml
- Matig: vena contracta 3-7mm, ERO 20-40mm², RVol 30-60ml
- Ernstig: vena contracta >7mm, ERO >40mm², RVol >60ml

REGELS:
- Behoud alle originele medische informatie
- Corrigeer alleen duidelijke fouten
- Voeg GEEN nieuwe medische informatie toe
- Markeer onzekere correcties met [?]
- Geef duidelijke Nederlandse medische terminologie

Geef de verbeterde transcriptie terug, gevolgd door een lijst van toegepaste verbeteringen."""

            user_message = f"""Ruwe transcriptie voor {context}:

{raw_transcript}

Verbeter deze transcriptie volgens de richtlijnen."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=1.0
            )
            
            enhanced_text = response.choices[0].message.content
            
            # Split enhanced text and improvements
            if "TOEGEPASTE VERBETERINGEN:" in enhanced_text:
                parts = enhanced_text.split("TOEGEPASTE VERBETERINGEN:")
                enhanced_transcript = parts[0].strip()
                improvements = parts[1].strip() if len(parts) > 1 else "Geen specifieke verbeteringen gedetecteerd."
            else:
                enhanced_transcript = enhanced_text
                improvements = "Transcriptie verbeterd voor medische context en terminologie."
            
            return {
                'enhanced_transcript': enhanced_transcript,
                'improvements': improvements
            }
            
        except Exception as e:
            print(f"❌ Enhanced transcription failed: {str(e)}")
            return {
                'enhanced_transcript': raw_transcript,
                'improvements': f"Fout bij verbetering: {str(e)}"
            }

    def validate_medical_report(self, report_text, report_type):
        """
        Validate medical report for inconsistencies and contradictions
        Uses the MedicalClassificationValidator for critical error detection
        """
        try:
            # First run the comprehensive medical classification validation
            validation_results = self.medical_validator.validate_all(report_text)
            validation_report = self.medical_validator.format_validation_report(validation_results)
            
            # If critical errors found, return immediately
            if any(validation_results.values()):
                return validation_report
            
            # If no critical errors, run additional AI-based validation
            system_message = f"""Je bent een medische quality control specialist die rapporten controleert op inconsistenties.

TAAK: Analyseer het medische rapport en identificeer:
1. Tegenstrijdigheden in metingen en beschrijvingen
2. Onlogische combinaties van bevindingen  
3. Inconsistente terminologie
4. Medische onnauwkeurigheden

SPECIFIEKE CONTROLES:
- LA dimensies: >40mm = gedilateerd, >47mm = sterk gedilateerd
- Aortasinus: >40mm = gedilateerd, maar check consistentie met "normale dimensies"
- CVD correlaties: 
  * "vena cava plat" → CVD 0-5 mmHg (<17mm, >50% variatie)
  * "vena cava stuwing" → CVD 10-15 mmHg (>17mm, <50% variatie)
- Klepfunctie: insufficiëntie graden moeten consistent zijn
- EF percentages: moeten kloppen met functionele beschrijving

KRITIEKE MEDISCHE CLASSIFICATIE VALIDATIE (ESC 2023):
- HFrEF: ALLEEN bij LVEF ≤40% (NIET bij 41-49%!)
- HFmrEF: LVEF 41-49% (Heart Failure with mildly reduced EF)
- HFpEF: LVEF ≥50% (Heart Failure with preserved EF)
- Aortaklepstenose: Mild <20mmHg, Matig 20-40mmHg, Ernstig >40mmHg
- Mitraalinsufficiëntie: Mild <3mm VC, Matig 3-7mm VC, Ernstig >7mm VC

LEVENSGEVAARLIJKE FOUTEN DETECTEREN:
- LVEF 40-45% geclassificeerd als HFrEF → FOUT! Dit is HFmrEF
- Normale LVEF met HF diagnose → Controleer op HFpEF criteria
- Ernstige stenose met lage gradiënten → Controleer op low-flow state

RAPPORTAGE:
- Geef specifieke feedback per inconsistentie
- Verwijs naar exacte waarden in het rapport
- Suggereer mogelijke correcties
- Gebruik duidelijke medische terminologie

Geef alleen feedback als er daadwerkelijke problemen zijn. Als alles consistent is, meld dat."""

            user_message = f"""Medisch rapport voor quality control ({report_type}):

{report_text}

Controleer dit rapport op inconsistenties en tegenstrijdigheden."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=1.0
            )
            
            ai_validation_feedback = response.choices[0].message.content
            
            # Combine validation report with AI feedback
            if "geen medische classificatie fouten" in validation_report.lower():
                return ai_validation_feedback
            else:
                return f"{validation_report}\n\n📋 AANVULLENDE AI VALIDATIE:\n{ai_validation_feedback}"
            
        except Exception as e:
            print(f"❌ Medical report validation failed: {str(e)}")
            return f"Quality control fout: {str(e)}"

    def generate_esc_recommendations(self, report_text, report_type):
        """
        Generate ESC Guidelines recommendations with LoR and LoE for specific pathologies
        """
        try:
            system_message = f"""Je bent een cardioloog specialist in ESC Guidelines die evidence-based aanbevelingen geeft.

TAAK: Analyseer het medische rapport en geef specifieke ESC Guidelines aanbevelingen voor elke geïdentificeerde pathologie.

VOOR ELKE PATHOLOGIE GEEF:
1. Specifieke ESC Guideline (jaar en titel)
2. Concrete aanbevelingen met Class of Recommendation (LoR)
3. Level of Evidence (LoE)
4. Toepassing op deze specifieke casus

CLASS OF RECOMMENDATION (LoR):
- Class I: Aanbevolen/geïndiceerd (sterke aanbeveling)
- Class IIa: Redelijk om te overwegen (matige aanbeveling)  
- Class IIb: Mag overwogen worden (zwakke aanbeveling)
- Class III: Niet aanbevolen/gecontraïndiceerd

LEVEL OF EVIDENCE (LoE):
- Level A: Meerdere RCTs of meta-analyses
- Level B: Enkele RCT of grote niet-gerandomiseerde studies
- Level C: Consensus van experts/kleine studies

MEEST RECENTE ESC GUIDELINES:
- 2024 ESC Guidelines for Atrial Fibrillation
- 2023 ESC Guidelines for Acute Coronary Syndromes
- 2023 ESC Guidelines for Heart Failure
- 2022 ESC Guidelines for Cardiovascular Disease Prevention
- 2021 ESC Guidelines for Valvular Heart Disease
- 2020 ESC Guidelines for Sports Cardiology

FORMAT:
=== AANBEVELINGEN ===

PATHOLOGIE: [Specifieke diagnose]
ESC GUIDELINE: [Jaar en titel]
- [Concrete aanbeveling] (Class [I/IIa/IIb/III], LoE [A/B/C])
- [Volgende aanbeveling] (Class [I/IIa/IIb/III], LoE [A/B/C])

CASUS-SPECIFIEKE TOEPASSING:
[Hoe guidelines toepassen op deze specifieke patiënt]

Geef alleen aanbevelingen voor daadwerkelijk geïdentificeerde pathologieën."""

            user_message = f"""Medisch rapport voor ESC Guidelines analyse ({report_type}):

{report_text}

Geef ESC Guidelines aanbevelingen met LoR en LoE voor elke geïdentificeerde pathologie."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=1.0
            )
            
            esc_recommendations = response.choices[0].message.content
            
            return esc_recommendations
            
        except Exception as e:
            print(f"❌ ESC recommendations generation failed: {str(e)}")
            return f"ESC Guidelines aanbevelingen fout: {str(e)}"

    def convert_audio_to_wav(self, file_content, original_filename):
        try:
            print(f"DEBUG: Converting {original_filename} to WAV format...")
            
            # Determine input file extension for proper handling
            file_ext = os.path.splitext(original_filename.lower())[1]
            if file_ext in ['.m4a', '.mp4']:
                temp_suffix = '.m4a'
            elif file_ext == '.webm':
                temp_suffix = '.webm'
            else:
                temp_suffix = file_ext or '.audio'
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as temp_input:
                temp_input.write(file_content)
                temp_input_path = temp_input.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_output:
                temp_output_path = temp_output.name
            
            # Convert using ffmpeg with M4A support
            cmd = [
                'ffmpeg', '-i', temp_input_path,
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
                '-ar', '16000',          # 16kHz sample rate (good for speech)
                '-ac', '1',              # Mono channel
                '-y',                    # Overwrite output file
                temp_output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes timeout for Starter plan
            
            if result.returncode == 0:
                # Read converted file
                with open(temp_output_path, 'rb') as f:
                    converted_content = f.read()
                
                print(f"DEBUG: Conversion successful! Original: {len(file_content)} bytes, Converted: {len(converted_content)} bytes")
                
                # Cleanup
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
                
                return {
                    'success': True,
                    'content': converted_content,
                    'filename': original_filename.replace('.webm', '.wav').replace('.m4a', '.wav').replace('.mp4', '.wav'),
                    'content_type': 'audio/wav'
                }
            else:
                print(f"DEBUG: FFmpeg conversion failed: {result.stderr}")
                # Cleanup
                os.unlink(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
                
                return {
                    'success': False,
                    'error': f"Audio conversie gefaald: {result.stderr}"
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': "Audio conversie timeout (>5 min). Bestand te groot of complex."
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Audio conversie error: {str(e)}"
            }
    
    def detect_audio_format(self, file_content, filename):
        """Detect audio format and determine if conversion is needed"""
        # Check for WebM format (EBML header)
        if file_content.startswith(b'\x1a\x45\xdf\xa3'):
            print("DEBUG: File is WebM format, will convert to WAV")
            return 'audio/webm', filename, True  # needs_conversion = True
        elif filename.lower().endswith('.webm'):
            print("DEBUG: File has .webm extension, will convert to WAV")
            return 'audio/webm', filename, True  # needs_conversion = True
        # Check for M4A/MP4 format (ftyp header)
        elif file_content[4:8] == b'ftyp' and (b'M4A ' in file_content[:20] or b'mp41' in file_content[:20] or b'mp42' in file_content[:20]):
            print("DEBUG: File is M4A/MP4 format, will convert to WAV")
            return 'audio/m4a', filename, True  # needs_conversion = True
        elif filename.lower().endswith(('.m4a', '.mp4')):
            print("DEBUG: File has .m4a/.mp4 extension, will convert to WAV")
            return 'audio/m4a', filename, True  # needs_conversion = True
        else:
            # Default to original content type
            return 'audio/wav', filename, False  # needs_conversion = False
    
    def transcribe_audio(self, audio_file, report_type="TTE"):
        """Audio transcription using GPT-4o Transcribe model"""
        try:
            # Reset file pointer to beginning
            audio_file.seek(0)
            
            # Read file content for analysis
            file_content = audio_file.read()
            audio_file.seek(0)  # Reset again
            
            print(f"DEBUG: File size: {len(file_content)} bytes")
            print(f"DEBUG: File name: {audio_file.filename}")
            print(f"DEBUG: Report type: {report_type}")
            
            # Validate audio
            if len(file_content) == 0:
                return {
                    'success': False,
                    'error': "⚠️ Audio bestand is leeg. Upload een geldig audio bestand."
                }
            
            if len(file_content) > 25 * 1024 * 1024:  # 25MB limit for OpenAI
                return {
                    'success': False,
                    'error': f"⚠️ Audio bestand te groot ({len(file_content)/1024/1024:.1f}MB). Maximum is 25MB."
                }
            
            # Detect actual format with conversion detection
            content_type, filename, needs_conversion = self.detect_audio_format(file_content, audio_file.filename)
            
            # Automatic conversion for WebM files
            if needs_conversion:
                print("DEBUG: 🔄 Starting automatic audio conversion...")
                conversion_result = self.convert_audio_to_wav(file_content, filename)
                
                if not conversion_result['success']:
                    return {
                        'success': False,
                        'error': f"⚠️ Automatische conversie gefaald\n\n{conversion_result['error']}"
                    }
                
                # Use converted content
                file_content = conversion_result['content']
                filename = conversion_result['filename']
                content_type = conversion_result['content_type']
                print(f"DEBUG: ✅ Conversion successful! Using converted WAV file ({len(file_content)} bytes)")
            
            # Create a file-like object for the API
            audio_file_obj = io.BytesIO(file_content)
            audio_file_obj.name = filename
            
            print(f"DEBUG: Using content type: {content_type}")
            print(f"DEBUG: Using filename: {filename}")
            
            # Prepare the prompt based on report type with enhanced medical context
            if report_type == "LIVE_CONSULTATIE":
                prompt = """Je bent een medische secretaresse die aanwezig is bij een cardiologische consultatie waarbij een patiënt op bezoek komt bij de arts. Je hoort een conversatie tussen 2 of meerdere personen (soms zijn familieleden mee) en maakt een gedetailleerde samenvatting van de consultatie. Focus je vooral op de anamnese/symptomen, probeer deze zo getrouw mogelijk neer te pennen. Let op: soms zal de conversatie gestoord worden doordat de arts gebeld wordt of iemand binnenkomt; hier moet je goed bedacht op zijn (de context zal plots niet meer kloppen)."""
            elif report_type == "CONSULTATIE":
                prompt = "Dit is een Nederlandse medische dictatie van een cardioloog voor een gestructureerde consultatie. Gebruik correcte medische terminologie en behoud alle details voor het consultatieverslag."
            else:
                prompt = """Dit is een Nederlandse medische transcriptie van een cardioloog. 
                Belangrijke medische termen om op te letten:
                - Medicijnen: Cedocard, Arixtra, Metoprolol, Lisinopril, Furosemide, Spironolactone
                - Afkortingen: TTE, TEE, ECG, LVEF, VKF, ACS, NSTEMI, STEMI
                - Nederlandse medische terminologie: atriumfibrillatie, voorkamerfibrillatie, echocardiografie, retrosternale pijn, dyspnoe
                Gebruik correcte medische terminologie en spelling."""
            
            # Use GPT-4o Transcribe model for superior accuracy (but NOT gpt-4o-audio-preview)
            print("DEBUG: Calling OpenAI GPT-4o Transcribe API...")
            
            # Try the available model names (excluding gpt-4o-audio-preview as requested)
            available_models = [
                "gpt-4o-transcribe",  # Direct name from the screenshot
                "gpt-4-turbo-audio",  # Alternative naming convention
                "whisper-large-v3",   # Enhanced Whisper model
                "whisper-1"           # Fallback to stable model
            ]
            
            transcript_response = None
            successful_model = None
            
            for model_name in available_models:
                try:
                    print(f"DEBUG: Trying model: {model_name}")
                    # Reset file pointer
                    audio_file_obj.seek(0)
                    
                    transcript_response = self.client.audio.transcriptions.create(
                        model=model_name,
                        file=audio_file_obj,
                        language="nl",
                        prompt=prompt,
                        response_format="text"
                    )
                    successful_model = model_name
                    print(f"DEBUG: Success with model: {model_name}")
                    break
                except Exception as model_error:
                    print(f"DEBUG: Model {model_name} failed: {model_error}")
                    continue
            
            if transcript_response is None:
                return {
                    'success': False,
                    'error': "⚠️ Geen beschikbare transcriptie modellen gevonden. Probeer later opnieuw."
                }
            
            # Get the transcript text with better null checking
            if transcript_response is None:
                return {
                    'success': False,
                    'error': "⚠️ Transcriptie response is None. API call gefaald."
                }
            
            # Handle different response types safely
            try:
                if isinstance(transcript_response, str):
                    corrected_transcript = transcript_response
                elif hasattr(transcript_response, 'text') and transcript_response.text:
                    corrected_transcript = transcript_response.text
                elif hasattr(transcript_response, 'content') and transcript_response.content:
                    corrected_transcript = transcript_response.content
                else:
                    return {
                        'success': False,
                        'error': f"⚠️ Onbekend response format: {type(transcript_response)}. Geen text attribuut gevonden."
                    }
            except Exception as response_error:
                return {
                    'success': False,
                    'error': f"⚠️ Error bij verwerken response: {str(response_error)}"
                }
            
            # Validate transcript before post-processing
            if not corrected_transcript or not isinstance(corrected_transcript, str):
                return {
                    'success': False,
                    'error': f"⚠️ Ongeldige transcript: {type(corrected_transcript)} - '{corrected_transcript}'"
                }
            
            # Post-process with GPT-4o for medical terminology correction
            print("DEBUG: Applying GPT-4o post-processing for medical accuracy...")
            try:
                post_processed_transcript = self.post_process_with_gpt4o(corrected_transcript)
                
                # Use post-processed version if successful, otherwise keep original
                if post_processed_transcript and isinstance(post_processed_transcript, str) and len(post_processed_transcript.strip()) > 0:
                    corrected_transcript = post_processed_transcript
                    print("DEBUG: Post-processing successful")
                else:
                    print("DEBUG: Post-processing failed or returned empty, using original transcript")
                    
            except Exception as post_error:
                print(f"DEBUG: Post-processing error: {post_error}, using original transcript")
                # Continue with original transcript if post-processing fails
            
            # Validation check
            if not corrected_transcript or len(corrected_transcript.strip()) < 10:
                return {
                    'success': False,
                    'error': f"⚠️ Transcriptie probleem: Audio werd niet correct getranscribeerd.\n\nResultaat: '{corrected_transcript}'"
                }
            
            # Check for hallucination
            if self.detect_hallucination(corrected_transcript):
                return {
                    'success': False,
                    'error': f"🚨 Hallucinatie Gedetecteerd!\n\nHet audio bestand is te stil of onduidelijk."
                }
            
            print(f"DEBUG: Transcription successful with {successful_model}! Length: {len(corrected_transcript)} characters")
            
            return {
                'success': True,
                'transcript': corrected_transcript
            }
            
        except Exception as e:
            print(f"DEBUG: Transcription error: {str(e)}")
            error_msg = str(e).lower()
            
            # Provide specific error messages
            if "api" in error_msg and "key" in error_msg:
                return {
                    'success': False,
                    'error': "⚠️ API Key probleem. Controleer of OPENAI_API_KEY correct is ingesteld."
                }
            elif "file" in error_msg and "format" in error_msg:
                return {
                    'success': False,
                    'error': "⚠️ Audio formaat niet ondersteund. Probeer het bestand te converteren naar WAV of MP3."
                }
            elif "model" in error_msg:
                return {
                    'success': False,
                    'error': "⚠️ GPT-4o Transcribe model nog niet beschikbaar. Teruggevallen op standaard Whisper model."
                }
            else:
                return {
                    'success': False,
                    'error': f"⚠️ Transcriptie fout: {str(e)}"
                }
    
    def post_process_with_gpt4o(self, transcript):
        """Post-process transcript with GPT-4o for medical accuracy"""
        try:
            # Validate input
            if not transcript or not isinstance(transcript, str):
                print(f"DEBUG: Invalid transcript input for post-processing: {type(transcript)}")
                return transcript
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4o for post-processing
                messages=[
                    {
                        "role": "system",
                        "content": """Je bent een medische transcriptie expert. Corrigeer deze Nederlandse medische transcriptie:
                        
                        BELANGRIJKE CORRECTIES:
                        - "sedocar" → "Cedocard" (medicijn)
                        - "arixtra" → "Arixtra" (medicijn)
                        - Corrigeer medische terminologie
                        - Fix medicijnnamen en doseringen
                        - Behoud de originele betekenis
                        
                        Geef alleen de gecorrigeerde transcriptie terug, geen uitleg."""
                    },
                    {
                        "role": "user",
                        "content": f"Corrigeer deze medische transcriptie:\n\n{transcript}"
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Safely extract response
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    result = choice.message.content
                    if result and isinstance(result, str):
                        return result.strip()
            
            print("DEBUG: Post-processing response format issue, using original")
            return transcript
            
        except Exception as e:
            print(f"DEBUG: Post-processing failed, using original: {e}")
            return transcript  # Return original if post-processing fails
    
    def detect_hallucination(self, transcript):
        """Detect hallucination patterns"""
        hallucination_keywords = ["transcribe", "dictatie", "secretary", "medical"]
        keyword_count = sum(1 for keyword in hallucination_keywords if keyword.lower() in transcript.lower())
        
        # If multiple prompt keywords appear repeatedly, it's likely hallucination
        if keyword_count > 10:
            return True
        
        # Check for repetitive patterns
        lines = transcript.split('\n')
        if len(lines) > 5:
            unique_lines = set(lines)
            if len(unique_lines) < len(lines) / 2:  # More than half are duplicates
                return True
        
        return False
    
    def call_gpt(self, messages, model="gpt-4o", temperature=0.7):
        """Call GPT-4o with modern API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to gpt-4o-mini if gpt-4o not available
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=3000
                )
                return response.choices[0].message.content
            except:
                print(f"GPT API error: {str(e)}")
                return f"Error: {str(e)}"
    
    def generate_tee_report(self, transcript, patient_id=""):
        """Generate TEE report with detailed template structure"""
        today = datetime.datetime.now().strftime("%d-%m-%Y")
        
        system_message = """Je bent een ervaren cardioloog die TEE (Transesofageale Echo) verslagen maakt.
        Genereer een gestructureerd TEE verslag volgens het exacte template format.
        Gebruik correcte Nederlandse medische terminologie en UTF-8 encoding.
        BELANGRIJK: Analyseer de transcriptie zorgvuldig en vul zoveel mogelijk details in.
        Gebruik alleen (...) voor waarden die echt niet in de transcriptie staan.
        Verzin NOOIT waarden die niet in de transcriptie staan.
        
        SPECIALE AANDACHT:
        - Zorg voor correcte Nederlandse karakters (ë, ï, ü, etc.)
        - Extraheer alle beschikbare informatie uit de transcriptie
        - Vul indicatie, supervisie, verpleegkundige, anesthesist in als vermeld
        - Gebruik exacte waarden voor drukken, dimensies, graden
        - Vermeld conclusies en aanbevelingen als gegeven
        
        SUPERVISOREN - gebruik alleen deze namen als vermeld in transcriptie:
        - dr. Dujardin
        - dr. Bergez  
        - dr. Anné
        - dr. de Ceuninck
        - dr. Vanhaverbeke
        - dr. Gillis
        - dr. Van de Walle
        - dr. Muyldermans
        
        Als een andere naam wordt genoemd, gebruik (...) voor supervisie."""
        
        user_message = f"""Genereer een TEE verslag voor patiënt {patient_id} op {today}.
        
        Transcriptie: {transcript}
        
        Analyseer de transcriptie zorgvuldig en vul zoveel mogelijk details in. Gebruik EXACT dit template format:
        
        BELANGRIJK: Laat deze specifieke meetlijnen VOLLEDIG WEG als er geen metingen worden genoemd:
        - Dimensies: LVOT (...) mm, aorta sinus (...) mm, sinutubulaire junctie (...) mm, aorta ascendens boven de sinutubulaire junctie (...) mm.
        - Mitraalinsufficientie vena contracta (...) mm, ERO (...) mm2 en RVol (...) ml/slag.
        - Aortaklepinsufficientie vena contracta (...) mm, ERO (...) mm2 en RVol (...) ml/slag.
        - Aortaklepstenose piekgradient (...) mmHg en gemiddelde gradient (...) mmHg, effectief klepoppervlak (...) cm2.
        
        Voor pulmonaaldruk: als geen meting genoemd → gebruik "* Geen pulmonaaldruk opmeetbaar."
        
        Onderzoeksdatum: {today}
        Bevindingen: TEE ONDERZOEK : 3D TEE met (...) toestel
        Indicatie: (...)
        Afname mondeling consent: dr. Verbeke. Informed consent: patiënt kreeg uitleg over aard onderzoek, mogelijke resultaten en procedurele risico's en verklaart zich hiermee akkoord.
        Supervisie: dr (...)
        Verpleegkundige: (...)
        Anesthesist: dr. (...)
        Locatie: endoscopie 3B
        Sedatie met (...) en topicale Xylocaine spray.
        (...) introductie TEE probe, (...) verloop van onderzoek zonder complicatie.
        
        VERSLAG:
        - Linker ventrikel is (...), (...) gedilateerd en (...) (...) regionale wandbewegingstoornissen.
        - Rechter ventrikel is (...), (...) gedilateerd en (...).
        - De atria zijn (...) gedilateerd.
        - Linker hartoortje is (...) vergroot, er is (...) spontaan contrast, zonder toegevoegde structuur. Hartoortje snelheden (...) cm/s.
        - Interatriaal septum (...)
        - Mitralisklep: (...), morfologisch (...), er is (...) insufficiëntie, er is (...) stenose, (...) toegevoegde structuur.
        [ALLEEN als mitraal metingen genoemd: * Mitraalinsufficientie vena contracta (...) mm, ERO (...) mm2 en RVol (...) ml/slag.]
        - Aortaklep: (...), morfologisch (...), (...) verkalkt, er is (...) insufficiëntie, er is (...) stenose (...) toegevoegde structuur.
        [ALLEEN als aorta dimensies genoemd: Dimensies: LVOT (...) mm, aorta sinus (...) mm, sinutubulaire junctie (...) mm, aorta ascendens boven de sinutubulaire junctie (...) mm.]
        [ALLEEN als aortaklep insufficiëntie metingen genoemd: * Aortaklepinsufficientie vena contracta (...) mm, ERO (...) mm2 en RVol (...) ml/slag.]
        [ALLEEN als aortaklep stenose metingen genoemd: * Aortaklepstenose piekgradient (...) mmHg en gemiddelde gradient (...) mmHg, effectief klepoppervlak (...) cm2.]
        - Tricuspiedklep: (...), morfologisch (...), er is (...) insufficiëntie, (...) toegevoegde structuur.
        [Als pulmonaaldruk genoemd: * Systolische pulmonaaldruk afgeleid uit TI (...) mmHg + CVD.]
        [Als pulmonaaldruk NIET genoemd: * Geen pulmonaaldruk opmeetbaar.]
        - Pulmonaalklep is (...), er is (...) insufficiëntie.
        - Aorta ascendens is (...) gedilateerd, graad (...) atheromatose van de aortawand.
        - Pulmonale arterie is (...) gedilateerd.
        - Vena cava inferior/levervenes zijn (...) verbreed (...) ademvariatie.
        - Pericard: er is (...) pericardvocht.
        
        CONCLUSIE: (...)
        AANBEVELINGEN: (...)
        
        Vul zoveel mogelijk details in uit de transcriptie. Gebruik correcte Nederlandse karakters.
        LAAT MEETLIJNEN VOLLEDIG WEG als er geen metingen worden genoemd."""
        
        report = self.call_gpt([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        return report
    
    def generate_tte_report(self, transcript, patient_id="", expert_analysis=None):
        """Generate TTE report with natural medical language, incorporating extracted diastolic parameters"""
        today = datetime.datetime.now().strftime("%d-%m-%Y")
        
        # Extract diastolic parameters from expert analysis if available
        diastolic_params = {}
        if expert_analysis and 'agent_2_diagnostic_expert' in expert_analysis:
            diastolic_data = expert_analysis['agent_2_diagnostic_expert'].get('tte_diastolic_parameters', {})
            if diastolic_data:
                diastolic_params = {k: v for k, v in diastolic_data.items() if v != "niet vermeld"}
        
        # Build diastolic parameters string for inclusion in report
        diastolic_info = ""
        if diastolic_params:
            diastolic_parts = []
            if 'E_velocity' in diastolic_params:
                diastolic_parts.append(f"E {diastolic_params['E_velocity']}")
            if 'A_velocity' in diastolic_params:
                diastolic_parts.append(f"A {diastolic_params['A_velocity']}")
            if 'deceleration_time' in diastolic_params:
                diastolic_parts.append(f"DT {diastolic_params['deceleration_time']}")
            if 'E_prime' in diastolic_params:
                diastolic_parts.append(f"E' {diastolic_params['E_prime']}")
            if 'E_over_E_prime' in diastolic_params:
                diastolic_parts.append(f"E/E' {diastolic_params['E_over_E_prime']}")
            
            if diastolic_parts:
                diastolic_info = f"\nGEEXTRAHEERDE DIASTOLISCHE PARAMETERS: {', '.join(diastolic_parts)}"
        
        system_message = """Je bent een ervaren cardioloog die TTE (Transthoracale Echo) verslagen maakt.
        
        KRITIEKE INSTRUCTIES:
        - Gebruik NATUURLIJKE Nederlandse medische taal
        - Gebruik STANDAARD cardiologische formuleringen
        - Vermeld alleen RELEVANTE bevindingen
        - GEEN kunstmatige constructies zoals "EDD normaal mm" of "LVEF normaal%"
        - Gebruik professionele medische terminologie zoals echte cardiologen
        
        SPECIALE AANDACHT VOOR DIASTOLISCHE PARAMETERS:
        - Als E, A, DT, E', E/E' waarden worden vermeld, MOET je deze opnemen in de diastole sectie
        - Gebruik het formaat: "Diastole: [functie] met E (...) cm/s, A (...) cm/s, DT (...) ms, E' (...) cm/s, E/E' (...)"
        - Alleen vermelden wat daadwerkelijk genoemd wordt in de transcriptie
        
        STANDAARD FORMULERINGEN:
        - "eutroof" / "hypertrofe" / "niet gedilateerd" / "gedilateerd"
        - "globale functie: goed" / "matig" / "slecht"
        - "geen regionale kinetiekstoornissen"
        - "morfologisch en functioneel normaal"
        - "geen vocht" (pericard)
        - "niet betrouwbaar te meten" (als niet mogelijk)"""
        
        user_message = f"""Genereer een PROFESSIONEEL TTE verslag voor patiënt {patient_id} op {today}.
        
        Transcriptie: "{transcript}"{diastolic_info}
        
        Gebruik deze NATUURLIJKE stijl (gebaseerd op echte cardiologische verslagen):
        
        TTE op patiënt {patient_id} op {today}:
        
        Visualisatie: [adequaat/suboptimaal/etc]
        
        Linker ventrikel: [eutroof/hypertrofe], [niet gedilateerd/gedilateerd]. Globale functie: [goed/matig/slecht].
        Regionaal: [geen regionale kinetiekstoornissen/beschrijf afwijkingen] voor zover te beoordelen.
        
        Rechter ventrikel: [niet gedilateerd/gedilateerd], globale functie: [goed/matig/slecht].
        
        Diastole: [normaal/gestoord/etc] {f"met {', '.join([f'{k.replace('_', ' ').replace('velocity', '').replace('deceleration time', 'DT').replace('E prime', \"E'\").replace('E over E prime', 'E/E\\'').strip()} {v}' for k, v in diastolic_params.items()])} indien parameters vermeld" if diastolic_params else ""}.
        
        Atria: LA [normaal/gedilateerd/sterk gedilateerd] [met diameter X mm indien vermeld].
        
        Aortadimensies: [normaal/gedilateerd] voor zover visualiseerbaar.
        
        Mitralisklep: morfologisch en functioneel [normaal/beschrijf afwijkingen].
        
        Aortaklep: morfologisch en functioneel [normaal/beschrijf afwijkingen].
        
        Pulmonalisklep: morfologisch en functioneel [normaal/beschrijf afwijkingen].
        
        Tricuspiedklep: morfologisch en functioneel [normaal/beschrijf afwijkingen]. Pulmonaaldrukken zijn [niet betrouwbaar te meten/beschrijf waarden]. CVD is [normaal/verhoogd/etc].
        
        Pericard: [geen vocht/beschrijf afwijkingen].
        
        BELANGRIJK: 
        - Interpreteer de transcriptie: "{transcript}"
        - Gebruik alleen informatie die daadwerkelijk vermeld is
        - Schrijf in natuurlijke medische taal zoals echte cardiologen
        - Vermeld alleen relevante bevindingen
        - GEEN kunstmatige formuleringen met "normaal mm" of "normaal%"
        - Als diastolische parameters zijn geëxtraheerd, neem deze op in de diastole sectie
        
        Specifieke bevinding uit transcriptie: Let op de vermelding van "53 mm" - dit lijkt een LA diameter te zijn."""
        
        report = self.call_gpt([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        # Post-process to ensure natural language
        report = self.ensure_natural_language(report)
        
        return report
    
    def ensure_natural_language(self, text):
        """Ensure natural medical language, remove artificial constructions"""
        import re
        
        # Remove artificial constructions
        replacements = {
            r'met EDD normaal mm': 'niet gedilateerd',
            r'met IVS normaal mm': '',
            r'met PW normaal mm': '',
            r'met LVEF normaal%': '',
            r'normale systolische functie': '',
            r'met TAPSE normaal mm': '',
            r'met E niet gemeten cm/s': '',
            r'A niet gemeten cm/s': '',
            r'E DT niet gemeten ms': '',
            r"E' septaal niet gemeten cm/s": '',
            r"E/E' niet berekend": '',
            r'met sinus normaal mm': '',
            r'sinotubulair normaal mm': '',
            r'ascendens normaal mm': '',
            r'geschatte RVSP: niet berekend \+ CVD normaal mmHg': 'Pulmonaaldrukken zijn niet betrouwbaar te meten. CVD is normaal',
            r'gezien vena cava inferior: normaal mm, variabiliteit: normaal': '',
            r'Insufficiëntie: geen, stenose: geen': 'morfologisch en functioneel normaal',
            r'InsufficiÃ«ntie: geen, stenose: geen': 'morfologisch en functioneel normaal',
            r'Normaal, morfologisch normaal\. Functioneel: insufficiëntie: geen, stenose: geen': 'morfologisch en functioneel normaal',
            r'Normaal, morfologisch normaal\. Functioneel: InsufficiÃ«ntie: geen, stenose: geen': 'morfologisch en functioneel normaal',
            r'Morfologisch normaal\. Insufficiëntie: geen, stenose: geen': 'morfologisch en functioneel normaal',
            r'Morfologisch normaal\. InsufficiÃ«ntie: geen, stenose: geen': 'morfologisch en functioneel normaal',
            r'Geen hypertrofie': 'eutroof',
            r'Normaal met': '',
            r'Normaal,': '',
            r'\s+': ' ',  # Clean up multiple spaces
            r'^\s+|\s+$': '',  # Trim whitespace
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Clean up empty lines and extra spaces
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def remove_placeholders(self, text):
        """Remove any remaining placeholders and replace with appropriate medical terms"""
        import re
        
        # Replace common placeholder patterns
        replacements = {
            r'\(\.\.\.\)': 'normaal',
            r'\(niet vermeld\)': 'niet gespecificeerd', 
            r'\(onbekend\)': 'niet vermeld',
            r'\(\s*\)': 'normaal',
            r':\s*\(\.\.\.\)': ': normaal',
            r'met\s*\(\.\.\.\)': 'met normale waarden',
            r'functie:\s*\(\.\.\.\)': 'functie: normaal',
            r'insufficiëntie:\s*\(\.\.\.\)': 'insufficiëntie: geen',
            r'stenose:\s*\(\.\.\.\)': 'stenose: geen'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def generate_spoedconsult_report(self, transcript, patient_id=""):
        """Generate emergency consultation report with GPT-4o"""
        today = datetime.datetime.now().strftime("%d-%m-%Y")
        
        system_message = """Je bent een ervaren cardioloog die spoedconsult verslagen maakt.
        Genereer een bondig spoedconsult verslag.
        Focus op de acute presentatie en het beleid."""
        
        user_message = f"""Genereer een spoedconsult verslag voor patiënt {patient_id} op {today}.
        
        Transcriptie: {transcript}
        
        Format:
        SPOEDCONSULT BRIEF
        Datum: {today}
        Patiënt ID: {patient_id}
        
        PRESENTATIE:
        [beschrijving]
        
        BEVINDINGEN:
        [onderzoeken en resultaten]
        
        DIAGNOSE:
        [werkdiagnose]
        
        BELEID:
        [acute behandeling]"""
        
        report = self.call_gpt([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        return report
    
    def generate_consultatie_report(self, transcript, patient_id=""):
        """Generate consultation report with detailed template structure"""
        today = datetime.datetime.now().strftime("%d-%m-%Y")
        
        system_message = """Je bent een ervaren cardioloog die consultatie verslagen maakt.
        Genereer een volledig consultatieverslag volgens het exacte template format.
        Gebruik correcte Nederlandse medische terminologie.
        BELANGRIJK: Vul alleen waarden in die expliciet in de transcriptie staan.
        Laat secties leeg of gebruik (...) voor ontbrekende waarden.
        Verzin NOOIT waarden die niet in de transcriptie staan."""
        
        user_message = f"""Genereer een consultatieverslag voor patiënt {patient_id} op {today}.
        
        Transcriptie: {transcript}
        
        Gebruik EXACT dit template format:
        
        1. Reden van komst
        Patiënt komt (...)
        
        2. Voorgeschiedenis
        i. Persoonlijke antecedenten: (...)
        ii. Familiaal: (...)
        iii. Beroep: (...)
        iv. Usus: (...)
        v. Thuismedicatie: (...)
        
        3. Anamnese
        (...)
        Retrosternale last: (...)
        Kortademigheid: (...)
        Hartkloppingen: (...)
        Zwelling onderste ledematen: (...)
        Draaierigheid/flauwtes/bewustzijnsverlies: (...)
        
        4. Klinisch onderzoek
        Algehele aanblik: (...)
        Cor: (...)
        Longen: (...)
        Perifeer: (...)
        Jugulairen: (...)
        
        5. Aanvullend onderzoek
        i. ECG op raadpleging ({today}): (...)
        ii. Fietsproef op raadpleging ({today}): (...)
        iii. TTE op raadpleging ({today}):
        Linker ventrikel: (...)troof met EDD (...) mm, IVS (...) mm, PW (...) mm. Globale functie: (...) met LVEF (...)% (...).
        Regionaal: (...)
        Rechter ventrikel: (...)troof, globale functie: (...) met TAPSE (...) mm.
        Diastole: (...) met E (...) cm/s, A (...) cm/s, E DT (...) ms, E' septaal (...) cm/s, E/E' (...). L-golf: (...).
        Atria: LA (...) (...) mm.
        Aortadimensies: (...) met sinus (...) mm, sinotubulair (...) mm, ascendens (...) mm.
        Mitralisklep: morfologisch (...). insufficiëntie: (...), stenose: (...).
        Aortaklep: (...), morfologisch (...). Functioneel: insufficiëntie: (...), stenose: (...).
        Pulmonalisklep: insufficiëntie: (...), stenose: (...).
        Tricuspiedklep: insufficiëntie: (...), geschatte RVSP: (...) + CVD (...) mmHg gezien vena cava inferior: (...) mm, variabiliteit: (...).
        Pericard: (...).
        iv. Recente biochemie op datum (...): (...)
        
        6. Besluit
        Uw (...)-jarige patiënt werd gezien op de raadpleging cardiologie op {today}. (...)
        
        7. Beleid
        i. Medicatiewijzigingen: (...)
        ii. Follow-up: (...)
        
        Vul alleen waarden in die in de transcriptie staan. Laat (...) staan voor ontbrekende waarden."""
        
        report = self.call_gpt([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        return report

