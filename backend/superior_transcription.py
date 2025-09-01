"""
Superior Medical Transcription System
Based on analysis of the superior v2 app
"""

import os
import io
import datetime
import openai
import subprocess
import tempfile

class SuperiorMedicalTranscription:
    def __init__(self):
        # CRITICAL: Remove proxy environment variables that cause issues on Render.com
        import os
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        for var in proxy_vars:
            if var in os.environ:
                print(f"🔧 Removing proxy env var: {var}")
                os.environ.pop(var, None)
        
        # For audio transcription, we need to use the standard OpenAI API
        # The Manus proxy doesn't support audio endpoints
        try:
            from openai import OpenAI
            
            # Check if we have a separate audio API key
            audio_api_key = os.environ.get('OPENAI_AUDIO_API_KEY') or os.environ.get('OPENAI_API_KEY')
            
            # Simple client initialization without extra parameters that might cause proxy issues
            self.audio_client = OpenAI(api_key=audio_api_key)
            print(f"🎤 Audio transcription client initialized with standard OpenAI API")
            print(f"🔑 Using API key: {audio_api_key[:10]}...{audio_api_key[-4:] if audio_api_key else 'None'}")
        except Exception as e:
            print(f"⚠️ OpenAI v1.0+ client failed: {e}")
            # Fallback to legacy OpenAI client
            import openai
            audio_api_key = os.environ.get('OPENAI_AUDIO_API_KEY') or os.environ.get('OPENAI_API_KEY')
            openai.api_key = audio_api_key
            # Don't set api_base to avoid proxy issues
            self.audio_client = None
            print(f"🎤 Using legacy OpenAI client for audio transcription")
            print(f"🔑 Using API key: {audio_api_key[:10]}...{audio_api_key[-4:] if audio_api_key else 'None'}")
    
    def convert_audio_to_wav(self, file_content, original_filename):
        """Convert audio file to WAV format using ffmpeg for gpt-4o-transcribe compatibility"""
        try:
            print(f"DEBUG: Converting {original_filename} to WAV format...")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_input:
                temp_input.write(file_content)
                temp_input_path = temp_input.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_output:
                temp_output_path = temp_output.name
            
            # Convert using ffmpeg
            cmd = [
                'ffmpeg', '-i', temp_input_path,
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
                '-ar', '16000',          # 16kHz sample rate (good for speech)
                '-ac', '1',              # Mono channel
                '-y',                    # Overwrite output file
                temp_output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
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
                    'filename': original_filename.replace('.webm', '.wav').replace('.mp4', '.wav'),
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
                'error': "Audio conversie timeout (>30s). Bestand te groot of complex."
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Audio conversie error: {str(e)}"
            }
    
    def detect_audio_format(self, file_content, filename):
        """Detect actual audio format from file content"""
        # Check if file is WebM (common issue with browser recordings)
        if file_content.startswith(b'\x1a\x45\xdf\xa3'):
            print("DEBUG: File is WebM format, will convert to WAV")
            return 'audio/webm', filename, True  # needs_conversion = True
        elif filename.lower().endswith('.webm'):
            print("DEBUG: File has .webm extension, will convert to WAV")
            return 'audio/webm', filename, True  # needs_conversion = True
        else:
            # Default to original content type
            return 'audio/wav', filename, False  # needs_conversion = False
    
    def transcribe_audio(self, audio_file, report_type="TTE"):
        """Superior audio transcription with automatic WebM conversion and enhanced error handling"""
        try:
            # Reset file pointer to beginning
            audio_file.seek(0)
            
            # Read file content for analysis
            file_content = audio_file.read()
            audio_file.seek(0)  # Reset again
            
            print(f"DEBUG: File size: {len(file_content)} bytes")
            print(f"DEBUG: File name: {audio_file.filename}")
            print(f"DEBUG: Report type: {report_type}")
            print(f"DEBUG: First 20 bytes: {file_content[:20]}")
            
            # Enhanced audio format validation for gpt-4o-transcribe
            if len(file_content) == 0:
                return {
                    'success': False,
                    'error': "⚠️ Audio bestand is leeg. Upload een geldig audio bestand."
                }
            
            if len(file_content) > 25 * 1024 * 1024:  # 25MB limit for OpenAI
                return {
                    'success': False,
                    'error': f"⚠️ Audio bestand te groot ({len(file_content)/1024/1024:.1f}MB). Maximum is 25MB voor gpt-4o-transcribe."
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
                        'error': f"⚠️ Automatische conversie gefaald\n\n{conversion_result['error']}\n\nProbeer handmatig te converteren naar .wav of .mp3 format."
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
            
            # Use whisper-1 (the actual OpenAI audio model) for all report types
            try:
                if self.audio_client:
                    # Use new OpenAI v1.0+ client
                    if report_type == "LIVE_CONSULTATIE":
                        # Use whisper-1 for live consultations with special prompt
                        transcript = self.audio_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file_obj,
                            language="nl",
                            prompt="""Je bent een medische secretaresse die aanwezig is bij een cardiologische consultatie waarbij een patiënt op bezoek komt bij de arts. Je hoort een conversatie tussen 2 of meerdere personen (soms zijn familieleden mee) en maakt een gedetailleerde samenvatting van de consultatie. Focus je vooral op de anamnese/symptomen, probeer deze zo getrouw mogelijk neer te pennen. Let op: soms zal de conversatie gestoord worden doordat de arts gebeld wordt of iemand binnenkomt; hier moet je goed bedacht op zijn (de context zal plots niet meer kloppen)."""
                        )
                    elif report_type == "CONSULTATIE":
                        # Use whisper-1 for structured consultation format
                        transcript = self.audio_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file_obj,
                            language="nl",
                            prompt="Dit is een Nederlandse medische dictatie van een cardioloog voor een gestructureerde consultatie. Gebruik correcte medische terminologie en behoud alle details voor het consultatieverslag."
                        )
                    else:
                        # Use whisper-1 for TTE and SPOEDCONSULT
                        transcript = self.audio_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file_obj,
                            language="nl",
                            prompt="Dit is een Nederlandse medische transcriptie van een cardioloog. Gebruik correcte medische terminologie."
                        )
                else:
                    # Use legacy OpenAI client with v1.0.0 syntax
                    if report_type == "LIVE_CONSULTATIE":
                        transcript = openai.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file_obj,
                            language="nl",
                            prompt="""Je bent een medische secretaresse die aanwezig is bij een cardiologische consultatie waarbij een patiënt op bezoek komt bij de arts. Je hoort een conversatie tussen 2 of meerdere personen (soms zijn familieleden mee) en maakt een gedetailleerde samenvatting van de consultatie. Focus je vooral op de anamnese/symptomen, probeer deze zo getrouw mogelijk neer te pennen. Let op: soms zal de conversatie gestoord worden doordat de arts gebeld wordt of iemand binnenkomt; hier moet je goed bedacht op zijn (de context zal plots niet meer kloppen)."""
                        )
                    elif report_type == "CONSULTATIE":
                        transcript = openai.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file_obj,
                            language="nl",
                            prompt="Dit is een Nederlandse medische dictatie van een cardioloog voor een gestructureerde consultatie. Gebruik correcte medische terminologie en behoud alle details voor het consultatieverslag."
                        )
                    else:
                        transcript = openai.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file_obj,
                            language="nl",
                            prompt="Dit is een Nederlandse medische transcriptie van een cardioloog. Gebruik correcte medische terminologie."
                        )
                    
            except Exception as transcription_error:
                error_msg = str(transcription_error).lower()
                
                if "corrupted" in error_msg or "unsupported" in error_msg:
                    return {
                        'success': False,
                        'error': f"⚠️ Audio Format Probleem\n\nHet .webm bestand kan niet worden verwerkt.\n\nMogelijke oplossingen:\n1. Converteer naar .wav of .mp3 format\n2. Gebruik een andere audio recorder\n3. Controleer of het bestand niet beschadigd is\n\nTechnische details:\n- Bestand: {filename}\n- Grootte: {len(file_content)} bytes\n- Error: {transcription_error}"
                    }
                elif "file size" in error_msg or "too large" in error_msg:
                    return {
                        'success': False,
                        'error': f"⚠️ Bestand te groot voor audio transcriptie\n\nHuidige grootte: {len(file_content)/1024/1024:.1f}MB\nMaximum: 25MB\n\nVerkort de opname of comprimeer het bestand."
                    }
                elif "401" in error_msg or "invalid_api_key" in error_msg:
                    return {
                        'success': False,
                        'error': f"⚠️ Audio Transcriptie API Configuratie Probleem\n\nDe OpenAI API key is niet geldig voor audio transcriptie.\n\nVereist:\n- Geldige OpenAI API key met audio transcriptie toegang\n- Stel OPENAI_AUDIO_API_KEY environment variable in\n\nTechnische details:\n- Error: {transcription_error}"
                    }
                else:
                    return {
                        'success': False,
                        'error': f"⚠️ Audio Transcriptie Fout\n\n{transcription_error}\n\nControleer:\n1. Audio bestand is niet beschadigd\n2. OpenAI API key is geldig\n3. Internet verbinding is stabiel\n\nProbeer opnieuw of neem contact op met de beheerder."
                    }
            
            # Get the transcript text
            corrected_transcript = transcript.text if hasattr(transcript, 'text') else str(transcript)
            
            # Validation check
            if not corrected_transcript or len(corrected_transcript.strip()) < 10:
                return {
                    'success': False,
                    'error': f"⚠️ Transcriptie probleem: Audio werd niet correct getranscribeerd.\n\nBestand info:\n- Grootte: {len(file_content)} bytes\n- Type: {content_type}\n- Resultaat: '{corrected_transcript}'\n\nProbeer opnieuw met een duidelijkere opname."
                }
            
            # Check for Whisper hallucination (repetitive prompt text)
            if self.detect_hallucination(corrected_transcript):
                return {
                    'success': False,
                    'error': f"🚨 Whisper Hallucinatie Gedetecteerd!\n\nHet audio bestand is te stil of onduidelijk. Whisper herhaalt de instructie in plaats van te transcriberen:\n\n'{corrected_transcript[:200]}...'\n\nOplossingen:\n- Spreek dichterbij de microfoon\n- Verhoog het volume\n- Verminder achtergrondgeluid\n- Spreek langzamer en duidelijker\n\nProbeer opnieuw met een betere opname."
                }
            
            print(f"DEBUG: Transcription length: {len(corrected_transcript)} characters")
            print(f"DEBUG: Transcription preview: {corrected_transcript[:200]}...")
            
            return {
                'success': True,
                'transcript': corrected_transcript
            }
            
        except Exception as e:
            print(f"DEBUG: Transcription error: {str(e)}")
            return {
                'success': False,
                'error': f"Transcriptie fout: {str(e)}"
            }
    
    def detect_hallucination(self, transcript):
        """Detect Whisper hallucination patterns"""
        if "transcribe" in transcript.lower() or "dictatie" in transcript.lower():
            # Count how many times the prompt appears
            prompt_count = transcript.lower().count("transcribe") + transcript.lower().count("dictatie")
            if prompt_count > 5:  # If prompt appears more than 5 times, it's likely hallucination
                return True
        return False
    
    def call_gpt(self, messages, model="gpt-4o", temperature=0.0):
        """Call GPT with error handling using older API"""
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def quality_control_review(self, structured_report, original_transcript):
        """Perform quality control review of the structured report"""
        
        review_instruction = f"""
Je bent een ervaren cardioloog die een tweede review doet van een medisch verslag. 
Controleer het verslag op de volgende punten:

CORRECTE MEDISCHE NEDERLANDSE TERMINOLOGIE (PRIORITEIT!):
- Corrigeer ALLE incorrecte samengestelde woorden:
  ❌ 'pulmonaardruk' → ✅ 'pulmonale druk'
  ❌ 'posteriorklepplat' → ✅ 'posterieur mitraalklepblad'
  ❌ 'tricuspiedklep' → ✅ 'tricuspidalisklep'
  ❌ 'mitraalsklep' → ✅ 'mitralisklep'
  ❌ 'aortaklep' → ✅ 'aortaklep'
  ❌ 'artredicotentie' → ✅ 'atriumfibrillatie'
  ❌ 'voorkamervipulatie' → ✅ 'voorkamerfibrillatie'
  ❌ 'zeeuwtachtigjarige' → ✅ '80-jarige'
  ❌ 'serocreatinine' → ✅ 'serumcreatinine'
  ❌ 'NC-program T' → ✅ 'NT-proBNP'
  ❌ 'spiekerlepensubstantie' → ✅ 'tricuspidalisklep insufficiëntie'
  ❌ 'sedocar' → ✅ 'Cedocard'
  ❌ 'arixtra' → ✅ 'Arixtra' (in ACS context)
- Gebruik ALTIJD correcte medische Nederlandse terminologie

MEDISCHE CONSISTENTIE:
- Zijn de metingen medisch logisch? (bijv. LVEF vs functie beschrijving)
- Zijn er tegenstrijdigheden tussen verschillende secties?
- Kloppen de verhoudingen tussen verschillende parameters?

TEMPLATE VOLLEDIGHEID:
- Zijn alle verplichte secties aanwezig?
- Is de formatting correct en consistent?
- Zijn er lege velden die ingevuld zouden moeten zijn?

LOGISCHE COHERENTIE:
- Klopt de conclusie met de bevindingen?
- Is het beleid logisch gebaseerd op de bevindingen?
- Zijn er missing links tussen bevindingen en conclusies?

MEDISCHE VEILIGHEID:
- Zijn er potentieel gevaarlijke inconsistenties?
- Zijn kritieke bevindingen correct weergegeven?
- Is de terminologie correct gebruikt?

DRUG NAME CORRECTIONS (CONTEXT-AWARE):
- In ACS context: "arixtra" is correct (not "xarelto")
- In angina context: "sedocar" → "Cedocard"
- Check medical context before correcting drug names

Als je fouten of inconsistenties vindt, corrigeer ze en geef het verbeterde verslag terug.
Als alles correct is, geef het originele verslag terug zonder wijzigingen.

BELANGRIJK: 
- Behoud de exacte template structuur
- Voeg GEEN nieuwe medische gegevens toe die niet in het origineel stonden
- Corrigeer alleen echte fouten en inconsistenties
- CORRIGEER ALTIJD incorrecte terminologie naar correcte medische Nederlandse termen
- Geef ALLEEN het gecorrigeerde verslag terug, geen uitleg

Origineel dictaat voor referentie:
{original_transcript}

Te reviewen verslag:
{structured_report}

Gecorrigeerd verslag:
"""
        
        try:
            reviewed_report = self.call_gpt([
                {"role": "system", "content": review_instruction},
                {"role": "user", "content": "Voer de quality control review uit."}
            ])
            
            return reviewed_report.strip()
        except Exception as e:
            # If review fails, return original report
            return structured_report
    
    def generate_consultatie_report(self, transcript: str, patient_id: str) -> str:
        """Generate structured consultatie report using GPT-5-mini with exact format"""
        try:
            # Import the template
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from consultatie_template import generate_consultatie_template
            
            # Get the base template
            base_template = generate_consultatie_template(transcript, patient_id)
            
            # Create prompt for GPT to fill in the template based on transcript
            prompt = f"""Je bent een ervaren cardioloog die een gestructureerd consultatieverslag moet maken.

TRANSCRIPT VAN DICTATIE:
{transcript}

INSTRUCTIES:
1. Vul het onderstaande consultatieverslag in gebaseerd op de informatie uit het transcript
2. Als iets niet wordt vermeld in het transcript, laat het dan staan maar vermeld "niet vermeld"
3. Als ik zeg "niet uitgevoerd", sla dan de beschrijving over en vermeld bijv. "Fietsproef op raadpleging (dd-mm-yyyy): niet uitgevoerd"
4. Behoud het EXACTE format en structuur
5. Gebruik alleen informatie uit het transcript, verzin niets
6. Vul alle relevante medische details in waar beschikbaar

CONSULTATIEVERSLAG TEMPLATE OM IN TE VULLEN:
{base_template}

Vul nu het verslag in gebaseerd op het transcript, behoud de exacte structuur en format:"""

            # Use the same OpenAI client configuration as the rest of the app
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Je bent een ervaren cardioloog die gestructureerde consultatieverlagen maakt. Volg het exacte format en vul alleen in wat uit het transcript blijkt."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=1.0
                )
                
                filled_report = response.choices[0].message.content.strip()
                
            except ImportError:
                # Fallback to legacy OpenAI client
                import openai
                openai.api_key = os.environ.get('OPENAI_API_KEY')
                # Don't set api_base to avoid proxy issues
                
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Je bent een ervaren cardioloog die gestructureerde consultatieverlagen maakt. Volg het exacte format en vul alleen in wat uit het transcript blijkt."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=1.0
                )
                
                filled_report = response.choices[0].message.content.strip()
            
            print(f"DEBUG: Generated consultatie report length: {len(filled_report)}")
            return filled_report
            
        except Exception as e:
            print(f"ERROR generating consultatie report: {e}")
            # Fallback to basic template
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from consultatie_template import generate_consultatie_template
            return generate_consultatie_template(transcript, patient_id)
    
    def generate_tte_report(self, transcript, patient_id=""):
        """Generate TTE report with superior template"""
        today = datetime.datetime.now().strftime("%d-%m-%Y")
        
        template_instruction = f"""
BELANGRIJK: U krijgt een intuïtief dictaat van een cardioloog. Dit betekent dat de informatie:
- Niet in de juiste volgorde staat
- In informele bewoordingen kan zijn
- Correcties kan bevatten
- Heen en weer kan springen tussen onderwerpen

KRITIEKE VEILIGHEIDSREGEL: VERZIN GEEN MEDISCHE GEGEVENS!

CORRECTE MEDISCHE NEDERLANDSE TERMINOLOGIE:
- Gebruik ALTIJD correcte medische Nederlandse termen
- GEEN samengestelde woorden zoals 'pulmonaardruk' → gebruik 'pulmonale druk'
- GEEN 'posteriorklepplat' → gebruik 'posterieur mitraalklepblad'
- GEEN 'tricuspiedklep' → gebruik 'tricuspidalisklep'
- GEEN 'mitraalsklep' → gebruik 'mitralisklep'
- GEEN 'aortaklep' → gebruik 'aortaklep'

CORRECTE TERMINOLOGIE VOORBEELDEN:
❌ FOUT: pulmonaardruk, posteriorklepplat, tricuspiedklep
✅ CORRECT: pulmonale druk, posterieur mitraalklepblad, tricuspidalisklep

Uw taak: Analyseer het dictaat en vul het TTE-template in met ALLEEN de WERKELIJK GENOEMDE BEVINDINGEN.

TEMPLATE STRUCTUUR REGELS:
- BEHOUD ALLE TEMPLATE LIJNEN - laat geen enkele regel weg
- Voor elke lijn: geef een medische beschrijving gebaseerd op wat genoemd is
- Voor specifieke parameters (cijfers): alleen invullen als expliciet genoemd
- Voor algemene beschrijvingen: gebruik logische medische termen
- GEBRUIK ALTIJD CORRECTE MEDISCHE NEDERLANDSE TERMINOLOGIE

INVUL REGELS:
1. EXPLICIET GENOEMDE AFWIJKINGEN: Vul exact in zoals gedicteerd MAAR met correcte terminologie
2. NIET GENOEMDE STRUCTUREN: Gebruik "normaal" of "eutroof" 
3. SPECIFIEKE CIJFERS: Alleen als letterlijk genoemd (EDD, LVEF, etc.)
4. ALGEMENE FUNCTIE: Afleiden uit context ("normale echo" = goede functie)
5. TERMINOLOGIE: Altijd correcte medische Nederlandse termen gebruiken

VOORBEELDEN VAN CORRECTE INVULLING:

Als "normale echo behalve..." gedicteerd:
- Linker ventrikel: eutroof, globale functie goed
- Regionaal: geen kinetiekstoornissen  
- Rechter ventrikel: normaal, globale functie goed

Als specifieke afwijking genoemd:
- Mitralisklep: morfologisch prolaps. insufficiëntie: spoortje
- Atria: LA licht vergroot 51 mm

Als niets specifiek genoemd:
- Aortaklep: tricuspied, morfologisch normaal. Functioneel: normaal
- Pericard: normaal

VOLLEDIGE TEMPLATE STRUCTUUR:

TTE op {today}:
- Linker ventrikel: [normaal/eutroof als niet anders vermeld, specifieke afwijkingen als genoemd]
- Regionaal: [geen kinetiekstoornissen als niet anders vermeld]
- Rechter ventrikel: [normaal als niet anders vermeld, specifieke afwijkingen als genoemd]
- Diastole: [normaal als niet anders vermeld, specifieke bevindingen als genoemd]
- Atria: [normaal als niet anders vermeld, specifieke afwijkingen als genoemd]
- Aortadimensies: [normaal als niet anders vermeld, specifieke metingen als genoemd]
- Mitralisklep: [morfologisch normaal als niet anders vermeld, specifieke afwijkingen als genoemd]
- Aortaklep: [tricuspied, morfologisch normaal als niet anders vermeld]
- Pulmonalisklep: [normaal als niet anders vermeld, specifieke afwijkingen als genoemd]
- Tricuspidalisklep: [normaal als niet anders vermeld, specifieke afwijkingen als genoemd]
- Pericard: [normaal als niet anders vermeld]

Recente biochemie op {today}:
[Alleen invullen als biochemie expliciet genoemd in dictaat]

Conclusie: [Samenvatting van werkelijk genoemde afwijkingen]

Beleid:
[Alleen invullen als expliciet genoemd in dictaat]

VEILIGHEIDSCHECK: Elk cijfer moet ECHT in het dictaat staan!
TERMINOLOGIE CHECK: Gebruik ALLEEN correcte medische Nederlandse termen!

DICTAAT:
{transcript}
"""
        
        try:
            # Generate initial report
            initial_report = self.call_gpt([
                {"role": "system", "content": template_instruction},
                {"role": "user", "content": "Genereer het TTE verslag volgens de instructies."}
            ])
            
            # Perform quality control review
            final_report = self.quality_control_review(initial_report, transcript)
            
            return final_report
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def generate_spoedconsult_report(self, transcript, patient_id=""):
        """Generate emergency consultation report from short keywords"""
        today = datetime.datetime.now().strftime("%d-%m-%Y")
        
        spoedconsult_instruction = f"""
Je bent een ervaren cardioloog die een spoedconsult brief moet schrijven op basis van zeer korte keywords.

BELANGRIJK: Dit is GEEN TTE verslag! Dit is een SPOEDCONSULT BRIEF!

SPOEDCONSULT INTELLIGENTIE:
- Analyseer de keywords en genereer relevante medische content
- "opname" → complete opname indicatie met beleid
- "standaardbehandeling" → specifieke medicatie en dosering gebaseerd op diagnose
- "ACS" → acuut coronair syndroom protocol
- "arixtra" in ACS context → Arixtra (NIET xarelto)
- "cedocard" voor angina → Cedocard

VERPLICHTE TEMPLATE STRUCTUUR (GEEN TTE!):

SPOEDCONSULT BRIEF
Patiënt ID: {patient_id}
Datum: {today}

PRESENTATIE:
[Uitgebreide beschrijving gebaseerd op keywords - GEEN echo bevindingen!]

BEVINDINGEN:
[Relevante onderzoeken en resultaten - GEEN TTE secties!]

DIAGNOSE:
[Primaire en secundaire diagnoses ALLEEN gebaseerd op vermelde keywords]

BELEID:
[Specifieke behandeling en medicatie ALLEEN voor geïdentificeerde condities]

FOLLOW-UP:
[Vervolgafspraken en monitoring]

KEYWORDS EXPANSIE REGELS:
- Analyseer keywords en genereer alleen relevante content
- "opname" → "Opname geïndiceerd voor monitoring en verdere diagnostiek/behandeling"
- "standaardbehandeling" → Specifieke medicatie en doseringen voor geïdentificeerde conditie
- "coronarografie" → "Coronarografie gepland voor evaluatie coronaire anatomie"

MEDICATIE CONTEXT-AWARENESS:
- ACS + anticoagulatie = Arixtra (niet Xarelto)
- Angina + nitraten = Cedocard
- Alleen medicatie voorschrijven voor daadwerkelijk geïdentificeerde condities

STRIKT VERBODEN:
- GEEN TTE secties (linker ventrikel, kleppen, etc.)
- GEEN biochemie secties
- GEEN echo bevindingen
- ALLEEN spoedconsult brief format!

DICTAAT KEYWORDS:
{transcript}
"""
        
        try:
            # Generate spoedconsult report WITHOUT quality control review
            # (to avoid TTE template contamination)
            report = self.call_gpt([
                {"role": "system", "content": spoedconsult_instruction},
                {"role": "user", "content": "Genereer een volledige spoedconsult brief volgens de EXACTE template structuur. GEEN TTE verslag!"}
            ])
            
            # Simple medical terminology correction only (no template changes)
            corrected_report = self.simple_medical_correction(report, transcript)
            
            return corrected_report
            
        except Exception as e:
            return f"Error generating spoedconsult report: {str(e)}"
    
    def simple_medical_correction(self, report, original_transcript):
        """Simple medical terminology correction without template changes"""
        
        # Basic medical term corrections
        corrections = {
            'artredicotentie': 'atriumfibrillatie',
            'voorkamervipulatie': 'voorkamerfibrillatie', 
            'zeeuwtachtigjarige': '80-jarige',
            'serocreatinine': 'serumcreatinine',
            'NC-program T': 'NT-proBNP',
            'sedocar': 'Cedocard',
            'arixtra': 'Arixtra',
            'mgpg': 'mg/dl',
            'kmpg': 'mmHg'
        }
        
        corrected_report = report
        for wrong, correct in corrections.items():
            corrected_report = corrected_report.replace(wrong, correct)
        
        return corrected_report

