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
        print(f"🎤 Audio transcription client initialized with OpenAI GPT-4o Transcribe")
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 10 else 'short'}")
    
    def convert_audio_to_wav(self, file_content, original_filename):
        """Convert audio file to WAV format using ffmpeg"""
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
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes timeout
            
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
                'error': "Audio conversie timeout (>5 min). Bestand te groot of complex."
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
        Gebruik correcte Nederlandse medische terminologie.
        BELANGRIJK: Vul alleen waarden in die expliciet in de transcriptie staan.
        Laat secties leeg of gebruik (...) voor ontbrekende waarden.
        Verzin NOOIT waarden die niet in de transcriptie staan."""
        
        user_message = f"""Genereer een TEE verslag voor patiënt {patient_id} op {today}.
        
        Transcriptie: {transcript}
        
        Gebruik EXACT dit template format:
        
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
        - Mitralisklep: (...), morfologisch (...), er is (...) insufficientie, er is (...) stenose, (...) toegevoegde structuur.
        * Mitraalinsufficientie vena contracta (...) mm, ERO (...) mm2 en RVol (...) ml/slag.
        - Aortaklep: (...), morfologisch (...), (...) verkalkt, er is (...) insufficientie, er is (...) stenose (...) toegevoegde structuur.
        Dimensies: LVOT (...) mm, aorta sinus (...) mm, sinutubulaire junctie (...) mm, aorta ascendens boven de sinutubulaire junctie (...) mm.
        * Aortaklepinsufficientie vena contracta (...) mm, ERO (...) mm2 en RVol (...) ml/slag.
        * Aortaklepstenose piekgradient (...) mmHg en gemiddelde gradient (...) mmHg, effectief klepoppervlak (...) cm2.
        - Tricuspiedklep: (...), morfologisch (...), er is (...) insufficientie, (...) toegevoegde structuur.
        * Systolische pulmonaaldruk afgeleid uit TI (...) mmHg + CVD.
        - Pulmonaalklep is (...), er is (...) insufficientie.
        - Aorta ascendens is (...) gedilateerd, graad (...) atheromatose van de aortawand.
        - Pulmonale arterie is (...) gedilateerd.
        - Vena cava inferior/levervenes zijn (...) verbreed (...) ademvariatie.
        - Pericard: er is (...) pericardvocht.
        
        Vul alleen waarden in die in de transcriptie staan. Laat (...) staan voor ontbrekende waarden."""
        
        report = self.call_gpt([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
    def generate_tte_report(self, transcript, patient_id=""):
        """Generate TTE report with updated structure"""
        today = datetime.datetime.now().strftime("%d-%m-%Y")
        
        system_message = """Je bent een ervaren cardioloog die TTE (Transthoracale Echo) verslagen maakt.
        Genereer een gestructureerd TTE verslag volgens het exacte template format.
        Gebruik correcte Nederlandse medische terminologie.
        BELANGRIJK: Vul alleen waarden in die expliciet in de transcriptie staan.
        Laat secties leeg of gebruik (...) voor ontbrekende waarden.
        Verzin NOOIT waarden die niet in de transcriptie staan."""
        
        user_message = f"""Genereer een TTE verslag voor patiënt {patient_id} op {today}.
        
        Transcriptie: {transcript}
        
        Gebruik EXACT dit template format:
        
        TTE op (...) op {today}:
        Visualisatie: (...)
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
        
        Vul alleen waarden in die in de transcriptie staan. Laat (...) staan voor ontbrekende waarden."""
        
        report = self.call_gpt([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        return report
    
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

