"""
Superior Medical Transcription System
Fixed version with modern OpenAI API syntax
"""

import os
import io
import datetime
import tempfile
import subprocess
from openai import OpenAI

class SuperiorMedicalTranscription:
    def __init__(self):
        # Initialize OpenAI client with modern syntax
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Simple, clean initialization without proxy parameters
        self.client = OpenAI(api_key=api_key)
        print(f"🎤 Audio transcription client initialized with OpenAI API")
    
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
        """Audio transcription using modern OpenAI API"""
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
            
            # Prepare the prompt based on report type
            if report_type == "LIVE_CONSULTATIE":
                prompt = """Je bent een medische secretaresse die aanwezig is bij een cardiologische consultatie waarbij een patiënt op bezoek komt bij de arts. Je hoort een conversatie tussen 2 of meerdere personen (soms zijn familieleden mee) en maakt een gedetailleerde samenvatting van de consultatie. Focus je vooral op de anamnese/symptomen, probeer deze zo getrouw mogelijk neer te pennen. Let op: soms zal de conversatie gestoord worden doordat de arts gebeld wordt of iemand binnenkomt; hier moet je goed bedacht op zijn (de context zal plots niet meer kloppen)."""
            elif report_type == "CONSULTATIE":
                prompt = "Dit is een Nederlandse medische dictatie van een cardioloog voor een gestructureerde consultatie. Gebruik correcte medische terminologie en behoud alle details voor het consultatieverslag."
            else:
                prompt = "Dit is een Nederlandse medische transcriptie van een cardioloog. Gebruik correcte medische terminologie."
            
            # Use modern OpenAI API syntax for transcription
            print("DEBUG: Calling OpenAI transcription API...")
            transcript_response = self.client.audio.transcriptions.create(
                model="whisper-1",  # The correct model name for audio transcription
                file=audio_file_obj,
                language="nl",
                prompt=prompt,
                response_format="text"
            )
            
            # Get the transcript text
            corrected_transcript = transcript_response if isinstance(transcript_response, str) else transcript_response.text
            
            # Validation check
            if not corrected_transcript or len(corrected_transcript.strip()) < 10:
                return {
                    'success': False,
                    'error': f"⚠️ Transcriptie probleem: Audio werd niet correct getranscribeerd.\n\nResultaat: '{corrected_transcript}'"
                }
            
            # Check for Whisper hallucination
            if self.detect_hallucination(corrected_transcript):
                return {
                    'success': False,
                    'error': f"🚨 Whisper Hallucinatie Gedetecteerd!\n\nHet audio bestand is te stil of onduidelijk."
                }
            
            print(f"DEBUG: Transcription successful! Length: {len(corrected_transcript)} characters")
            
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
            else:
                return {
                    'success': False,
                    'error': f"⚠️ Transcriptie fout: {str(e)}"
                }
    
    def detect_hallucination(self, transcript):
        """Detect Whisper hallucination patterns"""
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
    
    def call_gpt(self, messages, model="gpt-4o-mini", temperature=0.7):
        """Call GPT with modern API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"GPT API error: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_tte_report(self, transcript, patient_id=""):
        """Generate TTE report"""
        today = datetime.datetime.now().strftime("%d-%m-%Y")
        
        system_message = """Je bent een ervaren cardioloog die TTE verslagen maakt. 
        Genereer een gestructureerd TTE verslag op basis van de transcriptie.
        Gebruik correcte Nederlandse medische terminologie.
        Vul alleen in wat expliciet in de transcriptie staat."""
        
        user_message = f"""Genereer een TTE verslag voor patiënt {patient_id} op {today}.
        
        Transcriptie: {transcript}
        
        Gebruik dit format:
        TTE op {today}:
        - Linker ventrikel: [beschrijving]
        - Rechter ventrikel: [beschrijving]
        - Atria: [beschrijving]
        - Kleppen: [beschrijving]
        - Conclusie: [samenvatting]"""
        
        report = self.call_gpt([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        return report
    
    def generate_spoedconsult_report(self, transcript, patient_id=""):
        """Generate emergency consultation report"""
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
        """Generate consultation report"""
        today = datetime.datetime.now().strftime("%d-%m-%Y")
        
        system_message = """Je bent een ervaren cardioloog die consultatie verslagen maakt.
        Genereer een volledig consultatieverslag volgens het standaard format.
        Gebruik correcte Nederlandse medische terminologie."""
        
        user_message = f"""Genereer een consultatieverslag voor patiënt {patient_id} op {today}.
        
        Transcriptie: {transcript}
        
        Gebruik het standaard consultatie format met:
        - Reden van komst
        - Anamnese
        - Lichamelijk onderzoek
        - Aanvullend onderzoek
        - Conclusie
        - Beleid"""
        
        report = self.call_gpt([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        return report

