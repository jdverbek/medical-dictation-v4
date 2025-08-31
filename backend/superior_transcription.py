"""
Superior Medical Transcription System
Based on analysis of the superior v2 app
"""

import os
import io
import datetime
import openai

class SuperiorMedicalTranscription:
    def __init__(self):
        openai.api_key = os.environ.get('OPENAI_API_KEY')
    
    def detect_audio_format(self, file_content, filename):
        """Detect actual audio format from file content"""
        # Check if file is WebM (common issue with browser recordings)
        if file_content.startswith(b'\x1a\x45\xdf\xa3'):
            print("DEBUG: File is WebM format, adjusting content type")
            content_type = 'audio/webm'
            filename = filename.replace('.wav', '.webm')
        else:
            # Default to original content type
            content_type = 'audio/wav'
        
        return content_type, filename
    
    def transcribe_audio(self, audio_file):
        """Superior audio transcription with WebM detection and error handling"""
        try:
            # Reset file pointer to beginning
            audio_file.seek(0)
            
            # Read file content for analysis
            file_content = audio_file.read()
            audio_file.seek(0)  # Reset again
            
            print(f"DEBUG: File size: {len(file_content)} bytes")
            print(f"DEBUG: File name: {audio_file.filename}")
            print(f"DEBUG: First 20 bytes: {file_content[:20]}")
            
            # Detect actual format
            content_type, filename = self.detect_audio_format(file_content, audio_file.filename)
            
            # Transcribe with Whisper using older API
            # Create a file-like object for the older API
            audio_file_obj = io.BytesIO(file_content)
            audio_file_obj.name = filename
            
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file_obj,
                temperature=0.0
            )
            corrected_transcript = transcript.text
            
            # Check if transcription is empty or too short
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

SPOEDCONSULT INTELLIGENTIE:
- "nieuwe vkf" → volledige voorkamerfibrillatie uitleg met behandeling
- "opname" → complete opname indicatie met beleid
- "standaardbehandeling" → specifieke medicatie en dosering
- "ACS" → acuut coronair syndroom protocol
- "arixtra" in ACS context → Arixtra (NIET xarelto)
- "cedocard" voor angina → Cedocard

TEMPLATE STRUCTUUR:

SPOEDCONSULT BRIEF
Patiënt ID: {patient_id}
Datum: {today}

PRESENTATIE:
[Uitgebreide beschrijving gebaseerd op keywords]

BEVINDINGEN:
[Relevante onderzoeken en resultaten]

DIAGNOSE:
[Primaire en secundaire diagnoses]

BELEID:
[Specifieke behandeling en medicatie]

FOLLOW-UP:
[Vervolgafspraken en monitoring]

KEYWORDS EXPANSIE REGELS:
- "nieuwe vkf" → "Patiënt presenteert zich met nieuwe voorkamerfibrillatie. ECG toont irregulaire RR-intervallen zonder P-golven. Hemodynamisch stabiel."
- "opname" → "Opname geïndiceerd voor monitoring en verdere diagnostiek/behandeling"
- "standaardbehandeling" → Specifieke medicatie met doseringen
- "coronarografie" → "Coronarografie gepland voor evaluatie coronaire anatomie"

MEDICATIE CONTEXT-AWARENESS:
- ACS + anticoagulatie = Arixtra (niet Xarelto)
- Angina + nitraten = Cedocard
- VKF + anticoagulatie = context-afhankelijk

DICTAAT KEYWORDS:
{transcript}
"""
        
        try:
            # Generate spoedconsult report
            report = self.call_gpt([
                {"role": "system", "content": spoedconsult_instruction},
                {"role": "user", "content": "Genereer een volledige spoedconsult brief op basis van de keywords."}
            ])
            
            # Perform quality control review
            final_report = self.quality_control_review(report, transcript)
            
            return final_report
            
        except Exception as e:
            return f"Error generating spoedconsult report: {str(e)}"

