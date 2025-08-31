# backend/medical_intelligence.py - Advanced Medical AI System

"""
Medical Intelligence v4.0
- Context-aware drug recognition
- Emergency consultation expansion
- Professional echo report templates
- Multi-agent orchestration
"""

import re
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime

class MedicalContextAgent:
    """Understands clinical contexts and situations"""
    
    def __init__(self):
        self.clinical_contexts = {
            'acs': ['acuut coronair syndroom', 'hartinfarct', 'nstemi', 'stemi', 'instabiele angina'],
            'vkf': ['voorkamerfibrillatie', 'atriumfibrillatie', 'nieuwe vkf', 'vkf'],
            'hartfalen': ['hartfalen', 'decompensatio cordis', 'lvef', 'systolisch hartfalen'],
            'hypertensie': ['hypertensie', 'hoge bloeddruk', 'rvr'],
            'trombose': ['trombose', 'longembolie', 'dvt', 'pe'],
            'spoedconsult': ['spoed', 'urgent', 'acuut', 'opname', 'emergency']
        }
        
        self.context_drugs = {
            'acs': {
                'arixtra': ['fondaparinux', 'arixtra'],
                'xarelto': ['rivaroxaban', 'xarelto'],
                'plavix': ['clopidogrel', 'plavix'],
                'aspirin': ['aspirine', 'cardioaspirine', 'asacard']
            },
            'vkf': {
                'xarelto': ['rivaroxaban', 'xarelto'],
                'eliquis': ['apixaban', 'eliquis'],
                'pradaxa': ['dabigatran', 'pradaxa'],
                'sintrom': ['acenocoumarol', 'sintrom']
            },
            'hartfalen': {
                'lasix': ['furosemide', 'lasix'],
                'lisinopril': ['lisinopril', 'ace-remmer'],
                'metoprolol': ['metoprolol', 'selokeen']
            }
        }
    
    def detect_context(self, text: str) -> List[str]:
        """Detect clinical contexts in text"""
        contexts = []
        text_lower = text.lower()
        
        for context, keywords in self.clinical_contexts.items():
            if any(keyword in text_lower for keyword in keywords):
                contexts.append(context)
        
        return contexts
    
    def resolve_drug_in_context(self, drug_sound: str, contexts: List[str]) -> str:
        """Resolve drug name based on clinical context"""
        drug_lower = drug_sound.lower()
        
        # Special case: arixtra vs xarelto in ACS context
        if 'acs' in contexts:
            if any(sound in drug_lower for sound in ['arixtra', 'arikstra', 'aristra']):
                return 'arixtra (fondaparinux)'
            elif any(sound in drug_lower for sound in ['xarelto', 'sarelto']):
                # In ACS context, this is likely arixtra
                return 'arixtra (fondaparinux) - gecorrigeerd van xarelto obv ACS context'
        
        # Context-specific drug matching
        for context in contexts:
            if context in self.context_drugs:
                for correct_drug, variations in self.context_drugs[context].items():
                    if any(var in drug_lower for var in variations):
                        return f"{correct_drug} - gevalideerd voor {context}"
        
        return drug_sound  # Return original if no context match

class EmergencyConsultAgent:
    """Expands emergency consultation keywords into full medical letters"""
    
    def __init__(self):
        self.emergency_templates = {
            'nieuwe_vkf': {
                'keywords': ['nieuwe vkf', 'nieuwe voorkamerfibrillatie', 'nieuwe af'],
                'template': """
SPOEDCONSULT CARDIOLOGIE

Anamnese:
Patiënt presenteert zich met nieuwe voorkamerfibrillatie. 

Klinisch onderzoek:
- Irregulaire hartritme
- Hemodynamisch stabiel/instabiel (te specificeren)

Diagnostiek:
- ECG: voorkamerfibrillatie
- Laboratorium: TSH, elektrolieten, troponine
- Echocardiografie indien geïndiceerd

Beleid:
- Ritmestabilisatie vs frequentiecontrole
- Anticoagulatie volgens CHA2DS2-VASc score
- Oorzaak onderzoek

Medicatie:
- Metoprolol voor frequentiecontrole
- Anticoagulatie (rivaroxaban/apixaban) indien geïndiceerd
"""
            },
            'opname': {
                'keywords': ['opname', 'hospitalisatie', 'admission'],
                'template': """
OPNAME INDICATIE

Reden opname:
{reason}

Voorgesteld beleid:
- Monitoring vitale parameters
- Laboratorium controle
- Medicatie aanpassing indien nodig
- Multidisciplinair overleg

Behandelplan:
- Standaard cardiologische zorg
- Specifieke interventies naar gelang pathologie
"""
            },
            'standaardbehandeling': {
                'keywords': ['standaardbehandeling', 'standaard behandeling', 'routine care'],
                'template': """
STANDAARD CARDIOLOGISCHE BEHANDELING

Medicamenteuze therapie:
- ACE-remmer/ARB voor cardioprotectie
- Bètablokker voor frequentiecontrole
- Statine voor cholesterolverlaging
- Antiplatelet therapie indien geïndiceerd

Leefstijladviezen:
- Rookstop
- Gewichtscontrole
- Regelmatige lichaamsbeweging
- Zoutbeperking

Follow-up:
- Controle over 3 maanden
- Laboratorium: lipiden, nierfunctie
- Echocardiografie indien geïndiceerd
"""
            }
        }
    
    def detect_emergency_keywords(self, text: str) -> List[str]:
        """Detect emergency consultation keywords"""
        detected = []
        text_lower = text.lower()
        
        for template_name, template_data in self.emergency_templates.items():
            if any(keyword in text_lower for keyword in template_data['keywords']):
                detected.append(template_name)
        
        return detected
    
    def expand_emergency_consult(self, text: str, patient_id: str) -> str:
        """Expand emergency keywords into full consultation letter"""
        detected = self.detect_emergency_keywords(text)
        
        if not detected:
            return text
        
        expanded_parts = []
        
        for template_name in detected:
            template = self.emergency_templates[template_name]['template']
            expanded_parts.append(template.format(reason=text, patient_id=patient_id))
        
        if expanded_parts:
            header = f"""
SPOEDCONSULT BRIEF
Patiënt ID: {patient_id}
Datum: {datetime.now().strftime('%d-%m-%Y %H:%M')}

Oorspronkelijke input: "{text}"

UITGEBREIDE CONSULTATIE:
"""
            return header + "\n".join(expanded_parts)
        
        return text

class EchoReportAgent:
    """Professional echo report templates"""
    
    def __init__(self):
        self.tte_template = """
TRANSTHORACALE ECHOCARDIOGRAFIE (TTE)

TECHNISCHE KWALITEIT:
Goede/matige/slechte beeldkwaliteit

LINKER VENTRIKEL:
- Afmetingen: normaal/vergroot
- Wanddikte: normaal/hypertrofie
- Systolische functie: LVEF {lvef}% - {lvef_description}
- Regionale wandbewegingsstoornissen: {rwbs}
- Diastolische functie: {diastolic}

RECHTER VENTRIKEL:
- Afmetingen: normaal/vergroot
- Systolische functie: {rv_function}
- Systolische druk: {rvsp} mmHg (geschat)

ATRIA:
- Linker atrium: {la_size}
- Rechter atrium: {ra_size}

KLEPPEN:
- Aortaklep: {av_description}
- Mitralisklep: {mv_description}
- Tricuspidalisklep: {tv_description}
- Pulmonalisklep: {pv_description}

PERICARDIUM:
{pericardium}

CONCLUSIE:
{conclusion}
"""
        
        self.tee_template = """
TRANSESOFAGEALE ECHOCARDIOGRAFIE (TEE)

INDICATIE:
{indication}

TECHNISCHE KWALITEIT:
Goede beeldkwaliteit, volledige studie

LINKER VENTRIKEL:
- Systolische functie: LVEF {lvef}%
- Regionale wandbewegingsstoornissen: {rwbs}

LINKER ATRIUM:
- Afmetingen: {la_size}
- Spontane echo contrast: {sec}
- Linker hartoortje: {laa}

KLEPPEN (GEDETAILLEERD):
- Aortaklep: {av_detailed}
- Mitralisklep: {mv_detailed}
  * Anatomie: {mv_anatomy}
  * Functie: {mv_function}

AORTA:
- Aortawortel: {aortic_root}
- Ascenderende aorta: {ascending_aorta}

CONCLUSIE:
{conclusion}
"""
    
    def generate_tte_report(self, findings: Dict[str, Any]) -> str:
        """Generate structured TTE report"""
        
        # Default values with smart interpretation
        lvef = findings.get('lvef', 'niet vermeld')
        
        # LVEF interpretation
        if isinstance(lvef, (int, float)):
            if lvef >= 55:
                lvef_description = "normale systolische functie"
            elif lvef >= 45:
                lvef_description = "licht verminderde systolische functie"
            elif lvef >= 35:
                lvef_description = "matig verminderde systolische functie"
            else:
                lvef_description = "ernstig verminderde systolische functie"
        else:
            lvef_description = "systolische functie te beoordelen"
        
        return self.tte_template.format(
            lvef=lvef,
            lvef_description=lvef_description,
            rwbs=findings.get('rwbs', 'geen'),
            diastolic=findings.get('diastolic', 'normaal'),
            rv_function=findings.get('rv_function', 'normaal'),
            rvsp=findings.get('rvsp', 'niet meetbaar'),
            la_size=findings.get('la_size', 'normaal'),
            ra_size=findings.get('ra_size', 'normaal'),
            av_description=findings.get('av', 'normaal'),
            mv_description=findings.get('mv', 'normaal'),
            tv_description=findings.get('tv', 'normaal'),
            pv_description=findings.get('pv', 'normaal'),
            pericardium=findings.get('pericardium', 'geen vocht'),
            conclusion=findings.get('conclusion', 'Te completeren door cardioloog')
        )
    
    def generate_tee_report(self, findings: Dict[str, Any]) -> str:
        """Generate structured TEE report"""
        return self.tee_template.format(
            indication=findings.get('indication', 'Niet vermeld'),
            lvef=findings.get('lvef', 'niet vermeld'),
            rwbs=findings.get('rwbs', 'geen'),
            la_size=findings.get('la_size', 'normaal'),
            sec=findings.get('sec', 'geen'),
            laa=findings.get('laa', 'geen trombus'),
            av_detailed=findings.get('av_detailed', 'tricuspidaal, normaal'),
            mv_detailed=findings.get('mv_detailed', 'normaal'),
            mv_anatomy=findings.get('mv_anatomy', 'normale anatomie'),
            mv_function=findings.get('mv_function', 'competent'),
            aortic_root=findings.get('aortic_root', 'normaal'),
            ascending_aorta=findings.get('ascending_aorta', 'normaal'),
            conclusion=findings.get('conclusion', 'Te completeren door cardioloog')
        )

class MedicalOrchestratorV2:
    """Advanced multi-agent medical orchestrator"""
    
    def __init__(self):
        self.context_agent = MedicalContextAgent()
        self.emergency_agent = EmergencyConsultAgent()
        self.echo_agent = EchoReportAgent()
        self.max_iterations = 5
    
    def process_medical_transcript(self, transcript: str, patient_id: str, report_type: str) -> Dict[str, Any]:
        """Advanced multi-agent processing"""
        
        # Step 1: Detect clinical context
        contexts = self.context_agent.detect_context(transcript)
        
        # Step 2: Check for emergency consultation
        emergency_keywords = self.emergency_agent.detect_emergency_keywords(transcript)
        
        # Step 3: Process based on type
        if emergency_keywords:
            # Emergency consultation expansion
            expanded_text = self.emergency_agent.expand_emergency_consult(transcript, patient_id)
            return {
                'type': 'emergency_consult',
                'original_transcript': transcript,
                'expanded_consultation': expanded_text,
                'contexts': contexts,
                'emergency_keywords': emergency_keywords,
                'processing_notes': 'Spoedconsult automatisch uitgebreid'
            }
        
        elif report_type in ['TTE', 'TEE']:
            # Echo report processing
            findings = self._extract_echo_findings(transcript)
            
            if report_type == 'TTE':
                structured_report = self.echo_agent.generate_tte_report(findings)
            else:
                structured_report = self.echo_agent.generate_tee_report(findings)
            
            return {
                'type': 'echo_report',
                'original_transcript': transcript,
                'structured_report': structured_report,
                'findings': findings,
                'contexts': contexts,
                'processing_notes': f'Professioneel {report_type} rapport gegenereerd'
            }
        
        else:
            # Regular medical processing with drug context resolution
            improved_transcript = self._improve_with_context(transcript, contexts)
            
            return {
                'type': 'regular_medical',
                'original_transcript': transcript,
                'improved_transcript': improved_transcript,
                'contexts': contexts,
                'processing_notes': 'Medische context toegepast'
            }
    
    def _extract_echo_findings(self, transcript: str) -> Dict[str, Any]:
        """Extract echo findings from transcript"""
        findings = {}
        text_lower = transcript.lower()
        
        # LVEF extraction
        lvef_match = re.search(r'lvef\s*(\d+)%?', text_lower)
        if lvef_match:
            findings['lvef'] = int(lvef_match.group(1))
        
        # Other findings extraction (simplified for demo)
        if 'hypertrofie' in text_lower:
            findings['rwbs'] = 'hypertrofie'
        
        if 'mitralis' in text_lower and 'insufficientie' in text_lower:
            findings['mv'] = 'mitralisinsufficiëntie'
        
        return findings
    
    def _improve_with_context(self, transcript: str, contexts: List[str]) -> str:
        """Improve transcript with medical context"""
        improved = transcript
        
        # Drug name resolution
        words = improved.split()
        for i, word in enumerate(words):
            if any(drug_sound in word.lower() for drug_sound in ['arixtra', 'xarelto', 'arikstra']):
                corrected = self.context_agent.resolve_drug_in_context(word, contexts)
                if corrected != word:
                    words[i] = corrected
        
        return ' '.join(words)

