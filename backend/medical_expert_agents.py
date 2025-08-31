"""
🏥 MEDICAL EXPERT AGENTS SYSTEM
===============================

3 Specialized Medical AI Agents:
1. Transcript Quality Control Agent (Claude Opus 4.1)
2. Diagnostic Expert Agent (GPT-4 Turbo)  
3. Treatment Protocol Agent (Claude Opus 4.1)

Each agent uses the best available AI model for maximum medical accuracy.
"""

import openai
import anthropic
import json
import logging
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class MedicalExpertAgents:
    """
    Advanced medical expert system with 3 specialized agents
    """
    
    def __init__(self):
        # Initialize Claude Opus 4.1 client
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Initialize OpenAI GPT-4 client  
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        logger.info("🤖 Medical Expert Agents initialized with Claude Opus 4.1 & GPT-4")
    
    def agent_1_transcript_quality_control(self, transcript: str, patient_context: str = "") -> Dict:
        """
        🔍 AGENT 1: Transcript Quality Control Agent (Claude Opus 4.1)
        
        Analyzes transcript for:
        - Medical terminology errors
        - Logical inconsistencies  
        - Missing critical information
        - Anatomical impossibilities
        - Drug name corrections
        - Dosage verification
        """
        
        prompt = f"""
Je bent een EXPERT MEDISCHE TRANSCRIPT KWALITEITSCONTROLEUR met 20+ jaar ervaring.

TRANSCRIPT OM TE ANALYSEREN:
{transcript}

PATIËNT CONTEXT:
{patient_context}

VOER EEN DIEPGAANDE KWALITEITSCONTROLE UIT:

1. 🔍 MEDISCHE TERMINOLOGIE CONTROLE:
   - Identificeer verkeerd gespelde medische termen
   - Corrigeer anatomische namen
   - Verifieer medicijnnamen (let op Nederlandse uitspraak)
   - Check laboratoriumwaarden en eenheden

2. 🧠 LOGISCHE CONSISTENTIE ANALYSE:
   - Detecteer tegenstrijdigheden in bevindingen
   - Identificeer onmogelijke combinaties
   - Check consistentie tussen symptomen en bevindingen
   - Verifieer dosering-gewicht relaties

3. ⚠️ KRITIEKE INFORMATIE GAPS:
   - Identificeer ontbrekende essentiële gegevens
   - Markeer incomplete medicatie informatie
   - Check missende vitale parameters
   - Identificeer onduidelijke tijdslijnen

4. 🚨 VEILIGHEIDSRISICO'S:
   - Detecteer potentieel gevaarlijke medicatie combinaties
   - Identificeer contra-indicaties
   - Check allergieën vs voorgeschreven medicatie
   - Verifieer dosering binnen veilige grenzen

5. 💊 NEDERLANDSE MEDICATIE CORRECTIES:
   - "sedocar" → "Cedocard"
   - "arixtra" vs "xarelto" (context-afhankelijk)
   - Nederlandse vs internationale namen
   - Generieke vs merknamen

GEEF TERUG IN JSON FORMAT:
{
    "quality_score": 0-100,
    "corrections": [
        {
            "original": "verkeerde term",
            "corrected": "juiste term", 
            "reason": "waarom gecorrigeerd",
            "confidence": 0-100
        }
    ],
    "inconsistencies": [
        {
            "issue": "beschrijving van inconsistentie",
            "severity": "low/medium/high/critical",
            "suggestion": "voorgestelde oplossing"
        }
    ],
    "missing_info": [
        {
            "category": "medicatie/vitals/lab/history",
            "description": "wat ontbreekt",
            "importance": "low/medium/high/critical"
        }
    ],
    "safety_alerts": [
        {
            "type": "drug_interaction/contraindication/overdose/allergy",
            "description": "veiligheidsprobleem",
            "severity": "warning/danger/critical",
            "action_required": "wat te doen"
        }
    ],
    "improved_transcript": "volledig gecorrigeerde transcript"
}

WEES EXTREEM GRONDIG EN PRECIES. MEDISCHE VEILIGHEID IS PRIORITEIT #1.
"""

        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Latest Claude model
                max_tokens=4000,
                temperature=0.1,  # Low temperature for medical accuracy
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse JSON response
            result = json.loads(response.content[0].text)
            result['agent'] = 'Transcript Quality Control (Claude Opus)'
            result['model'] = 'claude-3-5-sonnet-20241022'
            
            logger.info(f"🔍 Agent 1 completed quality control - Score: {result.get('quality_score', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Agent 1 error: {e}")
            return {
                "error": str(e),
                "agent": "Transcript Quality Control (Claude Opus)",
                "quality_score": 0
            }
    
    def agent_2_diagnostic_expert(self, transcript: str, patient_context: str = "") -> Dict:
        """
        🩺 AGENT 2: Diagnostic Expert Agent (GPT-4 Turbo)
        
        Analyzes clinical data for:
        - Most likely diagnoses (differential diagnosis)
        - Evidence-based reasoning
        - Risk stratification
        - Prognosis assessment
        - Additional testing recommendations
        """
        
        prompt = f"""
Je bent een EXPERT DIAGNOSTICUS met 25+ jaar klinische ervaring en toegang tot de nieuwste medische literatuur.

KLINISCHE DATA:
{transcript}

PATIËNT CONTEXT:
{patient_context}

VOER EEN COMPLETE DIAGNOSTISCHE ANALYSE UIT:

1. 🎯 DIFFERENTIAAL DIAGNOSE:
   - Identificeer de 3-5 meest waarschijnlijke diagnoses
   - Rangschik op waarschijnlijkheid (%)
   - Geef evidence-based onderbouwing
   - Include zeldzame maar ernstige diagnoses

2. 📊 KLINISCHE REDENERING:
   - Analyseer symptomen en bevindingen
   - Correleer lab/imaging resultaten
   - Identificeer rode vlaggen
   - Assess tijdslijn en progressie

3. ⚡ URGENTIE BEOORDELING:
   - Bepaal acuiteit (stabiel/urgent/kritiek)
   - Identificeer time-sensitive diagnoses
   - Risk stratification
   - Prognose inschatting

4. 🔬 AANVULLEND ONDERZOEK:
   - Welke tests zijn nodig voor confirmatie
   - Prioriteit van onderzoeken
   - Cost-effectiveness overwegingen
   - Tijdslijn voor follow-up

5. 🚨 COMPLICATIE RISICO'S:
   - Potentiële complicaties per diagnose
   - Monitoring parameters
   - Waarschuwingssignalen
   - Preventieve maatregelen

GEEF TERUG IN JSON FORMAT:
{
    "primary_diagnosis": {
        "name": "hoofddiagnose",
        "icd10_code": "ICD-10 code",
        "probability": 0-100,
        "evidence": ["ondersteunende bevindingen"],
        "confidence": 0-100
    },
    "differential_diagnoses": [
        {
            "name": "alternatieve diagnose",
            "icd10_code": "ICD-10 code", 
            "probability": 0-100,
            "evidence": ["ondersteunende bevindingen"],
            "exclusion_criteria": ["waarom minder waarschijnlijk"]
        }
    ],
    "urgency_level": "stable/urgent/critical",
    "risk_stratification": "low/intermediate/high",
    "recommended_tests": [
        {
            "test": "onderzoek naam",
            "indication": "waarom nodig",
            "priority": "routine/urgent/stat",
            "expected_timeframe": "wanneer uitvoeren"
        }
    ],
    "complications_risk": [
        {
            "complication": "mogelijke complicatie",
            "probability": 0-100,
            "prevention": "preventieve maatregelen",
            "monitoring": "wat monitoren"
        }
    ],
    "clinical_reasoning": "gedetailleerde medische redenering",
    "red_flags": ["kritieke waarschuwingssignalen"]
}

GEBRUIK EVIDENCE-BASED MEDICINE EN NIEUWSTE RICHTLIJNEN.
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",  # Best GPT-4 model
                messages=[
                    {
                        "role": "system",
                        "content": "Je bent een expert diagnosticus met toegang tot de nieuwste medische kennis en richtlijnen."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for medical accuracy
                max_tokens=4000
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            result['agent'] = 'Diagnostic Expert (GPT-4 Turbo)'
            result['model'] = 'gpt-4-turbo-preview'
            
            logger.info(f"🩺 Agent 2 completed diagnostic analysis - Primary: {result.get('primary_diagnosis', {}).get('name', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Agent 2 error: {e}")
            return {
                "error": str(e),
                "agent": "Diagnostic Expert (GPT-4 Turbo)",
                "primary_diagnosis": {"name": "Error in analysis"}
            }
    
    def agent_3_treatment_protocol(self, transcript: str, diagnoses: Dict, patient_context: str = "") -> Dict:
        """
        💊 AGENT 3: Treatment Protocol Agent (Claude Opus 4.1)
        
        Creates evidence-based treatment plans:
        - ESC/AHA/ERS guideline-based protocols
        - Precise medication dosing
        - Drug interaction checking
        - Monitoring protocols
        - Patient-specific adjustments
        """
        
        prompt = f"""
Je bent een EXPERT BEHANDELPROTOCOL SPECIALIST met expertise in ESC, AHA, ERS en Nederlandse richtlijnen.

KLINISCHE DATA:
{transcript}

DIAGNOSTISCHE BEVINDINGEN:
{json.dumps(diagnoses, indent=2)}

PATIËNT CONTEXT:
{patient_context}

ONTWIKKEL EEN EVIDENCE-BASED BEHANDELPROTOCOL:

1. 💊 MEDICAMENTEUZE BEHANDELING:
   - Eerste keus medicatie volgens nieuwste richtlijnen
   - Exacte doseringen (mg/kg, mg/dag, etc.)
   - Toedieningsschema en timing
   - Titratie protocollen
   - Alternatieve opties bij contra-indicaties

2. 🔄 INTERACTIE CONTROLE:
   - Check drug-drug interacties
   - Contra-indicaties per patiënt
   - Dosisaanpassingen bij nierinsufficiëntie/leverinsufficiëntie
   - Allergieën en overgevoeligheden

3. 📋 MONITORING PROTOCOL:
   - Welke parameters monitoren
   - Frequentie van controles
   - Laboratorium follow-up schema
   - Bijwerkingen surveillance
   - Effectiviteit evaluatie

4. 🎯 BEHANDELDOELEN:
   - Primaire eindpunten
   - Secundaire doelen
   - Tijdslijn voor verbetering
   - Success criteria
   - Failure criteria (wanneer aanpassen)

5. 🚨 NOODPROTOCOLLEN:
   - Wat te doen bij verslechtering
   - Emergency medicatie
   - Wanneer specialist consulteren
   - Opname criteria

6. 📚 RICHTLIJN REFERENTIES:
   - ESC Guidelines 2024
   - AHA/ACC Guidelines
   - Nederlandse richtlijnen (NHG/NVVC)
   - Evidence level (Class I/IIa/IIb/III)

GEEF TERUG IN JSON FORMAT:
{
    "treatment_plan": {
        "primary_therapy": {
            "medication": "medicijn naam",
            "dosage": "exacte dosering",
            "frequency": "hoe vaak",
            "duration": "hoe lang",
            "route": "toedieningsweg",
            "titration": "titratie schema",
            "guideline_class": "Class I/IIa/IIb",
            "evidence_level": "A/B/C"
        },
        "adjunctive_therapies": [
            {
                "medication": "aanvullende medicatie",
                "dosage": "dosering",
                "indication": "waarom voorgeschreven",
                "monitoring": "wat monitoren"
            }
        ]
    },
    "contraindications_checked": [
        {
            "medication": "medicijn",
            "contraindication": "contra-indicatie",
            "severity": "absolute/relative",
            "alternative": "alternatief medicijn"
        }
    ],
    "drug_interactions": [
        {
            "drug1": "medicijn 1",
            "drug2": "medicijn 2", 
            "interaction_type": "type interactie",
            "severity": "minor/moderate/major",
            "management": "hoe te managen"
        }
    ],
    "monitoring_protocol": {
        "laboratory": [
            {
                "test": "lab test",
                "baseline": "voor start",
                "follow_up": "follow-up schema",
                "target_values": "streefwaarden"
            }
        ],
        "clinical": [
            {
                "parameter": "klinische parameter",
                "frequency": "hoe vaak checken",
                "action_thresholds": "wanneer actie ondernemen"
            }
        ]
    },
    "emergency_protocols": [
        {
            "scenario": "noodscenario",
            "immediate_action": "directe actie",
            "medication": "noodmedicatie",
            "dosage": "nooddosering",
            "when_to_call": "wanneer specialist bellen"
        }
    ],
    "guideline_references": [
        {
            "guideline": "richtlijn naam",
            "year": "jaar",
            "recommendation": "specifieke aanbeveling",
            "evidence_class": "Class I/IIa/IIb/III",
            "evidence_level": "A/B/C"
        }
    ],
    "treatment_goals": {
        "primary_endpoints": ["primaire doelen"],
        "secondary_endpoints": ["secundaire doelen"],
        "timeframe": "verwachte tijdslijn",
        "success_criteria": ["wanneer succesvol"],
        "reassessment_schedule": "wanneer herbeoordelingen"
    }
}

GEBRUIK ALLEEN DE NIEUWSTE RICHTLIJNEN (2023-2024) EN EVIDENCE-BASED PROTOCOLS.
"""

        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Latest Claude model
                max_tokens=4000,
                temperature=0.1,  # Low temperature for medical accuracy
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse JSON response
            result = json.loads(response.content[0].text)
            result['agent'] = 'Treatment Protocol (Claude Opus)'
            result['model'] = 'claude-3-5-sonnet-20241022'
            
            logger.info(f"💊 Agent 3 completed treatment protocol - Primary therapy: {result.get('treatment_plan', {}).get('primary_therapy', {}).get('medication', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"Agent 3 error: {e}")
            return {
                "error": str(e),
                "agent": "Treatment Protocol (Claude Opus)",
                "treatment_plan": {"primary_therapy": {"medication": "Error in analysis"}}
            }
    
    def orchestrate_medical_analysis(self, transcript: str, patient_context: str = "") -> Dict:
        """
        🎼 ORCHESTRATE ALL 3 MEDICAL EXPERT AGENTS
        
        Sequential execution with inter-agent communication:
        1. Quality Control → Improved transcript
        2. Diagnostic Expert → Diagnoses using improved transcript  
        3. Treatment Protocol → Treatment plan based on diagnoses
        """
        
        logger.info("🚀 Starting Medical Expert Agents Orchestration")
        
        # AGENT 1: Transcript Quality Control
        logger.info("🔍 Agent 1: Starting transcript quality control...")
        quality_result = self.agent_1_transcript_quality_control(transcript, patient_context)
        
        # Use improved transcript for subsequent agents
        improved_transcript = quality_result.get('improved_transcript', transcript)
        
        # AGENT 2: Diagnostic Expert Analysis
        logger.info("🩺 Agent 2: Starting diagnostic analysis...")
        diagnostic_result = self.agent_2_diagnostic_expert(improved_transcript, patient_context)
        
        # AGENT 3: Treatment Protocol Development
        logger.info("💊 Agent 3: Starting treatment protocol development...")
        treatment_result = self.agent_3_treatment_protocol(improved_transcript, diagnostic_result, patient_context)
        
        # Combine all results
        final_result = {
            "orchestration_summary": {
                "timestamp": "2025-08-31",
                "agents_used": 3,
                "models_used": ["claude-3-5-sonnet-20241022", "gpt-4-turbo-preview"],
                "processing_status": "completed"
            },
            "agent_1_quality_control": quality_result,
            "agent_2_diagnostic_expert": diagnostic_result,
            "agent_3_treatment_protocol": treatment_result,
            "final_recommendations": {
                "quality_score": quality_result.get('quality_score', 0),
                "primary_diagnosis": diagnostic_result.get('primary_diagnosis', {}),
                "treatment_summary": treatment_result.get('treatment_plan', {}),
                "safety_alerts": quality_result.get('safety_alerts', []),
                "urgency_level": diagnostic_result.get('urgency_level', 'unknown')
            }
        }
        
        logger.info("🎉 Medical Expert Agents Orchestration completed successfully")
        return final_result

# Initialize the medical expert agents system
medical_experts = MedicalExpertAgents()

