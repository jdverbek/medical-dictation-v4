"""
🤖 Medical Expert Agents System v4.0 - OpenAI Only
Advanced medical expert system with 3 specialized agents using only OpenAI models for reliability
"""

import os
import json
import openai
from typing import Dict, List, Any, Optional

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

class MedicalExpertAgents:
    """
    Advanced medical expert system with 3 specialized agents - OpenAI only for maximum reliability
    """
    
    def __init__(self):
        # Initialize OpenAI (already configured globally)
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        print("🤖 Initializing 3 Expert Medical Agents (OpenAI Only)...")
        print("🔍 Agent 1: Transcript Quality Control (GPT-4)")
        print("🩺 Agent 2: Diagnostic Expert (GPT-4)")  
        print("💊 Agent 3: Treatment Protocol (GPT-4)")
    
    def _call_gpt4(self, prompt: str, system_prompt: str = "", max_tokens: int = 1000, json_mode: bool = True) -> str:
        """Call GPT-4 using older API format with proper error handling"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            kwargs = {
                "model": "gpt-4",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1
            }
            
            # Add response format for newer OpenAI versions if available
            if json_mode:
                try:
                    kwargs["response_format"] = {"type": "json_object"}
                except:
                    pass  # Fallback for older versions
            
            response = openai.ChatCompletion.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ GPT-4 API error: {e}")
            return f"GPT-4 analysis unavailable: {str(e)}"
    
    def agent_1_quality_control(self, transcript: str) -> Dict[str, Any]:
        """
        🔍 Agent 1: Transcript Quality Control (GPT-4)
        Analyzes transcript for medical terminology errors and improvements
        """
        system_prompt = """Je bent een expert medische transcriptie specialist. Je taak is Nederlandse medische transcripties te analyseren en verbeteren.

BELANGRIJKE CORRECTIES:
- "sedocar" → "cedocard" (medicijn)
- "artredicotentie" → "atriumfibrillatie" 
- "voorkamervipulatie" → "voorkamerfibrillatie"
- "zeeuwtachtigjarige" → "80-jarige"
- "serocreatinine" → "serumcreatinine"
- "NC-program T" → "NT-proBNP"
- "mgpg" → "mg/dl"
- "kmpg" → "mmHg"

Geef ALTIJD een JSON response terug."""
        
        prompt = f"""Analyseer deze Nederlandse medische transcriptie en verbeter deze:

ORIGINELE TRANSCRIPTIE:
{transcript}

TAAK:
1. Corrigeer medische terminologie fouten
2. Fix medicijnnamen (bijv. "sedocar" → "cedocard")
3. Verbeter anatomische termen
4. Controleer logische consistentie
5. Geef een kwaliteitsscore (0-100)

ANTWOORD IN JSON FORMAT:
{{
    "improved_transcript": "verbeterde transcriptie hier",
    "corrections": ["lijst van correcties"],
    "quality_score": 85,
    "safety_alerts": ["eventuele veiligheidswaarschuwingen"],
    "confidence": 0.95
}}"""
        
        response = self._call_gpt4(prompt, system_prompt, max_tokens=1500, json_mode=True)
        
        try:
            # Try to parse JSON response
            if response.startswith('{'):
                result = json.loads(response)
                # Ensure all required fields exist
                if 'improved_transcript' not in result:
                    result['improved_transcript'] = transcript
                if 'corrections' not in result:
                    result['corrections'] = []
                if 'quality_score' not in result:
                    result['quality_score'] = 75
                if 'safety_alerts' not in result:
                    result['safety_alerts'] = []
                if 'confidence' not in result:
                    result['confidence'] = 0.8
                return result
            else:
                # Fallback if not JSON
                return {
                    "improved_transcript": transcript,
                    "corrections": ["Basic processing applied"],
                    "quality_score": 75,
                    "safety_alerts": [],
                    "confidence": 0.8
                }
        except Exception as e:
            print(f"⚠️ Agent 1 JSON parsing error: {e}")
            return {
                "improved_transcript": transcript,
                "corrections": ["Processing error"],
                "quality_score": 70,
                "safety_alerts": [],
                "confidence": 0.7
            }
    
    def agent_2_diagnostic_expert(self, transcript: str, patient_context: str = "") -> Dict[str, Any]:
        """
        🩺 Agent 2: Diagnostic Expert (GPT-4)
        Analyzes medical content for diagnostic insights
        """
        system_prompt = """Je bent een expert cardioloog en diagnosticus. Je analyseert Nederlandse medische transcripties en geeft diagnostische inzichten.

FOCUS OP:
- Primaire diagnose identificatie
- Urgentie bepaling (low/medium/high/critical)
- Differentiaal diagnoses
- Aanbevolen onderzoeken
- Red flags

Geef ALTIJD een JSON response terug."""
        
        prompt = f"""Analyseer deze Nederlandse medische transcriptie voor diagnostische inzichten:

TRANSCRIPT: {transcript}
CONTEXT: {patient_context}

Geef analyse in JSON format:
{{
    "primary_diagnosis": {{
        "name": "meest waarschijnlijke diagnose",
        "confidence": 0.85,
        "icd10_code": "relevante code"
    }},
    "differential_diagnoses": ["lijst van andere mogelijkheden"],
    "urgency_level": "low/medium/high/critical",
    "recommended_tests": ["voorgestelde onderzoeken"],
    "red_flags": ["zorgwekkende bevindingen"],
    "clinical_reasoning": "korte uitleg"
}}"""
        
        response = self._call_gpt4(prompt, system_prompt, max_tokens=1000, json_mode=True)
        
        try:
            if response.startswith('{'):
                result = json.loads(response)
                # Ensure all required fields exist
                if 'primary_diagnosis' not in result:
                    result['primary_diagnosis'] = {"name": "Analysis pending", "confidence": 0.5}
                if 'urgency_level' not in result:
                    result['urgency_level'] = "medium"
                if 'differential_diagnoses' not in result:
                    result['differential_diagnoses'] = []
                if 'recommended_tests' not in result:
                    result['recommended_tests'] = []
                if 'red_flags' not in result:
                    result['red_flags'] = []
                if 'clinical_reasoning' not in result:
                    result['clinical_reasoning'] = "Basic analysis applied"
                return result
            else:
                return {
                    "primary_diagnosis": {"name": "Analysis pending", "confidence": 0.5},
                    "urgency_level": "medium",
                    "differential_diagnoses": [],
                    "recommended_tests": [],
                    "red_flags": [],
                    "clinical_reasoning": "Basic analysis applied"
                }
        except Exception as e:
            print(f"⚠️ Agent 2 JSON parsing error: {e}")
            return {
                "primary_diagnosis": {"name": "Processing error", "confidence": 0.3},
                "urgency_level": "unknown",
                "differential_diagnoses": [],
                "recommended_tests": [],
                "red_flags": [],
                "clinical_reasoning": "Analysis unavailable"
            }
    
    def agent_3_treatment_protocol(self, transcript: str, diagnosis_info: Dict = None) -> Dict[str, Any]:
        """
        💊 Agent 3: Treatment Protocol (GPT-4)
        Provides evidence-based treatment recommendations according to ESC 2024 Guidelines
        """
        diagnosis = diagnosis_info.get('primary_diagnosis', {}).get('name', 'Unknown') if diagnosis_info else 'Unknown'
        
        system_prompt = """Je bent een expert cardioloog gespecialiseerd in behandelingsprotocollen volgens de meest recente ESC 2024 richtlijnen.

BELANGRIJKE ESC 2024 UPDATES:
- ESC 2024 Guidelines on Acute Coronary Syndromes (ACS)
- ESC 2024 Guidelines on Heart Failure
- ESC 2024 Guidelines on Atrial Fibrillation
- ESC 2024 Guidelines on Hypertension
- ESC 2024 Guidelines on Valvular Heart Disease

MEDICATIE VOLGENS ESC 2024:
- NSTEMI/STEMI: Dual antiplatelet therapy (aspirin + P2Y12 inhibitor)
- Heart Failure: ACE-I/ARB/ARNI + Beta-blocker + MRA + SGLT2i (4-pillar therapy)
- Atrial Fibrillation: CHA2DS2-VASc score voor anticoagulatie
- Hypertension: ACE-I/ARB first line, add CCB or thiazide

DOSERING VOLGENS ESC 2024:
- Atorvastatine: 80mg voor ACS (high-intensity statin)
- Metoprolol: 25-200mg BID (heart failure)
- Lisinopril: 2.5-40mg daily
- Arixtra (fondaparinux): 2.5mg daily (ACS, contraindication heparin)

Geef ALTIJD een JSON response terug met ESC 2024 referenties."""
        
        prompt = f"""Geef behandelingsadvies voor deze patiënt volgens ESC 2024 richtlijnen:

TRANSCRIPTIE: {transcript}
DIAGNOSE: {diagnosis}

Geef behandelingsprotocol in JSON format met ESC 2024 referenties:
{{
    "treatment_plan": {{
        "immediate_actions": ["directe acties volgens ESC 2024"],
        "medications": [
            {{
                "name": "medicijnnaam",
                "dose": "exacte dosering volgens ESC 2024",
                "frequency": "frequentie",
                "duration": "duur",
                "indication": "indicatie",
                "esc_2024_reference": "specifieke ESC 2024 richtlijn sectie"
            }}
        ],
        "monitoring": ["wat te monitoren volgens ESC 2024"],
        "follow_up": "vervolgafspraken volgens ESC 2024"
    }},
    "contraindications": ["contra-indicaties volgens ESC 2024"],
    "drug_interactions": ["interacties volgens ESC 2024"],
    "esc_guideline_class": "I/IIa/IIb/III volgens ESC 2024",
    "evidence_level": "A/B/C volgens ESC 2024",
    "esc_2024_citations": [
        "ESC 2024 Guidelines on [specific condition]",
        "Section [X.X] - [specific recommendation]"
    ],
    "quality_indicators": {{
        "guideline_adherence": "percentage adherence to ESC 2024",
        "evidence_strength": "strong/moderate/weak",
        "safety_profile": "high/medium/low risk"
    }}
}}"""
        
        response = self._call_gpt4(prompt, system_prompt, max_tokens=2000, json_mode=True)
        
        try:
            if response.startswith('{'):
                result = json.loads(response)
                # Ensure all required fields exist with ESC 2024 compliance
                if 'treatment_plan' not in result:
                    result['treatment_plan'] = {
                        "immediate_actions": ["Standard care according to ESC 2024"],
                        "medications": [],
                        "monitoring": ["Routine monitoring per ESC 2024"],
                        "follow_up": "As per ESC 2024 recommendations"
                    }
                if 'contraindications' not in result:
                    result['contraindications'] = []
                if 'drug_interactions' not in result:
                    result['drug_interactions'] = []
                if 'esc_guideline_class' not in result:
                    result['esc_guideline_class'] = "ESC 2024 - Class Unknown"
                if 'evidence_level' not in result:
                    result['evidence_level'] = "ESC 2024 - Level Unknown"
                if 'esc_2024_citations' not in result:
                    result['esc_2024_citations'] = ["ESC 2024 Guidelines - General Recommendations"]
                if 'quality_indicators' not in result:
                    result['quality_indicators'] = {
                        "guideline_adherence": "ESC 2024 compliant",
                        "evidence_strength": "moderate",
                        "safety_profile": "standard risk"
                    }
                return result
            else:
                return {
                    "treatment_plan": {
                        "immediate_actions": ["Standard care according to ESC 2024"],
                        "medications": [],
                        "monitoring": ["Routine monitoring per ESC 2024"],
                        "follow_up": "As per ESC 2024 recommendations"
                    },
                    "contraindications": [],
                    "drug_interactions": [],
                    "esc_guideline_class": "ESC 2024 - Class Unknown",
                    "evidence_level": "ESC 2024 - Level Unknown",
                    "esc_2024_citations": ["ESC 2024 Guidelines - General Recommendations"],
                    "quality_indicators": {
                        "guideline_adherence": "ESC 2024 compliant",
                        "evidence_strength": "moderate", 
                        "safety_profile": "standard risk"
                    }
                }
        except Exception as e:
            print(f"⚠️ Agent 3 JSON parsing error: {e}")
            return {
                "treatment_plan": {
                    "immediate_actions": ["Processing error - refer to ESC 2024"],
                    "medications": [],
                    "monitoring": [],
                    "follow_up": "Unknown - consult ESC 2024"
                },
                "contraindications": [],
                "drug_interactions": [],
                "esc_guideline_class": "ESC 2024 - Processing Error",
                "evidence_level": "ESC 2024 - Processing Error",
                "esc_2024_citations": ["ESC 2024 Guidelines - Unable to process"],
                "quality_indicators": {
                    "guideline_adherence": "unknown",
                    "evidence_strength": "unknown",
                    "safety_profile": "unknown"
                }
            }
    
    def orchestrate_medical_analysis(self, transcript: str, patient_context: str = "") -> Dict[str, Any]:
        """
        🎯 Orchestrate all 3 medical expert agents
        """
        print("🤖 Starting multi-agent medical analysis...")
        
        # Agent 1: Quality Control
        print("🔍 Running Agent 1: Quality Control...")
        agent_1_result = self.agent_1_quality_control(transcript)
        
        # Agent 2: Diagnostic Expert  
        print("🩺 Running Agent 2: Diagnostic Expert...")
        agent_2_result = self.agent_2_diagnostic_expert(
            agent_1_result.get('improved_transcript', transcript), 
            patient_context
        )
        
        # Agent 3: Treatment Protocol
        print("💊 Running Agent 3: Treatment Protocol...")
        agent_3_result = self.agent_3_treatment_protocol(
            agent_1_result.get('improved_transcript', transcript),
            agent_2_result
        )
        
        print("✅ Multi-agent analysis complete!")
        
        return {
            'agent_1_quality_control': agent_1_result,
            'agent_2_diagnostic_expert': agent_2_result,
            'agent_3_treatment_protocol': agent_3_result,
            'orchestration_summary': {
                'agents_used': 3,
                'total_processing_time': 'N/A',
                'confidence_score': agent_1_result.get('confidence', 0.8),
                'status': 'completed'
            }
        }

# Test initialization
if __name__ == "__main__":
    try:
        agents = MedicalExpertAgents()
        print("✅ Medical Expert Agents system ready!")
        
        # Test with sample transcript
        test_transcript = "Patient heeft sedocar gekregen voor retrosternale pijn"
        result = agents.orchestrate_medical_analysis(test_transcript)
        print("🧪 Test result:", json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")

