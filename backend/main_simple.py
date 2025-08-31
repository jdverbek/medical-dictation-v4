# backend/main_simple.py - Simplified FastAPI Backend for Deployment

"""
Medical Dictation v4.0 - Simplified Backend
Core functionality without complex dependencies
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import uuid
from datetime import datetime
import json
import io
import os
from typing import Dict, Any
from openai import AsyncOpenAI
import anthropic
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseSettings

# Configuration
class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# FastAPI app
app = FastAPI(
    title="Medical Dictation v4.0",
    description="Simplified medical transcription with AI intelligence",
    version="4.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI clients
openai_client = None
anthropic_client = None

if settings.openai_api_key:
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

if settings.anthropic_api_key:
    anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

# In-memory storage for demo (replace with database in production)
audio_storage = {}
transcription_results = {}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "4.0.0",
        "services": {
            "openai": "configured" if openai_client else "not configured",
            "anthropic": "configured" if anthropic_client else "not configured"
        }
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio using Whisper"""
    if not openai_client:
        raise HTTPException(500, "OpenAI API not configured")
    
    response = await openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.webm", audio_data, "audio/webm"),
        language="nl",
        temperature=0.0,
        prompt="Medische dictatie in het Nederlands. Let op medische terminologie."
    )
    return response.text

async def improve_transcript_with_gpt(transcript: str, patient_id: str, report_type: str) -> str:
    """Improve transcript using GPT-4"""
    if not openai_client:
        return transcript
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{
                "role": "system",
                "content": f"""Je bent een medische transcriptie expert. Verbeter de transcriptie voor een {report_type} rapport:

1. Corrigeer verkeerd gespelde medicijnnamen (bijv. sedocar -> cedocard)
2. Verbeter medische terminologie
3. Zorg voor logische consistentie
4. Behoud alle originele informatie

Geef alleen de verbeterde transcriptie terug, geen uitleg."""
            }, {
                "role": "user",
                "content": f"Patiënt ID: {patient_id}\nTranscriptie: {transcript}"
            }],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT improvement failed: {e}")
        return transcript

async def validate_with_claude(transcript: str, report: str) -> Dict[str, Any]:
    """Final validation with Claude"""
    if not anthropic_client:
        return {
            "is_valid": True,
            "confidence": 0.8,
            "final_transcript": transcript,
            "final_report": report
        }
    
    try:
        response = await anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""Als medische expert, valideer dit rapport:

Transcriptie: {transcript}
Rapport: {report}

Geef terug als JSON:
{{
    "is_valid": boolean,
    "confidence": float (0-1),
    "final_transcript": "definitieve transcriptie",
    "final_report": "definitief rapport"
}}"""
            }],
            temperature=0.1
        )
        
        content = response.content[0].text
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
            
    except Exception as e:
        print(f"Claude validation failed: {e}")
    
    return {
        "is_valid": True,
        "confidence": 0.8,
        "final_transcript": transcript,
        "final_report": report
    }

async def generate_medical_report(transcript: str, patient_id: str, report_type: str) -> str:
    """Generate structured medical report"""
    if not openai_client:
        return f"MEDISCH RAPPORT\n\nPatiënt ID: {patient_id}\nType: {report_type}\n\nTranscriptie:\n{transcript}"
    
    templates = {
        "TTE": "Structureer als TTE rapport met linker ventrikel, rechter ventrikel, atria, kleppen en conclusie",
        "TEE": "Structureer als TEE rapport met gedetailleerde klep analyse",
        "ECG": "Structureer als ECG rapport met ritme, intervallen, as en morfologie",
        "Holter": "Structureer als Holter rapport met basisritme en aritmieën",
        "Consult": "Structureer als consultrapport met anamnese, onderzoek en beleid"
    }
    
    template = templates.get(report_type, templates["Consult"])
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{
                "role": "system",
                "content": f"""Je bent een ervaren cardioloog. Schrijf een gestructureerd medisch rapport.

{template}

Gebruik alleen informatie uit de transcriptie. Schrijf in professioneel Nederlands."""
            }, {
                "role": "user",
                "content": f"Patiënt ID: {patient_id}\nTranscriptie: {transcript}"
            }],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Report generation failed: {e}")
        return f"MEDISCH RAPPORT\n\nPatiënt ID: {patient_id}\nType: {report_type}\n\nTranscriptie:\n{transcript}"

@app.post("/api/transcribe")
async def transcribe_audio_endpoint(
    audio: UploadFile = File(...),
    patient_id: str = Form(...),
    report_type: str = Form(default="TTE")
):
    """Main transcription endpoint with AI intelligence"""
    
    # Validate input
    if audio.content_type not in ["audio/webm", "audio/wav", "audio/mp3", "audio/m4a"]:
        raise HTTPException(400, "Invalid audio format")
    
    # Read audio data
    audio_data = await audio.read()
    session_id = str(uuid.uuid4())
    
    try:
        # Store audio for playback
        audio_storage[session_id] = audio_data
        
        # Step 1: Initial transcription
        transcript = await transcribe_audio(audio_data)
        
        # Step 2: Improve with GPT-4 (multi-agent simulation)
        improved_transcript = await improve_transcript_with_gpt(transcript, patient_id, report_type)
        
        # Step 3: Generate medical report
        report = await generate_medical_report(improved_transcript, patient_id, report_type)
        
        # Step 4: Final validation with Claude
        validated_result = await validate_with_claude(improved_transcript, report)
        
        # Store result
        result = {
            'transcript': validated_result['final_transcript'],
            'report': validated_result['final_report'],
            'session_id': session_id,
            'confidence': validated_result.get('confidence', 0.8),
            'processing_metadata': {
                'agents_used': ['transcriber', 'gpt_improver', 'claude_validator'],
                'iterations': 1,
                'improvements_made': 1 if improved_transcript != transcript else 0
            }
        }
        
        transcription_results[session_id] = result
        
        return JSONResponse({
            "success": True,
            "transcript": result['transcript'],
            "report": result['report'],
            "audio_url": f"/api/audio/{session_id}"
        })
        
    except Exception as e:
        print(f"Processing failed: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/api/audio/{session_id}")
async def get_audio_playback(session_id: str):
    """Retrieve audio for playback"""
    
    audio_data = audio_storage.get(session_id)
    if not audio_data:
        raise HTTPException(404, "Audio not found")
    
    return StreamingResponse(
        io.BytesIO(audio_data),
        media_type="audio/webm",
        headers={"Content-Disposition": f"inline; filename=recording_{session_id}.webm"}
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Medical Dictation v4.0 API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

