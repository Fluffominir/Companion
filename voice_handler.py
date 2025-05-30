
import os
import json
import tempfile
from typing import Dict, Any
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class VoiceHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def speech_to_text(self, audio_file_path: str) -> str:
        """Convert speech audio to text using OpenAI Whisper"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text"
                )
            return transcript
        except Exception as e:
            logger.error(f"Speech to text error: {e}")
            return ""
    
    def text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        """Convert text to speech using OpenAI TTS"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,  # alloy, echo, fable, onyx, nova, shimmer
                input=text[:4096]  # Limit text length
            )
            return response.content
        except Exception as e:
            logger.error(f"Text to speech error: {e}")
            return b""
    
    def process_voice_message(self, audio_data: bytes) -> Dict[str, Any]:
        """Process voice input and return both text and audio response"""
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            # Convert speech to text
            text_input = self.speech_to_text(temp_audio_path)
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            if not text_input:
                return {"error": "Could not understand audio"}
            
            return {
                "transcribed_text": text_input,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return {"error": str(e)}
    
    def generate_voice_response(self, text_response: str, voice_preference: str = "alloy") -> bytes:
        """Generate voice response from text"""
        return self.text_to_speech(text_response, voice_preference)
