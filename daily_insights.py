
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from openai import OpenAI
from pinecone import Pinecone
from integrations import IntegrationsManager
import logging

logger = logging.getLogger(__name__)

class DailyInsightsManager:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("companion-memory")
        self.integrations = IntegrationsManager()
    
    def generate_daily_briefing(self) -> Dict[str, Any]:
        """Generate comprehensive daily briefing"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Get goal reminders
            goals = self.get_goal_reminders()
            
            # Get integration insights
            integration_insights = self.integrations.generate_daily_insights()
            
            # Get personal patterns
            personal_patterns = self.analyze_personal_patterns()
            
            # Get calendar events (if available)
            upcoming_events = self.get_upcoming_events()
            
            # Generate AI briefing
            briefing_prompt = f"""Generate a personalized daily briefing for Michael based on:

Goals to focus on: {goals}
Today's insights: {integration_insights}
Personal patterns: {personal_patterns}
Upcoming events: {upcoming_events}

Create a motivating, actionable daily briefing that includes:
1. Key priorities for today
2. Goal progress reminders
3. Insights from his data patterns
4. Suggestions for optimization
5. Encouraging personal note

Keep it concise but meaningful, like a trusted personal assistant."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": briefing_prompt}],
                temperature=0.7,
                max_tokens=600
            )
            
            return {
                "date": today,
                "briefing": response.choices[0].message.content,
                "goals": goals,
                "insights": integration_insights,
                "patterns": personal_patterns,
                "events": upcoming_events
            }
            
        except Exception as e:
            logger.error(f"Error generating daily briefing: {e}")
            return {"error": str(e)}
    
    def get_goal_reminders(self) -> List[Dict[str, Any]]:
        """Get relevant goal reminders based on stored data"""
        try:
            results = self.index.query(
                vector=self.embed("goals objectives targets priorities today focus"),
                top_k=10,
                include_metadata=True,
                filter={"category": {"$in": ["goals", "projects", "ideas"]}}
            )
            
            goal_reminders = []
            for match in results.matches:
                if match.score > 0.75:
                    goal_reminders.append({
                        "text": match.metadata.get("text", ""),
                        "category": match.metadata.get("category", ""),
                        "relevance": match.score,
                        "source": match.metadata.get("source", "")
                    })
            
            return goal_reminders[:5]
        except Exception as e:
            logger.error(f"Error getting goal reminders: {e}")
            return []
    
    def analyze_personal_patterns(self) -> Dict[str, Any]:
        """Analyze personal patterns from journal and behavior data"""
        try:
            # Get recent journal entries
            journal_results = self.index.query(
                vector=self.embed("today feeling energy mood productive thoughts"),
                top_k=20,
                include_metadata=True,
                filter={"category": {"$in": ["personal_journal", "journal_2025"]}}
            )
            
            patterns = {
                "energy_trends": [],
                "productivity_insights": [],
                "mood_patterns": [],
                "recent_themes": []
            }
            
            # Analyze recent entries for patterns
            for match in journal_results.matches:
                text = match.metadata.get("text", "").lower()
                
                # Look for energy indicators
                if any(word in text for word in ["tired", "exhausted", "low energy"]):
                    patterns["energy_trends"].append("low")
                elif any(word in text for word in ["energized", "motivated", "high energy"]):
                    patterns["energy_trends"].append("high")
                
                # Look for productivity indicators
                if any(word in text for word in ["productive", "accomplished", "completed"]):
                    patterns["productivity_insights"].append("high")
                elif any(word in text for word in ["stuck", "distracted", "procrastinated"]):
                    patterns["productivity_insights"].append("low")
                
                # Extract themes
                patterns["recent_themes"].append(text[:100])
            
            return patterns
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {}
    
    def get_upcoming_events(self) -> List[Dict[str, Any]]:
        """Get upcoming events (placeholder for calendar integration)"""
        # This would integrate with the existing Google Calendar functionality
        return []
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            return self.client.embeddings.create(
                model="text-embedding-3-small", input=text
            ).data[0].embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * 1536
    
    def store_daily_reflection(self, reflection_text: str, mood_score: int = 5, energy_level: int = 5):
        """Store daily reflection in memory"""
        try:
            timestamp = datetime.now().isoformat()
            vector_id = f"daily_reflection_{datetime.now().strftime('%Y%m%d')}"
            
            metadata = {
                "text": reflection_text,
                "source": "daily_reflection",
                "category": "personal_journal",
                "mood_score": mood_score,
                "energy_level": energy_level,
                "timestamp": timestamp,
                "file_type": "reflection"
            }
            
            embedding = self.embed(reflection_text)
            self.index.upsert([(vector_id, embedding, metadata)])
            
            return {"success": True, "stored_at": timestamp}
        except Exception as e:
            logger.error(f"Error storing reflection: {e}")
            return {"error": str(e)}
