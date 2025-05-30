
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from openai import OpenAI
from pinecone import Pinecone
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter
import seaborn as sns
import io
import base64

import os
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone

class PersonalAnalytics:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("companion-memory")
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of content across categories"""
        try:
            # Query all vectors
            results = self.index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=True
            )
            
            categories = Counter()
            for match in results.matches:
                category = match.metadata.get('category', 'unknown')
                categories[category] += 1
                
            return dict(categories)
        except Exception as e:
            print(f"Error getting category distribution: {e}")
            return {}
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze patterns over time from journals"""
        try:
            results = self.index.query(
                vector=self.embed("journal diary reflection thoughts"),
                top_k=500,
                include_metadata=True,
                filter={"category": {"$in": ["journal_2022", "journal_2024", "journal_2025", "personal_journal"]}}
            )
            
            temporal_data = defaultdict(list)
            mood_indicators = {
                'positive': ['happy', 'good', 'great', 'amazing', 'wonderful', 'excited', 'grateful'],
                'negative': ['sad', 'bad', 'terrible', 'awful', 'frustrated', 'angry', 'stressed'],
                'growth': ['learned', 'grew', 'improved', 'achieved', 'accomplished', 'progress']
            }
            
            for match in results.matches:
                text = match.metadata.get('text', '').lower()
                category = match.metadata.get('category', '')
                
                year = None
                if 'journal_2022' in category:
                    year = '2022'
                elif 'journal_2024' in category:
                    year = '2024'
                elif 'journal_2025' in category:
                    year = '2025'
                
                if year:
                    mood_score = 0
                    for mood, words in mood_indicators.items():
                        count = sum(1 for word in words if word in text)
                        if mood == 'positive' or mood == 'growth':
                            mood_score += count
                        elif mood == 'negative':
                            mood_score -= count
                    
                    temporal_data[year].append({
                        'mood_score': mood_score,
                        'text_snippet': text[:200],
                        'score': match.score
                    })
            
            return dict(temporal_data)
        except Exception as e:
            print(f"Error analyzing temporal patterns: {e}")
            return {}
    
    def get_personality_insights(self) -> Dict[str, Any]:
        """Extract key personality insights"""
        try:
            results = self.index.query(
                vector=self.embed("personality traits characteristics strengths weaknesses"),
                top_k=100,
                include_metadata=True,
                filter={"category": "personality"}
            )
            
            insights = []
            for match in results.matches:
                if match.score > 0.7:
                    insights.append({
                        'text': match.metadata.get('text', ''),
                        'relevance': match.score,
                        'source': match.metadata.get('source', '')
                    })
            
            return {'insights': insights}
        except Exception as e:
            print(f"Error getting personality insights: {e}")
            return {}
    
    def analyze_goal_progress(self) -> Dict[str, Any]:
        """Analyze goal-related content and progress"""
        try:
            results = self.index.query(
                vector=self.embed("goals objectives achievements progress completed"),
                top_k=200,
                include_metadata=True,
                filter={"category": {"$in": ["goals", "projects", "work_rls"]}}
            )
            
            goal_data = {
                'mentioned_goals': [],
                'achievements': [],
                'challenges': []
            }
            
            achievement_words = ['completed', 'achieved', 'accomplished', 'finished', 'success']
            challenge_words = ['difficult', 'struggle', 'challenge', 'hard', 'problem']
            
            for match in results.matches:
                text = match.metadata.get('text', '').lower()
                
                if any(word in text for word in achievement_words):
                    goal_data['achievements'].append({
                        'text': match.metadata.get('text', ''),
                        'score': match.score
                    })
                elif any(word in text for word in challenge_words):
                    goal_data['challenges'].append({
                        'text': match.metadata.get('text', ''),
                        'score': match.score
                    })
                else:
                    goal_data['mentioned_goals'].append({
                        'text': match.metadata.get('text', ''),
                        'score': match.score
                    })
            
            return goal_data
        except Exception as e:
            print(f"Error analyzing goal progress: {e}")
            return {}
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive AI-powered analysis report"""
        try:
            # Gather all data
            categories = self.get_category_distribution()
            temporal = self.analyze_temporal_patterns()
            personality = self.get_personality_insights()
            goals = self.analyze_goal_progress()
            
            # Create analysis prompt
            analysis_prompt = f"""Based on Michael's comprehensive personal data, provide a detailed analysis report covering:

            1. CONTENT OVERVIEW:
            Categories: {categories}
            
            2. TEMPORAL PATTERNS:
            Journal patterns across years: {temporal}
            
            3. PERSONALITY INSIGHTS:
            Key traits and characteristics: {personality}
            
            4. GOAL ANALYSIS:
            Goals, achievements, and challenges: {goals}
            
            Provide insights on:
            - Personal growth patterns and trends
            - Key strengths and areas for development
            - Progress on goals and projects
            - Recommendations for optimization
            - Notable patterns or changes over time
            
            Format as a structured report with clear sections and actionable insights."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating report: {e}"
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            return self.client.embeddings.create(
                model="text-embedding-3-small", input=text
            ).data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 1536
