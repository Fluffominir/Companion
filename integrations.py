
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class IntegrationsManager:
    def __init__(self):
        self.spotify_token = os.getenv("SPOTIFY_ACCESS_TOKEN")
        self.hue_bridge_ip = os.getenv("HUE_BRIDGE_IP")
        self.hue_username = os.getenv("HUE_USERNAME")
        self.nas_host = os.getenv("NAS_HOST")
        self.nas_username = os.getenv("NAS_USERNAME")
        self.nas_password = os.getenv("NAS_PASSWORD")
        
    # Apple Health Integration
    def parse_apple_health_export(self, export_path: str) -> Dict[str, Any]:
        """Parse Apple Health XML export data"""
        try:
            if not os.path.exists(export_path):
                return {"error": "Apple Health export file not found"}
            
            tree = ET.parse(export_path)
            root = tree.getroot()
            
            health_data = {
                "workouts": [],
                "heart_rate": [],
                "steps": [],
                "sleep": [],
                "weight": [],
                "activity": []
            }
            
            # Parse health records
            for record in root.findall('.//Record'):
                record_type = record.get('type', '')
                value = record.get('value', '')
                start_date = record.get('startDate', '')
                end_date = record.get('endDate', '')
                
                if 'StepCount' in record_type:
                    health_data["steps"].append({
                        "value": int(float(value)),
                        "date": start_date,
                        "source": record.get('sourceName', '')
                    })
                elif 'HeartRate' in record_type:
                    health_data["heart_rate"].append({
                        "value": float(value),
                        "date": start_date,
                        "unit": record.get('unit', 'bpm')
                    })
                elif 'BodyMass' in record_type:
                    health_data["weight"].append({
                        "value": float(value),
                        "date": start_date,
                        "unit": record.get('unit', 'kg')
                    })
                elif 'SleepAnalysis' in record_type:
                    health_data["sleep"].append({
                        "start": start_date,
                        "end": end_date,
                        "value": value
                    })
            
            # Parse workouts
            for workout in root.findall('.//Workout'):
                health_data["workouts"].append({
                    "type": workout.get('workoutActivityType', ''),
                    "duration": workout.get('duration', ''),
                    "start": workout.get('startDate', ''),
                    "end": workout.get('endDate', ''),
                    "calories": workout.get('totalEnergyBurned', ''),
                    "distance": workout.get('totalDistance', '')
                })
            
            return health_data
        except Exception as e:
            logger.error(f"Error parsing Apple Health data: {e}")
            return {"error": str(e)}
    
    # Spotify Integration
    def get_spotify_data(self) -> Dict[str, Any]:
        """Get recent Spotify listening data"""
        if not self.spotify_token:
            return {"error": "Spotify token not configured"}
        
        try:
            headers = {"Authorization": f"Bearer {self.spotify_token}"}
            
            # Get recently played tracks
            recent_url = "https://api.spotify.com/v1/me/player/recently-played?limit=50"
            recent_response = requests.get(recent_url, headers=headers)
            
            # Get current user profile
            profile_url = "https://api.spotify.com/v1/me"
            profile_response = requests.get(profile_url, headers=headers)
            
            # Get top tracks and artists
            top_tracks_url = "https://api.spotify.com/v1/me/top/tracks?time_range=short_term&limit=20"
            top_tracks_response = requests.get(top_tracks_url, headers=headers)
            
            top_artists_url = "https://api.spotify.com/v1/me/top/artists?time_range=short_term&limit=20"
            top_artists_response = requests.get(top_artists_url, headers=headers)
            
            return {
                "recent_tracks": recent_response.json() if recent_response.status_code == 200 else {},
                "profile": profile_response.json() if profile_response.status_code == 200 else {},
                "top_tracks": top_tracks_response.json() if top_tracks_response.status_code == 200 else {},
                "top_artists": top_artists_response.json() if top_artists_response.status_code == 200 else {}
            }
        except Exception as e:
            logger.error(f"Spotify API error: {e}")
            return {"error": str(e)}
    
    # Philips Hue Integration
    def get_hue_data(self) -> Dict[str, Any]:
        """Get Philips Hue lights status and usage patterns"""
        if not self.hue_bridge_ip or not self.hue_username:
            return {"error": "Hue bridge not configured"}
        
        try:
            base_url = f"http://{self.hue_bridge_ip}/api/{self.hue_username}"
            
            # Get lights status
            lights_response = requests.get(f"{base_url}/lights")
            
            # Get sensors (including motion and daylight)
            sensors_response = requests.get(f"{base_url}/sensors")
            
            # Get groups/rooms
            groups_response = requests.get(f"{base_url}/groups")
            
            return {
                "lights": lights_response.json() if lights_response.status_code == 200 else {},
                "sensors": sensors_response.json() if sensors_response.status_code == 200 else {},
                "groups": groups_response.json() if groups_response.status_code == 200 else {}
            }
        except Exception as e:
            logger.error(f"Hue API error: {e}")
            return {"error": str(e)}
    
    # NAS File Access
    def scan_nas_files(self, extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Scan NAS for files (requires NAS to be mounted or accessible via SMB/FTP)"""
        if not self.nas_host:
            return []
        
        if extensions is None:
            extensions = ['.pdf', '.txt', '.md', '.docx', '.jpg', '.png']
        
        try:
            files_found = []
            
            # This is a simplified approach - in production you'd want proper SMB/FTP client
            # For now, assuming NAS is mounted or accessible via network path
            nas_paths = [
                f"/mnt/{self.nas_host}",  # Linux mount point
                f"///{self.nas_host}",     # SMB path
                f"/Volumes/{self.nas_host}"  # macOS mount point
            ]
            
            for nas_path in nas_paths:
                if os.path.exists(nas_path):
                    for ext in extensions:
                        pattern = f"{nas_path}/**/*{ext}"
                        for file_path in glob.glob(pattern, recursive=True):
                            stat = os.stat(file_path)
                            files_found.append({
                                "path": file_path,
                                "name": os.path.basename(file_path),
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "extension": ext
                            })
                    break
            
            return files_found[:100]  # Limit results
        except Exception as e:
            logger.error(f"NAS scan error: {e}")
            return []
    
    def process_nas_file(self, file_path: str) -> Optional[str]:
        """Process a file from NAS and extract text content"""
        try:
            if not os.path.exists(file_path):
                return None
            
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.txt' or ext == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif ext == '.pdf':
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text
            # Add more file type processors as needed
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    # YouTube Data (requires YouTube Data API)
    def get_youtube_data(self) -> Dict[str, Any]:
        """Get YouTube watch history and preferences (requires API key)"""
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if not youtube_api_key:
            return {"error": "YouTube API key not configured"}
        
        try:
            # This would require OAuth for personal data
            # For now, return placeholder structure
            return {
                "watch_history": [],
                "subscriptions": [],
                "liked_videos": [],
                "playlists": []
            }
        except Exception as e:
            logger.error(f"YouTube API error: {e}")
            return {"error": str(e)}
    
    def generate_daily_insights(self) -> Dict[str, Any]:
        """Generate daily insights from all integrated data sources"""
        insights = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "health_summary": {},
            "music_mood": {},
            "environment": {},
            "productivity_indicators": {},
            "recommendations": []
        }
        
        try:
            # Get data from all sources
            spotify_data = self.get_spotify_data()
            hue_data = self.get_hue_data()
            
            # Analyze music mood
            if "recent_tracks" in spotify_data and spotify_data["recent_tracks"]:
                recent = spotify_data["recent_tracks"].get("items", [])
                insights["music_mood"] = {
                    "tracks_today": len(recent),
                    "top_artist": recent[0]["track"]["artists"][0]["name"] if recent else None,
                    "listening_active": len(recent) > 10
                }
            
            # Analyze environment (Hue lights)
            if "lights" in hue_data and hue_data["lights"]:
                lights_on = sum(1 for light in hue_data["lights"].values() 
                              if light.get("state", {}).get("on", False))
                insights["environment"] = {
                    "lights_active": lights_on,
                    "brightness_avg": sum(light.get("state", {}).get("bri", 0) 
                                        for light in hue_data["lights"].values()) / len(hue_data["lights"])
                }
            
            # Generate recommendations
            if insights["music_mood"].get("listening_active"):
                insights["recommendations"].append("High music activity detected - great creative energy today!")
            
            if insights["environment"].get("lights_active", 0) > 3:
                insights["recommendations"].append("Multiple lights on - productive work environment set up")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights["error"] = str(e)
        
        return insights
