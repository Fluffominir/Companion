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
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

class IntegrationsManager:
    def __init__(self):
        self.spotify_token = os.getenv("SPOTIFY_ACCESS_TOKEN")
        self.hue_bridge_ip = os.getenv("HUE_BRIDGE_IP")
        self.hue_username = os.getenv("HUE_USERNAME")
        self.nas_host = os.getenv("NAS_HOST")
        self.nas_username = os.getenv("NAS_USERNAME")
        self.nas_password = os.getenv("NAS_PASSWORD")
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")

    def get_google_service(self, service_name: str, version: str, credentials_dict: dict):
        """Create Google API service with user credentials"""
        try:
            credentials = Credentials.from_authorized_user_info(credentials_dict)
            return build(service_name, version, credentials=credentials)
        except Exception as e:
            logger.error(f"Error creating {service_name} service: {e}")
            return None

    # YouTube Integration
    def get_youtube_data(self, credentials_dict: dict = None) -> Dict[str, Any]:
        """Get YouTube watch history, subscriptions, and preferences"""
        try:
            if credentials_dict:
                # Use OAuth for personal data
                youtube = self.get_google_service('youtube', 'v3', credentials_dict)
                if not youtube:
                    return {"error": "Failed to create YouTube service"}

                # Get user's channel info
                channels_response = youtube.channels().list(
                    part='snippet,statistics',
                    mine=True
                ).execute()

                # Get subscriptions
                subscriptions_response = youtube.subscriptions().list(
                    part='snippet',
                    mine=True,
                    maxResults=50
                ).execute()

                # Get liked videos
                playlists_response = youtube.playlists().list(
                    part='snippet',
                    mine=True,
                    maxResults=50
                ).execute()

                return {
                    "channels": channels_response.get('items', []),
                    "subscriptions": subscriptions_response.get('items', []),
                    "playlists": playlists_response.get('items', []),
                    "status": "success"
                }
            else:
                return {"error": "YouTube OAuth credentials required for personal data"}

        except Exception as e:
            logger.error(f"YouTube API error: {e}")
            return {"error": str(e)}

    # Gmail Integration
    def get_gmail_data(self, credentials_dict: dict) -> Dict[str, Any]:
        """Get Gmail insights and recent emails"""
        try:
            gmail = self.get_google_service('gmail', 'v1', credentials_dict)
            if not gmail:
                return {"error": "Failed to create Gmail service"}

            # Get profile
            profile = gmail.users().getProfile(userId='me').execute()

            # Get recent messages (last 50)
            messages = gmail.users().messages().list(
                userId='me',
                maxResults=50,
                q='is:unread OR newer_than:7d'
            ).execute()

            # Get labels
            labels = gmail.users().labels().list(userId='me').execute()

            # Get detailed info for recent messages
            recent_emails = []
            for msg in messages.get('messages', [])[:10]:  # Only get details for 10 most recent
                try:
                    email = gmail.users().messages().get(
                        userId='me', 
                        id=msg['id'],
                        format='metadata'
                    ).execute()

                    headers = email.get('payload', {}).get('headers', [])
                    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                    sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                    date = next((h['value'] for h in headers if h['name'] == 'Date'), '')

                    recent_emails.append({
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'snippet': email.get('snippet', '')
                    })
                except:
                    continue

            return {
                "profile": profile,
                "recent_emails": recent_emails,
                "total_messages": profile.get('messagesTotal', 0),
                "labels": labels.get('labels', []),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Gmail API error: {e}")
            return {"error": str(e)}

    # Google Drive Integration
    def get_drive_data(self, credentials_dict: dict) -> Dict[str, Any]:
        """Get Google Drive file insights"""
        try:
            drive = self.get_google_service('drive', 'v3', credentials_dict)
            if not drive:
                return {"error": "Failed to create Drive service"}

            # Get recent files
            files = drive.files().list(
                pageSize=50,
                orderBy='modifiedTime desc',
                fields="nextPageToken, files(id, name, size, mimeType, modifiedTime, owners)"
            ).execute()

            # Get storage quota
            about = drive.about().get(fields="storageQuota, user").execute()

            # Categorize files by type
            file_types = {}
            total_size = 0

            for file in files.get('files', []):
                mime_type = file.get('mimeType', 'unknown')
                file_types[mime_type] = file_types[mime_type] + 1 if mime_type in file_types else 1

                size = file.get('size')
                if size:
                    total_size += int(size)

            return {
                "recent_files": files.get('files', []),
                "file_type_distribution": file_types,
                "total_files": len(files.get('files', [])),
                "storage_quota": about.get('storageQuota', {}),
                "user_info": about.get('user', {}),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Drive API error: {e}")
            return {"error": str(e)}

    # NAS Integration Setup
    def setup_nas_connection(self) -> Dict[str, Any]:
        """Setup and test NAS connection"""
        if not all([self.nas_host, self.nas_username, self.nas_password]):
            return {
                "error": "NAS credentials not configured",
                "setup_required": True,
                "instructions": [
                    "1. Set NAS_HOST environment variable (e.g., 192.168.1.100 or nas.local)",
                    "2. Set NAS_USERNAME environment variable",
                    "3. Set NAS_PASSWORD environment variable",
                    "4. Ensure SMB/CIFS is enabled on your Synology NAS",
                    "5. Create a user account with appropriate permissions"
                ]
            }

        try:
            # Test connection using smbclient (if available)
            test_cmd = f"timeout 10 smbclient -L //{self.nas_host} -U {self.nas_username}%{self.nas_password} --option='client min protocol=SMB2'"
            result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                return {
                    "status": "connected",
                    "message": "NAS connection successful",
                    "shares": result.stdout
                }
            else:
                return {
                    "status": "connection_failed",
                    "error": result.stderr,
                    "troubleshooting": [
                        "Check if NAS IP address is correct",
                        "Verify username and password",
                        "Ensure SMB service is running on NAS",
                        "Check firewall settings"
                    ]
                }

        except Exception as e:
            return {
                "status": "setup_incomplete",
                "error": str(e),
                "alternative_setup": [
                    "Install smbclient: apt-get install smbclient",
                    "Or manually mount NAS as network drive",
                    "Or use Synology Drive client"
                ]
            }

    def scan_nas_files_advanced(self, share_name: str = None) -> List[Dict[str, Any]]:
        """Advanced NAS file scanning with SMB protocol"""
        if not self.nas_host:
            return []

        try:
            from smbprotocol.connection import Connection
            from smbprotocol.session import Session
            from smbprotocol.tree import TreeConnect
            from smbprotocol.open import Open, CreateDisposition, ImpersonationLevel, CreateOptions
            from smbprotocol.file_info import FileInformationClass
            import socket

            # Use smbprotocol for better compatibility
            connection = Connection(socket.gethostbyname(self.nas_host), 445)
            connection.connect()

            session = Session(connection, self.nas_username, self.nas_password)
            session.connect()

            share = share_name or "homes"  # Default to homes share
            tree = TreeConnect(session, f"\\\\{self.nas_host}\\{share}")
            tree.connect()

            # List files in root directory
            files = []
            root_dir = Open(tree, "")
            try:
                root_dir.create(
                    ImpersonationLevel.Impersonation,
                    CreateOptions.FILE_DIRECTORY_FILE,
                    CreateDisposition.FILE_OPEN
                )

                for file_info in root_dir.query_directory("*", FileInformationClass.FILE_DIRECTORY_INFORMATION):
                    if not file_info.file_name.startswith('.'):
                        files.append({
                            "name": file_info.file_name,
                            "size": file_info.end_of_file,
                            "modified": file_info.last_write_time.isoformat(),
                            "is_directory": bool(file_info.file_attributes & 0x10),
                            "path": f"//{self.nas_host}/{share}/{file_info.file_name}"
                        })

                root_dir.close()
            except Exception as e:
                logger.error(f"Directory listing error: {e}")

            tree.disconnect()
            session.disconnect()
            connection.disconnect()

            return files[:50]  # Limit results

        except Exception as e:
            logger.error(f"SMB protocol error: {e}")
            # Fallback to smbclient if available
            try:
                share = share_name or "homes"
                cmd = f"smbclient '//{self.nas_host}/{share}' -U '{self.nas_username}%{self.nas_password}' -c 'ls'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    files = []
                    for line in result.stdout.split('\n'):
                        if line.strip() and not line.startswith('  .') and not line.startswith('NT_'):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                filename = ' '.join(parts[:-1]) if len(parts) > 2 else parts[0]
                                files.append({
                                    "name": filename,
                                    "size": "unknown",
                                    "modified": "unknown",
                                    "path": f"//{self.nas_host}/{share}/{filename}"
                                })
                    return files[:50]
                else:
                    logger.error(f"smbclient fallback failed: {result.stderr}")
                    return []
            except Exception as fallback_error:
                logger.error(f"Fallback error: {fallback_error}")
                return []

    def process_nas_file(self, file_path: str) -> str:
        """Download and process a file from NAS"""
        try:
            import tempfile
            from smbprotocol.connection import Connection
            from smbprotocol.session import Session
            from smbprotocol.tree import TreeConnect
            from smbprotocol.open import Open, CreateDisposition, ImpersonationLevel
            import socket

            # Parse file path
            parts = file_path.replace('//', '').split('/')
            host = parts[0]
            share = parts[1]
            file_name = '/'.join(parts[2:])

            # Connect to NAS
            connection = Connection(socket.gethostbyname(host), 445)
            connection.connect()

            session = Session(connection, self.nas_username, self.nas_password)
            session.connect()

            tree = TreeConnect(session, f"\\\\{host}\\{share}")
            tree.connect()

            # Open and read file
            file_open = Open(tree, file_name)
            file_open.create(
                ImpersonationLevel.Impersonation,
                CreateDisposition.FILE_OPEN
            )

            # Read file content
            file_content = file_open.read(0, file_open.end_of_file)
            file_open.close()

            tree.disconnect()
            session.disconnect()
            connection.disconnect()

            # Process based on file type
            file_ext = file_name.lower().split('.')[-1]
            
            if file_ext == 'pdf':
                # Save temporarily and extract text
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name

                import pdfplumber
                text_content = ""
                with pdfplumber.open(temp_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                
                os.unlink(temp_path)
                return text_content

            elif file_ext in ['txt', 'md', 'log']:
                return file_content.decode('utf-8', errors='ignore')
            
            elif file_ext in ['docx']:
                # Handle Word documents
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name

                from docx import Document
                doc = Document(temp_path)
                text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                
                os.unlink(temp_path)
                return text_content

            else:
                return f"File type {file_ext} not supported for content extraction"

        except Exception as e:
            logger.error(f"NAS file processing error: {e}")
            return ""

    # Apple Health Integration (existing code)
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

    # Spotify Integration (existing code)
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

    # Philips Hue Integration (existing code)
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