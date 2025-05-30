
import os
import json
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GoogleCalendarManager:
    def __init__(self):
        # Load credentials from the JSON file
        credentials_path = 'attached_assets/client_secret_343105907066-lfufef3jo7vk0vdokurifst8rfp94lon.apps.googleusercontent.com.json'
        if os.path.exists(credentials_path):
            with open(credentials_path, 'r') as f:
                self.client_config = json.load(f)
        else:
            raise FileNotFoundError(f"Google credentials file not found at {credentials_path}")
        
        self.scopes = [
            'https://www.googleapis.com/auth/calendar.readonly',
            'https://www.googleapis.com/auth/calendar.events',
            'https://www.googleapis.com/auth/youtube.readonly',
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/drive.readonly',
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile'
        ]
        
    def create_auth_flow(self, redirect_uri):
        """Create OAuth flow for calendar access"""
        flow = Flow.from_client_config(
            self.client_config,
            scopes=self.scopes
        )
        flow.redirect_uri = redirect_uri
        return flow
    
    def get_calendar_service(self, credentials_dict):
        """Create calendar service with user credentials"""
        credentials = Credentials.from_authorized_user_info(credentials_dict)
        return build('calendar', 'v3', credentials=credentials)
    
    def get_upcoming_events(self, service, max_results=10):
        """Get upcoming calendar events"""
        try:
            now = datetime.utcnow().isoformat() + 'Z'
            events_result = service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            
            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                formatted_events.append({
                    'summary': event.get('summary', 'No Title'),
                    'start': start,
                    'description': event.get('description', ''),
                    'location': event.get('location', '')
                })
            
            return formatted_events
        except HttpError as error:
            print(f'An error occurred: {error}')
            return []
    
    def create_event(self, service, event_data):
        """Create a new calendar event"""
        try:
            event = service.events().insert(
                calendarId='primary',
                body=event_data
            ).execute()
            return event
        except HttpError as error:
            print(f'An error occurred: {error}')
            return None
