
import os
from pathlib import Path

def setup_environment():
    """Setup environment variables for all integrations"""
    
    env_template = """
# AI Companion Integration Settings
# Copy this to .env and fill in your actual values

# Required - Already configured
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Spotify Integration (Optional)
# Get these from: https://developer.spotify.com/dashboard/
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_ACCESS_TOKEN=your_spotify_access_token

# Philips Hue Integration (Optional)
# Find your bridge IP and create a username
HUE_BRIDGE_IP=192.168.1.xxx
HUE_USERNAME=your_hue_username

# NAS Integration (Optional)
# Configure your Synology NAS access
NAS_HOST=your_nas_hostname_or_ip
NAS_USERNAME=your_nas_username
NAS_PASSWORD=your_nas_password

# YouTube Data API (Optional)
# Get from: https://console.developers.google.com/
YOUTUBE_API_KEY=your_youtube_api_key

# Apple Health (Manual Import)
# Export health data from iPhone Health app > Export All Health Data
# Then upload the XML file through the web interface

# Additional Settings
VOICE_ENABLED=true
DAILY_BRIEFING_TIME=08:00
TIMEZONE=America/New_York
"""
    
    # Write template to .env.template
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("🔧 Integration setup created!")
    print("📝 Check .env.template for configuration options")
    print("\n📋 Setup checklist:")
    print("1. ✅ Core AI features (already working)")
    print("2. 🔄 Copy .env.template to .env and fill in your API keys")
    print("3. 🎵 Spotify: Get API credentials from developer.spotify.com")
    print("4. 💡 Philips Hue: Find bridge IP and create username")
    print("5. 💾 NAS: Configure network access to your Synology")
    print("6. 🏥 Apple Health: Export data from iPhone Health app")
    print("7. 🎥 YouTube: Get API key from Google Cloud Console")
    print("\n🚀 Features you can use right now:")
    print("- Voice interaction (upload audio files)")
    print("- Daily briefing and insights")
    print("- File processing from any source")
    print("- Enhanced mobile experience")

if __name__ == "__main__":
    setup_environment()
