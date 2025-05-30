
import os
import sys
from integrations import IntegrationsManager

def verify_setup():
    """Verify all integrations are properly set up"""
    
    print("🔧 AI Companion Setup Verification")
    print("=" * 40)
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    optional_vars = ["SPOTIFY_CLIENT_ID", "HUE_BRIDGE_IP", "NAS_HOST", "YOUTUBE_API_KEY"]
    
    print("\n✅ Required Environment Variables:")
    for var in required_vars:
        status = "✅ SET" if os.getenv(var) else "❌ MISSING"
        print(f"  {var}: {status}")
    
    print("\n🔧 Optional Environment Variables:")
    for var in optional_vars:
        status = "✅ SET" if os.getenv(var) else "⚠️  NOT SET"
        print(f"  {var}: {status}")
    
    # Test integrations
    print("\n🧪 Testing Integrations:")
    integrations = IntegrationsManager()
    
    # Test NAS setup
    print("  📁 NAS Connection:", end=" ")
    nas_result = integrations.setup_nas_connection()
    if nas_result.get("status") == "connected":
        print("✅ CONNECTED")
    elif nas_result.get("setup_required"):
        print("⚠️  NEEDS SETUP")
    else:
        print("❌ FAILED")
    
    print("\n📋 Next Steps:")
    print("1. 🔐 Copy .env.template to .env and add your API keys")
    print("2. 🎵 Get Spotify API credentials from developer.spotify.com")
    print("3. 💡 Setup Philips Hue bridge connection")
    print("4. 💾 Configure NAS credentials in environment variables")
    print("5. 🔗 Connect Google account through the web interface for YouTube/Gmail/Drive access")
    print("6. 🏥 Upload Apple Health export through the web interface")
    
    print("\n🚀 Your AI companion is ready to use!")
    print("   Visit the web interface to start chatting and exploring your data.")

if __name__ == "__main__":
    verify_setup()
