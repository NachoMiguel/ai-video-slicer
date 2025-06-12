import os
import sys

# Load environment variables from backend/.env
try:
    from dotenv import load_dotenv
    # Look for .env file in backend directory
    env_path = os.path.join(os.path.dirname(__file__), 'backend', '.env')
    load_dotenv(env_path)
    print(f"🔧 Loaded environment variables from: {env_path}")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_google_api():
    """Test Google Custom Search API configuration"""
    
    print("🔍 Testing Google Custom Search API Configuration")
    print("=" * 50)
    
    # Check environment variables
    google_api_key = os.getenv('GOOGLE_API_KEY')
    google_cse_id = os.getenv('GOOGLE_CSE_ID')
    
    print(f"📋 Environment Variables:")
    print(f"   GOOGLE_API_KEY: {'✓ Set' if google_api_key else '✗ Missing'}")
    print(f"   GOOGLE_CSE_ID: {'✓ Set' if google_cse_id else '✗ Missing'}")
    
    if google_api_key:
        print(f"   API Key length: {len(google_api_key)} characters")
        print(f"   API Key preview: {google_api_key[:10]}...{google_api_key[-4:]}")
    
    if google_cse_id:
        print(f"   CSE ID: {google_cse_id}")
    
    if not google_api_key or not google_cse_id:
        print("\n❌ Missing required credentials")
        return False
    
    # Test basic API connection
    try:
        print(f"\n🔗 Testing API Connection...")
        from googleapiclient.discovery import build
        
        service = build("customsearch", "v1", developerKey=google_api_key)
        print("   ✓ Google API service created successfully")
        
        # Test basic text search first
        print(f"\n📝 Testing basic text search...")
        result = service.cse().list(
            q="test search",
            cx=google_cse_id,
            num=1
        ).execute()
        
        print(f"   ✓ Basic search successful")
        print(f"   ✓ Found {len(result.get('items', []))} results")
        
        # Test image search
        print(f"\n🖼️  Testing image search...")
        result = service.cse().list(
            q="Leonardo DiCaprio",
            cx=google_cse_id,
            searchType='image',
            num=3
        ).execute()
        
        items = result.get('items', [])
        print(f"   ✓ Image search successful")
        print(f"   ✓ Found {len(items)} image results")
        
        if items:
            print(f"\n📸 Sample image results:")
            for i, item in enumerate(items[:3]):
                print(f"   {i+1}. {item.get('title', 'No title')}")
                print(f"      URL: {item.get('link', 'No link')[:80]}...")
        else:
            print(f"\n⚠️  No image results found")
            print(f"      This might indicate:")
            print(f"      - Image search is not enabled in your Custom Search Engine")
            print(f"      - Search parameters are too restrictive")
            print(f"      - Custom Search Engine configuration needs adjustment")
        
        return len(items) > 0
        
    except Exception as e:
        print(f"\n❌ API Test failed: {e}")
        return False

def test_custom_search_configuration():
    """Test and provide guidance on Custom Search Engine configuration"""
    
    print(f"\n🛠️  Custom Search Engine Configuration Check")
    print("=" * 50)
    
    google_cse_id = os.getenv('GOOGLE_CSE_ID')
    
    if google_cse_id:
        print(f"📋 Your Custom Search Engine ID: {google_cse_id}")
        print(f"\n🔧 Required Configuration Settings:")
        print(f"   1. Go to: https://cse.google.com/cse/setup/basic?cx={google_cse_id}")
        print(f"   2. Check 'Image search' is ON")
        print(f"   3. Check 'Search the entire web' is ON")
        print(f"   4. In 'Sites to search', should have '*' (asterisk)")
        print(f"   5. SafeSearch can be OFF for better results")
        
        print(f"\n💡 If getting 0 results:")
        print(f"   - Verify image search is enabled")
        print(f"   - Try searching for simpler terms like 'cat' or 'dog' first")
        print(f"   - Check your Google API quotas in Google Cloud Console")
    
    return True

if __name__ == "__main__":
    print("🧪 Google API Diagnostic Test")
    print("Testing Google Custom Search API setup and configuration\n")
    
    # Test 1: Basic API functionality
    api_works = test_google_api()
    
    # Test 2: Configuration guidance
    test_custom_search_configuration()
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if api_works:
        print(f"✅ Google API is working correctly!")
        print(f"✅ Image search is functional")
        print(f"🎉 Ready for real celebrity image downloads")
    else:
        print(f"❌ Google API needs configuration")
        print(f"📚 Follow the configuration guidance above")
        print(f"🔄 The curated database fallback will work in the meantime") 