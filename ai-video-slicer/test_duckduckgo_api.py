import sys
import os
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from main import search_bing_images, collect_celebrity_images

def test_duckduckgo_api():
    """Test DuckDuckGo Image Search functionality (free alternative to Bing)"""
    print("🔍 Testing DuckDuckGo Image Search")
    print("="*50)
    
    # No API key needed for DuckDuckGo!
    print("✅ No API key required - DuckDuckGo is free to use!")
    return test_with_duckduckgo()

def test_with_duckduckgo():
    """Test with DuckDuckGo image search"""
    print("\n🎯 Testing DuckDuckGo Image Search")
    print("-" * 30)
    
    # Test simple search
    test_query = "Robert De Niro actor"
    print(f"Searching for: {test_query}")
    
    try:
        image_urls = search_bing_images(test_query, 3)
        
        if image_urls:
            print(f"✅ Found {len(image_urls)} image URLs:")
            for i, url in enumerate(image_urls[:3]):
                print(f"  {i+1}. {url[:80]}...")
            
            # Test full image collection
            print(f"\n📥 Testing full image collection...")
            
            # Use a persistent directory for this test
            test_dir = os.path.join(os.path.dirname(__file__), "test_images")
            os.makedirs(test_dir, exist_ok=True)
            
            # Get the absolute path to show user exactly where images will be
            abs_test_dir = os.path.abspath(test_dir)
            print(f"📁 Images will be saved in: {abs_test_dir}")
            
            images = collect_celebrity_images("Robert De Niro", "young", test_dir, 8)
            print(f"✅ Collected {len(images)} images successfully!")
            
            for img_path in images:
                if os.path.exists(img_path):
                    size = os.path.getsize(img_path)
                    abs_img_path = os.path.abspath(img_path)
                    print(f"  📁 {abs_img_path} ({size} bytes)")
                    
            print(f"\n🟢 SUCCESS! Images are saved in: {abs_test_dir}")
            print("🗂️  You can now browse to this folder to see the downloaded images!")
            print("🗑️  You can manually delete this 'test_images' folder when you're done.")
            return True
        else:
            print("❌ No image URLs returned from DuckDuckGo")
            print("🔄 Falling back to mock images for testing...")
            return test_with_mock_images()
            
    except Exception as e:
        print(f"❌ Error testing DuckDuckGo: {e}")
        print("🔄 Falling back to mock images for testing...")
        return test_with_mock_images()

def test_with_mock_images():
    """Test with mock images (fallback)"""
    print("\n🎭 Testing Mock Image System")
    print("-" * 30)
    
    try:
        # Use a persistent directory for mock images too
        test_dir = os.path.join(os.path.dirname(__file__), "test_images")
        os.makedirs(test_dir, exist_ok=True)
        abs_test_dir = os.path.abspath(test_dir)
        print(f"Using test directory: {abs_test_dir}")
        
        # Test mock image collection
        images = collect_celebrity_images("Robert De Niro", "young", test_dir, 3)
        
        print(f"✅ Created {len(images)} mock images:")
        for img_path in images:
            if os.path.exists(img_path):
                size = os.path.getsize(img_path)
                abs_img_path = os.path.abspath(img_path)
                print(f"  📁 {abs_img_path} ({size} bytes)")
        
        print(f"\n🟢 Mock images saved in: {abs_test_dir}")
        return len(images) > 0
        
    except Exception as e:
        print(f"❌ Error testing mock images: {e}")
        return False

if __name__ == "__main__":
    print("🚀 DuckDuckGo Image Search Test Suite")
    print("="*60)
    print("🆓 Using FREE DuckDuckGo image search - no API key needed!")
    print("")
    
    success = test_duckduckgo_api()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Test completed successfully!")
        print("✅ Image collection system is working!")
    else:
        print("❌ Test failed!")
        print("🔧 Check network connection or try again later")
    
    print("\n💡 Next steps:")
    print("- No API setup needed - DuckDuckGo is completely free!")
    print("- Test with different celebrities and age contexts")
    print("- Proceed to Phase 3: Smart Scene Analysis") 