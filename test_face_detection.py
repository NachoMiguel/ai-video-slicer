#!/usr/bin/env python3
"""
Test script for Phase 4: Face Detection Filter

This script tests the new face detection functionality that filters images
to ensure only images containing human faces are kept for AI training.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the backend directory to the Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_face_detection_setup():
    """Test that face detection libraries are properly installed."""
    
    print("=== Phase 4: Face Detection Setup Test ===\n")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        print(f"  OpenCV version: {cv2.__version__}")
        
        import face_recognition
        print("✓ face_recognition imported successfully")
        
        import dlib
        print("✓ dlib imported successfully")
        print(f"  dlib version: {dlib.version}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install required packages with: pip install -r backend/requirements.txt")
        return False

def test_face_detection_function():
    """Test the face detection function with sample images."""
    
    print("\n=== Face Detection Function Test ===\n")
    
    try:
        from main import detect_faces_in_image, ENABLE_FACE_DETECTION, MIN_FACE_SIZE
        
        print(f"Face detection enabled: {ENABLE_FACE_DETECTION}")
        print(f"Minimum face size: {MIN_FACE_SIZE} pixels")
        
        # Test with a simple test image (create a basic image for testing)
        test_image_path = create_test_image()
        
        if test_image_path:
            print(f"Testing face detection on: {test_image_path}")
            
            result = detect_faces_in_image(test_image_path)
            print(f"Detection result: {result}")
            
            # Clean up test image
            try:
                os.remove(test_image_path)
            except:
                pass
            
            return result['method_used'] != 'error'
        else:
            print("Could not create test image")
            return False
            
    except Exception as e:
        print(f"✗ Face detection function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_image():
    """Create a simple test image for face detection testing."""
    
    try:
        from PIL import Image, ImageDraw
        import tempfile
        
        # Create a simple image with a circle (simulating a face)
        width, height = 300, 300
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw a simple "face" - circle with two dots for eyes
        draw.ellipse([50, 50, 250, 250], fill='lightpink', outline='black')
        draw.ellipse([100, 120, 120, 140], fill='black')  # Left eye
        draw.ellipse([180, 120, 200, 140], fill='black')  # Right eye
        draw.arc([120, 160, 180, 200], 0, 180, fill='black')  # Smile
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        image.save(temp_file.name, 'JPEG')
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        print(f"Could not create test image: {e}")
        return None

def test_celebrity_image_collection():
    """Test the enhanced celebrity image collection with face detection."""
    
    print("\n=== Celebrity Image Collection with Face Detection ===\n")
    
    try:
        from main import collect_celebrity_images
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Check if API keys are available
        google_api_key = os.getenv('GOOGLE_API_KEY')
        google_cse_id = os.getenv('GOOGLE_CSE_ID')
        
        if not google_api_key or not google_cse_id:
            print("⚠️  Google API credentials not found in environment variables")
            print("Please set GOOGLE_API_KEY and GOOGLE_CSE_ID to test real API functionality")
            print("The test will demonstrate face detection logic with mock data\n")
        else:
            print("✓ Google Custom Search API credentials found")
            print(f"  API Key: {'*' * 20}{google_api_key[-8:]}")
            print(f"  CSE ID: {google_cse_id}\n")
        
        # Test with a well-known celebrity
        test_character = "Leonardo DiCaprio"
        test_age_context = "any"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Testing image collection for: {test_character} ({test_age_context})")
            print(f"Using temporary directory: {temp_dir}")
            
            # Collect images with face detection enabled
            images = collect_celebrity_images(
                character=test_character,
                age_context=test_age_context,
                temp_dir=temp_dir,
                num_images=3  # Small number for testing
            )
            
            print(f"\nCollection results:")
            print(f"  Images collected: {len(images)}")
            
            if images:
                print("  Image paths:")
                for i, img_path in enumerate(images, 1):
                    file_size = os.path.getsize(img_path) if os.path.exists(img_path) else 0
                    print(f"    {i}. {img_path} ({file_size} bytes)")
                
                # Test face detection on collected images
                print("\n  Face detection verification:")
                for img_path in images:
                    if os.path.exists(img_path):
                        from main import detect_faces_in_image
                        result = detect_faces_in_image(img_path)
                        print(f"    {os.path.basename(img_path)}: {result['face_count']} faces (quality: {result['quality_score']:.2f})")
            
            return len(images) > 0
            
    except Exception as e:
        print(f"✗ Celebrity image collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_detection_configuration():
    """Test different face detection configuration options."""
    
    print("\n=== Face Detection Configuration Test ===\n")
    
    try:
        # Test environment variable loading
        import os
        from main import ENABLE_FACE_DETECTION, MIN_FACE_SIZE, FACE_DETECTION_METHOD, MAX_FACES_PER_IMAGE
        
        print("Current configuration:")
        print(f"  ENABLE_FACE_DETECTION: {ENABLE_FACE_DETECTION}")
        print(f"  MIN_FACE_SIZE: {MIN_FACE_SIZE}")
        print(f"  FACE_DETECTION_METHOD: {FACE_DETECTION_METHOD}")
        print(f"  MAX_FACES_PER_IMAGE: {MAX_FACES_PER_IMAGE}")
        
        # Test different methods if possible
        methods_to_test = ["face_recognition", "opencv"]
        
        for method in methods_to_test:
            print(f"\nTesting with method: {method}")
            
            # Temporarily set the method
            original_method = os.environ.get('FACE_DETECTION_METHOD', 'face_recognition')
            os.environ['FACE_DETECTION_METHOD'] = method
            
            # Create a test image and test detection
            test_image_path = create_test_image()
            if test_image_path:
                try:
                    # Reload the module to pick up new environment variable
                    import importlib
                    import main
                    importlib.reload(main)
                    
                    result = main.detect_faces_in_image(test_image_path)
                    print(f"  Method {method}: {result['method_used']} (faces: {result['face_count']})")
                    
                    os.remove(test_image_path)
                except Exception as e:
                    print(f"  Method {method} failed: {e}")
            
            # Restore original method
            os.environ['FACE_DETECTION_METHOD'] = original_method
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all Phase 4 tests."""
    
    print("🔍 Phase 4: Face Detection Filter - Comprehensive Test Suite")
    print("=" * 70)
    
    tests = [
        ("Setup Test", test_face_detection_setup),
        ("Function Test", test_face_detection_function),
        ("Configuration Test", test_face_detection_configuration),
        ("Celebrity Collection Test", test_celebrity_image_collection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print(f"\n{'='*70}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 4 is ready for use.")
    elif passed >= total * 0.7:  # 70% success rate
        print("⚠️  Most tests passed. Some features may have issues.")
    else:
        print("❌ Multiple test failures. Please check your setup.")
    
    print("\n💡 Next steps:")
    print("  1. Install missing dependencies if setup failed")
    print("  2. Set Google API credentials for full functionality")
    print("  3. Run the main application to test end-to-end")
    print("  4. Check face detection logs for performance metrics")

if __name__ == "__main__":
    main() 