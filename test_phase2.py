import sys
import os
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from main import extract_characters_with_age_context, setup_face_recognition_database

# Test script content
test_script = """
This is a story about the legendary actor Robert De Niro. 
We start with young Robert De Niro in the 1970s during his early career in Taxi Driver.
The scene transitions to show old Robert De Niro in his later years.
We also see Al Pacino in his prime, showcasing his incredible acting range.
The film features middle-aged Al Pacino in Scarface.
"""

if __name__ == "__main__":
    print("Testing Phase 2: Web Scraping & Image Collection")
    print("="*60)
    
    try:
        # Step 1: Extract characters (from Phase 1)
        print("Step 1: Extracting characters from script...")
        characters = extract_characters_with_age_context(test_script)
        print(f"Extracted characters: {characters}")
        
        if not characters:
            print("No characters found, exiting...")
            exit(1)
        
        print("\n" + "="*60)
        
        # Step 2: Set up face recognition database (includes image collection)
        print("Step 2: Setting up face recognition database...")
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # This will call collect_celebrity_images for each character/age combination
            face_database = setup_face_recognition_database(characters, temp_dir)
            
            print(f"\nFace database setup complete!")
            print(f"Database contains {len(face_database)} character/age combinations:")
            
            for key, data in face_database.items():
                print(f"  - {key}: {len(data['images'])} images")
                print(f"    Character: {data['character']}")
                print(f"    Age context: {data['age_context']}")
                print(f"    Image paths: {data['images'][:2]}...")  # Show first 2 paths
                print()
        
        print("Phase 2 testing completed successfully!")
        
    except Exception as e:
        print(f"Error during Phase 2 testing: {e}")
        import traceback
        traceback.print_exc() 