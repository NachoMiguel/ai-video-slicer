import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from main import extract_characters_with_age_context

# Test script content
test_script = """
This is a story about the legendary actor Robert De Niro. 
We start with young Robert De Niro in the 1970s during his early career in Taxi Driver.
The scene transitions to show old Robert De Niro in his later years.
We also see Al Pacino in his prime, showcasing his incredible acting range.
The film features middle-aged Al Pacino in Scarface.
"""

if __name__ == "__main__":
    print("Testing character extraction...")
    print(f"Test script: {test_script}")
    print("\n" + "="*50)
    
    try:
        result = extract_characters_with_age_context(test_script)
        print(f"Extracted characters: {result}")
        
        # Expected: {"Robert De Niro": ["young", "old"], "Al Pacino": ["any", "middle-aged"]}
        
    except Exception as e:
        print(f"Error during testing: {e}") 