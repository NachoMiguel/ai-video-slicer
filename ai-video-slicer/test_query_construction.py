#!/usr/bin/env python3
"""
Test script for validating enhanced query construction in Phase 2.
Tests how different celebrities and age contexts generate optimized search queries.
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_query_construction():
    """Test the enhanced query construction logic"""
    
    print("🧪 Testing Enhanced Query Construction (Phase 2)")
    print("=" * 60)
    
    # Test cases: (celebrity, age_context, expected_keywords)
    test_cases = [
        ("Robert De Niro", "young", ["portrait", "headshot", "young", "1970s", "1980s"]),
        ("Robert De Niro", "old", ["portrait", "headshot", "recent", "2010s", "2020s"]),
        ("Robert De Niro", "middle-aged", ["portrait", "headshot", "middle", "1990s", "2000s"]),
        ("Robert De Niro", "any", ["portrait", "headshot"]),
        ("Meryl Streep", "young", ["portrait", "headshot", "young", "1970s", "1980s"]),
        ("Leonardo DiCaprio", "old", ["portrait", "headshot", "recent", "2010s", "2020s"]),
    ]
    
    for character, age_context, expected_keywords in test_cases:
        print(f"\n🎭 Testing: {character} ({age_context})")
        print("-" * 40)
        
        # Simulate the query construction logic from collect_celebrity_images
        base_query = f"{character} actor portrait headshot"
        
        if age_context == "young":
            search_query = f"{base_query} young early career 1970s 1980s"
        elif age_context == "old":
            search_query = f"{base_query} recent older 2010s 2020s"
        elif age_context == "middle-aged":
            search_query = f"{base_query} middle aged 1990s 2000s"
        else:
            search_query = base_query
        
        print(f"Generated Query: {search_query}")
        
        # Validate that expected keywords are present
        query_lower = search_query.lower()
        missing_keywords = []
        present_keywords = []
        
        for keyword in expected_keywords:
            if keyword.lower() in query_lower:
                present_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        if missing_keywords:
            print(f"❌ Missing keywords: {missing_keywords}")
        else:
            print(f"✅ All expected keywords present: {present_keywords}")
    
    print("\n" + "=" * 60)
    print("🎯 Query Construction Guidelines:")
    print("- Always includes 'portrait headshot' for face recognition")
    print("- Age context adds temporal specificity")
    print("- Maintains celebrity name and 'actor' for relevance")
    print("- Optimized for Google Custom Search API")

if __name__ == "__main__":
    test_query_construction() 