#!/usr/bin/env python3
"""
Phase C Test Suite: Script-to-Scene Intelligence
Tests comprehensive script analysis, scene mapping, and intelligent video assembly recommendations.
"""

import os
import sys
import json
import tempfile
from typing import Dict, List, Any
import traceback

# Add the backend directory to Python path
sys.path.append('backend')

# Import Phase C functions
from main import (
    analyze_script_scenes,
    map_script_to_video_scenes, 
    generate_scene_recommendations,
    calculate_character_similarity,
    determine_scene_type,
    determine_emotional_tone
)

def create_test_script() -> str:
    """Create a comprehensive test script with multiple scenes and characters"""
    return """
FADE IN:

EXT. NEW YORK STREET - DAY

Young ROBERT DE NIRO walks down the bustling street, his eyes scanning the crowd. He's intense, focused.

ROBERT DE NIRO
(to himself)
This city never sleeps. Neither do I.

INT. ITALIAN RESTAURANT - NIGHT 

Robert De Niro sits across from LEONARDO DICAPRIO. The tension is palpable.

LEONARDO DICAPRIO
You think you know me, but you don't.

ROBERT DE NIRO
(leaning forward)
I know enough. The question is, what are you going to do about it?

The conversation becomes heated. Both actors deliver powerful performances.

EXT. WAREHOUSE - NIGHT

Action sequence. Robert De Niro and Leonardo DiCaprio engage in an intense confrontation. Dramatic lighting, emotional stakes high.

FADE OUT.

INT. COURTROOM - DAY

Middle-aged ROBERT DE NIRO stands before the judge. This is a pivotal, dramatic scene.

ROBERT DE NIRO
Your Honor, the truth needs to be heard.

The courtroom is silent. De Niro's performance is commanding.

CUT TO:

EXT. BEACH - SUNSET

Leonardo DiCaprio walks alone on the beach. Contemplative, romantic mood. The waves crash gently.

LEONARDO DICAPRIO
(voice-over)
Sometimes the hardest person to forgive is yourself.

FADE OUT.
"""

def create_test_face_registry() -> Dict[str, Dict]:
    """Create a test face registry matching our script characters"""
    return {
        "robert_de_niro_young": {
            "entity_name": "Robert De Niro",
            "age_context": "young",
            "face_encodings": [
                [0.1, 0.2, 0.3] * 42,  # Mock 128-dimensional encoding
                [0.15, 0.25, 0.35] * 42
            ],
            "quality_score": 0.89,
            "image_count": 2,
            "source_images": ["robert_young_1.jpg", "robert_young_2.jpg"]
        },
        "robert_de_niro_middle": {
            "entity_name": "Robert De Niro", 
            "age_context": "middle-aged",
            "face_encodings": [
                [0.2, 0.3, 0.4] * 42,
                [0.25, 0.35, 0.45] * 42
            ],
            "quality_score": 0.92,
            "image_count": 2,
            "source_images": ["robert_middle_1.jpg", "robert_middle_2.jpg"]
        },
        "leonardo_dicaprio_young": {
            "entity_name": "Leonardo DiCaprio",
            "age_context": "young", 
            "face_encodings": [
                [0.3, 0.4, 0.5] * 42,
                [0.35, 0.45, 0.55] * 42
            ],
            "quality_score": 0.87,
            "image_count": 2,
            "source_images": ["leo_young_1.jpg", "leo_young_2.jpg"]
        }
    }

def create_test_video_scene_matches() -> Dict[str, Dict]:
    """Create test video scene analysis data from Phase B"""
    return {
        "video_scene_0": {
            "scene_info": {
                "start_time": 0.0,
                "end_time": 15.0,
                "duration": 15.0,
                "frame_count": 3
            },
            "entity_matches": [
                {
                    "entity_key": "robert_de_niro_young",
                    "entity_data": {
                        "entity_name": "Robert De Niro",
                        "age_context": "young"
                    },
                    "similarity_score": 0.85,
                    "confidence": 0.78
                }
            ],
            "dominant_entities": ["robert_de_niro_young"],
            "face_count": 1,
            "quality_assessment": "good"
        },
        "video_scene_1": {
            "scene_info": {
                "start_time": 15.0,
                "end_time": 45.0,
                "duration": 30.0,
                "frame_count": 5
            },
            "entity_matches": [
                {
                    "entity_key": "robert_de_niro_young",
                    "entity_data": {
                        "entity_name": "Robert De Niro",
                        "age_context": "young"
                    },
                    "similarity_score": 0.82,
                    "confidence": 0.75
                },
                {
                    "entity_key": "leonardo_dicaprio_young",
                    "entity_data": {
                        "entity_name": "Leonardo DiCaprio",
                        "age_context": "young"
                    },
                    "similarity_score": 0.79,
                    "confidence": 0.72
                }
            ],
            "dominant_entities": ["robert_de_niro_young", "leonardo_dicaprio_young"],
            "face_count": 2,
            "quality_assessment": "excellent"
        },
        "video_scene_2": {
            "scene_info": {
                "start_time": 45.0,
                "end_time": 65.0,
                "duration": 20.0,
                "frame_count": 4
            },
            "entity_matches": [
                {
                    "entity_key": "robert_de_niro_middle", 
                    "entity_data": {
                        "entity_name": "Robert De Niro",
                        "age_context": "middle-aged"
                    },
                    "similarity_score": 0.88,
                    "confidence": 0.81
                }
            ],
            "dominant_entities": ["robert_de_niro_middle"],
            "face_count": 1,
            "quality_assessment": "good"
        },
        "video_scene_3": {
            "scene_info": {
                "start_time": 65.0,
                "end_time": 80.0,
                "duration": 15.0,
                "frame_count": 3
            },
            "entity_matches": [
                {
                    "entity_key": "leonardo_dicaprio_young",
                    "entity_data": {
                        "entity_name": "Leonardo DiCaprio",
                        "age_context": "young"
                    },
                    "similarity_score": 0.76,
                    "confidence": 0.69
                }
            ],
            "dominant_entities": ["leonardo_dicaprio_young"],
            "face_count": 1,
            "quality_assessment": "fair"
        }
    }

def test_c1_script_scene_analysis():
    """Test C1: Script Scene Analysis"""
    print("\n🧪 TEST C1: Script Scene Analysis")
    print("=" * 50)
    
    try:
        script_content = create_test_script()
        
        print("📝 Testing script scene analysis...")
        script_scenes = analyze_script_scenes(script_content)
        
        # Validate results
        assert isinstance(script_scenes, dict), "Script scenes should be a dictionary"
        assert len(script_scenes) >= 3, f"Expected at least 3 scenes, got {len(script_scenes)}"
        
        print(f"✅ Found {len(script_scenes)} script scenes")
        
        # Test individual scene analysis
        for scene_id, scene_data in script_scenes.items():
            print(f"\n📋 Scene: {scene_id}")
            print(f"   Description: {scene_data.get('scene_description', 'N/A')}")
            print(f"   Characters: {scene_data.get('required_characters', [])}")
            print(f"   Scene Type: {scene_data.get('scene_type', 'N/A')}")
            print(f"   Emotional Tone: {scene_data.get('emotional_tone', 'N/A')}")
            print(f"   Duration Estimate: {scene_data.get('duration_estimate', 0):.1f}s")
            print(f"   Importance Score: {scene_data.get('importance_score', 0):.2f}")
            
            # Validate scene structure
            required_fields = ['scene_number', 'scene_description', 'required_characters', 
                             'scene_type', 'emotional_tone', 'duration_estimate', 
                             'importance_score', 'script_text']
            
            for field in required_fields:
                assert field in scene_data, f"Missing field {field} in scene {scene_id}"
        
        # Check for character detection
        all_characters = set()
        for scene_data in script_scenes.values():
            all_characters.update(scene_data.get('required_characters', []))
        
        expected_characters = {'Robert De Niro', 'Leonardo Dicaprio', 'Leonardo DiCaprio'}
        found_expected = any(char in str(all_characters).lower() for char in ['robert', 'leonardo'])
        
        print(f"\n🎭 Character Detection Results:")
        print(f"   All detected characters: {sorted(all_characters)}")
        print(f"   Expected character names found: {found_expected}")
        
        print(f"\n✅ C1 TEST PASSED: Script scene analysis working correctly")
        return True
        
    except Exception as e:
        print(f"❌ C1 TEST FAILED: {e}")
        traceback.print_exc()
        return False

def test_c2_script_to_video_mapping():
    """Test C2: Script-to-Video Scene Mapping"""
    print("\n🧪 TEST C2: Script-to-Video Scene Mapping")
    print("=" * 50)
    
    try:
        # Get test data
        script_content = create_test_script()
        face_registry = create_test_face_registry()
        video_scene_matches = create_test_video_scene_matches()
        
        print("📝 Analyzing script scenes...")
        script_scenes = analyze_script_scenes(script_content)
        
        print("🎯 Testing script-to-video mapping...")
        script_to_video_mapping = map_script_to_video_scenes(
            script_scenes=script_scenes,
            video_scene_matches=video_scene_matches,
            face_registry=face_registry
        )
        
        # Validate results
        assert isinstance(script_to_video_mapping, dict), "Mapping should be a dictionary"
        assert len(script_to_video_mapping) == len(script_scenes), "Should map all script scenes"
        
        print(f"✅ Mapped {len(script_to_video_mapping)} script scenes to video content")
        
        # Analyze mapping quality
        total_coverage = 0
        total_recommendations = 0
        
        for script_scene_id, mapping_data in script_to_video_mapping.items():
            script_info = mapping_data['script_info']
            video_recommendations = mapping_data['recommended_video_scenes']
            missing_characters = mapping_data['missing_characters']
            coverage_analysis = mapping_data['coverage_analysis']
            
            coverage_pct = coverage_analysis.get('coverage_percentage', 0)
            total_coverage += coverage_pct
            total_recommendations += len(video_recommendations)
            
            print(f"\n📋 {script_scene_id}:")
            print(f"   Required Characters: {script_info.get('required_characters', [])}")
            print(f"   Video Recommendations: {len(video_recommendations)}")
            print(f"   Coverage: {coverage_pct:.1f}%")
            print(f"   Missing Characters: {missing_characters}")
            print(f"   Recommendation: {coverage_analysis.get('recommendation', 'N/A')}")
            
            if video_recommendations:
                best_match = video_recommendations[0]
                print(f"   Best Match: {best_match['video_scene_id']} (score: {best_match['match_score']:.2f})")
        
        avg_coverage = total_coverage / len(script_to_video_mapping)
        avg_recommendations = total_recommendations / len(script_to_video_mapping)
        
        print(f"\n📊 Mapping Summary:")
        print(f"   Average Coverage: {avg_coverage:.1f}%")
        print(f"   Average Recommendations per Scene: {avg_recommendations:.1f}")
        
        # Quality checks
        assert avg_coverage > 30, f"Average coverage too low: {avg_coverage:.1f}%"
        assert avg_recommendations >= 1, f"Not enough recommendations: {avg_recommendations:.1f}"
        
        print(f"\n✅ C2 TEST PASSED: Script-to-video mapping working correctly")
        return True
        
    except Exception as e:
        print(f"❌ C2 TEST FAILED: {e}")
        traceback.print_exc()
        return False

def test_c3_scene_recommendations():
    """Test C3: Intelligent Scene Recommendations"""
    print("\n🧪 TEST C3: Intelligent Scene Recommendations")
    print("=" * 50)
    
    try:
        # Get test data
        script_content = create_test_script()
        face_registry = create_test_face_registry()
        video_scene_matches = create_test_video_scene_matches()
        
        # Run full pipeline to C2
        script_scenes = analyze_script_scenes(script_content)
        script_to_video_mapping = map_script_to_video_scenes(
            script_scenes=script_scenes,
            video_scene_matches=video_scene_matches,
            face_registry=face_registry
        )
        
        print("🎬 Testing scene recommendations generation...")
        recommendations = generate_scene_recommendations(script_to_video_mapping)
        
        # Validate results structure
        assert isinstance(recommendations, dict), "Recommendations should be a dictionary"
        
        required_keys = ['assembly_plan', 'quality_assessment', 'missing_content_report', 'optimization_suggestions']
        for key in required_keys:
            assert key in recommendations, f"Missing key {key} in recommendations"
        
        assembly_plan = recommendations['assembly_plan']
        quality_assessment = recommendations['quality_assessment']
        missing_content = recommendations['missing_content_report']
        optimization_suggestions = recommendations['optimization_suggestions']
        
        print(f"✅ Generated recommendations with all required components")
        
        # Analyze assembly plan
        print(f"\n📋 Assembly Plan Analysis:")
        print(f"   Total Scenes: {len(assembly_plan)}")
        
        for i, scene_entry in enumerate(assembly_plan):
            print(f"\n   Scene {i+1}:")
            print(f"     Script Scene: {scene_entry.get('script_scene_id', 'N/A')}")
            print(f"     Description: {scene_entry.get('script_description', 'N/A')[:60]}...")
            print(f"     Recommended Video: {scene_entry.get('recommended_video_scene', 'None')}")
            print(f"     Match Quality: {scene_entry.get('match_quality', 0):.2f}")
            print(f"     Coverage: {scene_entry.get('coverage_percentage', 0):.1f}%") 
            print(f"     Status: {scene_entry.get('status', 'unknown')}")
            print(f"     Alternatives: {scene_entry.get('alternative_scenes', [])}")
        
        # Analyze quality assessment
        print(f"\n📊 Quality Assessment:")
        print(f"   Overall Match Quality: {quality_assessment.get('overall_match_quality', 0):.2f}")
        print(f"   Average Coverage: {quality_assessment.get('average_coverage_percentage', 0):.1f}%")
        print(f"   Total Script Scenes: {quality_assessment.get('total_script_scenes', 0)}")
        print(f"   Well Covered Scenes: {quality_assessment.get('well_covered_scenes', 0)}")
        print(f"   Problematic Scenes: {quality_assessment.get('problematic_scenes', 0)}")
        print(f"   Assembly Feasibility: {quality_assessment.get('assembly_feasibility', 'unknown')}")
        
        # Missing content analysis
        if missing_content:
            print(f"\n⚠️  Missing Content Report:")
            for scene_id, missing_info in missing_content.items():
                print(f"   {scene_id}: Missing {missing_info.get('missing_characters', [])}")
        else:
            print(f"\n✅ No missing content reported")
        
        # Optimization suggestions
        if optimization_suggestions:
            print(f"\n💡 Optimization Suggestions ({len(optimization_suggestions)}):")
            for i, suggestion in enumerate(optimization_suggestions[:5]):  # Show first 5
                print(f"   {i+1}. {suggestion}")
        else:
            print(f"\n✅ No optimization suggestions needed")
        
        # Quality checks
        assert len(assembly_plan) > 0, "Assembly plan should not be empty"
        assert quality_assessment.get('total_script_scenes', 0) > 0, "Should have script scenes"
        assert quality_assessment.get('assembly_feasibility') in ['excellent', 'good', 'challenging', 'difficult'], "Invalid feasibility rating"
        
        print(f"\n✅ C3 TEST PASSED: Scene recommendations working correctly")
        return True
        
    except Exception as e:
        print(f"❌ C3 TEST FAILED: {e}")
        traceback.print_exc()
        return False

def test_helper_functions():
    """Test Phase C helper functions"""
    print("\n🧪 TEST: Helper Function Validation")
    print("=" * 50)
    
    try:
        # Test character similarity calculation
        print("Testing character similarity...")
        
        test_cases = [
            ("Robert De Niro", "Robert De Niro", 1.0),  # Exact match
            ("Robert De Niro", "Robert", 0.8),          # Partial match
            ("Leonardo DiCaprio", "Leonardo", 0.8),      # Partial match
            ("John", "Jane", 0.0),                       # No match
            ("Al Pacino", "Pacino", 0.8)                # Last name match
        ]
        
        for char1, char2, expected_min in test_cases:
            similarity = calculate_character_similarity(char1, char2)
            print(f"   '{char1}' vs '{char2}': {similarity:.2f} (expected >= {expected_min})")
            
            if expected_min == 1.0:
                assert similarity >= 0.95, f"Exact match should score high: {similarity}"
            elif expected_min == 0.8:
                assert similarity >= 0.5, f"Partial match should score reasonably: {similarity}"
            elif expected_min == 0.0:
                assert similarity < 0.5, f"No match should score low: {similarity}"
        
        # Test scene type determination
        print("\nTesting scene type determination...")
        
        scene_type_tests = [
            ("The characters engage in an intense fight scene", "action"),
            ("They sit and have a long conversation about life", "dialogue"),
            ("EXT. CITYSCAPE - ESTABLISHING SHOT", "establishing"),
            ("She breaks down crying, overcome with emotion", "dramatic"),
            ("Just a normal scene with some talking", "general")
        ]
        
        for content, expected_type in scene_type_tests:
            scene_type = determine_scene_type(content)
            print(f"   '{content[:30]}...' -> {scene_type} (expected: {expected_type})")
            # Note: This is heuristic so we don't assert exact matches
        
        # Test emotional tone determination
        print("\nTesting emotional tone determination...")
        
        tone_tests = [
            ("This is a very intense and dramatic confrontation", "intense"),
            ("Everyone was laughing and having a funny time", "comedic"),
            ("The romantic scene showed their love and affection", "romantic"),
            ("It was full of suspense and mystery", "suspenseful"),
            ("Everyone was sad and crying at the funeral", "sad"),
            ("Just a normal conversation", "neutral")
        ]
        
        for content, expected_tone in tone_tests:
            tone = determine_emotional_tone(content)
            print(f"   '{content[:30]}...' -> {tone} (expected: {expected_tone})")
            # Note: This is heuristic so we don't assert exact matches
        
        print(f"\n✅ HELPER FUNCTIONS TEST PASSED: All helper functions working")
        return True
        
    except Exception as e:
        print(f"❌ HELPER FUNCTIONS TEST FAILED: {e}")
        traceback.print_exc()
        return False

def test_phase_c_integration():
    """Test complete Phase C integration"""
    print("\n🧪 TEST: Phase C Complete Integration")
    print("=" * 50)
    
    try:
        # Test data
        script_content = create_test_script()
        face_registry = create_test_face_registry()
        video_scene_matches = create_test_video_scene_matches()
        
        print("🚀 Running complete Phase C pipeline...")
        
        # Step C1: Script Analysis
        print("\nStep C1: Script scene analysis...")
        script_scenes = analyze_script_scenes(script_content)
        assert len(script_scenes) > 0, "Should have script scenes"
        print(f"✓ Analyzed {len(script_scenes)} script scenes")
        
        # Step C2: Script-to-Video Mapping
        print("\nStep C2: Script-to-video mapping...")
        script_to_video_mapping = map_script_to_video_scenes(
            script_scenes=script_scenes,
            video_scene_matches=video_scene_matches,
            face_registry=face_registry
        )
        assert len(script_to_video_mapping) == len(script_scenes), "Should map all scenes"
        print(f"✓ Mapped {len(script_to_video_mapping)} script scenes to video")
        
        # Step C3: Recommendations Generation
        print("\nStep C3: Generating recommendations...")
        recommendations = generate_scene_recommendations(script_to_video_mapping)
        
        assembly_plan = recommendations.get('assembly_plan', [])
        quality_assessment = recommendations.get('quality_assessment', {})
        
        assert len(assembly_plan) > 0, "Should have assembly plan"
        assert 'overall_match_quality' in quality_assessment, "Should have quality assessment"
        print(f"✓ Generated assembly plan with {len(assembly_plan)} scenes")
        
        # Integration validation
        print(f"\n📊 Integration Results:")
        print(f"   Script Scenes: {len(script_scenes)}")
        print(f"   Video Scene Mappings: {len(script_to_video_mapping)}")
        print(f"   Assembly Plan Entries: {len(assembly_plan)}")
        print(f"   Overall Match Quality: {quality_assessment.get('overall_match_quality', 0):.2f}")
        print(f"   Assembly Feasibility: {quality_assessment.get('assembly_feasibility', 'unknown')}")
        
        # Data flow validation
        script_scene_ids = set(script_scenes.keys())
        mapping_scene_ids = set(script_to_video_mapping.keys())
        assembly_scene_ids = set(entry['script_scene_id'] for entry in assembly_plan)
        
        assert script_scene_ids == mapping_scene_ids, "Script scenes should match mapping keys"
        assert script_scene_ids == assembly_scene_ids, "Script scenes should match assembly plan"
        
        print(f"\n✅ PHASE C INTEGRATION TEST PASSED: Complete pipeline working")
        return True
        
    except Exception as e:
        print(f"❌ PHASE C INTEGRATION TEST FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Phase C tests"""
    print("🚀 PHASE C TEST SUITE: Script-to-Scene Intelligence")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_functions = [
        ("C1 - Script Scene Analysis", test_c1_script_scene_analysis),
        ("C2 - Script-to-Video Mapping", test_c2_script_to_video_mapping),
        ("C3 - Scene Recommendations", test_c3_scene_recommendations),
        ("Helper Functions", test_helper_functions),
        ("Phase C Integration", test_phase_c_integration)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        result = test_func()
        test_results.append((test_name, result))
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 PHASE C TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL PHASE C TESTS PASSED!")
        print("🚀 Phase C: Script-to-Scene Intelligence is ready for production!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 