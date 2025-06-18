import os
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import openai
from backend.openai_account_manager import create_openai_client
from backend.script_session_manager import ScriptAnalysis

@dataclass
class StructureAnalysis:
    overall_score: float  # 1-10
    hook_effectiveness: float  # 1-10
    flow_quality: float  # 1-10
    engagement_level: float  # 1-10
    conclusion_strength: float  # 1-10
    pacing_analysis: Dict[str, Any]
    transition_quality: float  # 1-10
    
@dataclass
class ReadabilityMetrics:
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    avg_sentence_length: float
    avg_word_length: float
    complexity_score: float  # 1-10 (1 = very simple, 10 = very complex)
    
@dataclass
class ContentSuggestion:
    type: str  # 'structure', 'content', 'engagement', 'length'
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    specific_action: str
    estimated_impact: str  # 'high', 'medium', 'low'

@dataclass
class DetectedSection:
    title: str
    content: str
    word_count: int
    start_position: int
    end_position: int
    section_type: str  # 'intro', 'body', 'conclusion', 'transition'
    engagement_score: float  # 1-10

class ScriptAnalyzer:
    def __init__(self):
        self.openai_client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY_1") or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = create_openai_client(api_key)
            else:
                print("[WARNING] No OpenAI API key found for script analysis")
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenAI client for script analysis: {e}")
    
    async def analyze_script(self, script: str, filename: str = None) -> ScriptAnalysis:
        """Perform comprehensive script analysis"""
        print(f"[DEBUG] Starting script analysis for script of length {len(script)} characters")
        
        try:
            # Basic metrics
            word_count = len(script.split())
            char_count = len(script)
            
            # Perform different types of analysis
            structure_analysis = await self._analyze_structure(script)
            readability_metrics = self._calculate_readability(script)
            detected_sections = self._detect_sections(script)
            suggestions = await self._generate_suggestions(script, structure_analysis, readability_metrics)
            key_points = await self._extract_key_points(script)
            
            # Create analysis object
            analysis = ScriptAnalysis(
                structure_score=structure_analysis.overall_score,
                readability_score=readability_metrics.complexity_score,
                suggestions=[s.description for s in suggestions],
                key_points=key_points,
                detected_sections=[asdict(section) for section in detected_sections],
                metadata={
                    'filename': filename,
                    'word_count': word_count,
                    'char_count': char_count,
                    'structure_analysis': asdict(structure_analysis),
                    'readability_metrics': asdict(readability_metrics),
                    'detailed_suggestions': [asdict(s) for s in suggestions]
                }
            )
            
            print(f"[DEBUG] Script analysis completed successfully")
            return analysis
            
        except Exception as e:
            print(f"[ERROR] Script analysis failed: {e}")
            # Return basic analysis on error
            return ScriptAnalysis(
                structure_score=5.0,
                readability_score=5.0,
                suggestions=["Script analysis temporarily unavailable. Please try again."],
                key_points=["Analysis pending"],
                detected_sections=[],
                metadata={'error': str(e), 'word_count': len(script.split())}
            )
    
    async def _analyze_structure(self, script: str) -> StructureAnalysis:
        """Analyze script structure using AI"""
        if not self.openai_client:
            return self._fallback_structure_analysis(script)
        
        try:
            prompt = f"""
            Analyze this script's structure and provide scores (1-10) for each category:
            
            1. Overall Structure Score (1-10): How well-organized is the script?
            2. Hook Effectiveness (1-10): How compelling is the opening?
            3. Flow Quality (1-10): How smoothly does it transition between ideas?
            4. Engagement Level (1-10): How likely is it to keep viewers interested?
            5. Conclusion Strength (1-10): How satisfying is the ending?
            6. Transition Quality (1-10): How well does it connect different sections?
            7. Pacing Analysis: Is the pacing too fast, too slow, or just right?
            
            Provide your response in this JSON format:
            {{
                "overall_score": 8.5,
                "hook_effectiveness": 7.0,
                "flow_quality": 8.0,
                "engagement_level": 7.5,
                "conclusion_strength": 6.0,
                "transition_quality": 8.0,
                "pacing_analysis": {{
                    "overall_pacing": "slightly fast",
                    "intro_pacing": "good",
                    "body_pacing": "too fast",
                    "conclusion_pacing": "rushed"
                }}
            }}
            
            Script to analyze (first 2000 characters):
            {script[:2000]}...
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # Parse JSON response
            analysis_data = json.loads(response.choices[0].message.content)
            
            return StructureAnalysis(
                overall_score=analysis_data.get('overall_score', 5.0),
                hook_effectiveness=analysis_data.get('hook_effectiveness', 5.0),
                flow_quality=analysis_data.get('flow_quality', 5.0),
                engagement_level=analysis_data.get('engagement_level', 5.0),
                conclusion_strength=analysis_data.get('conclusion_strength', 5.0),
                pacing_analysis=analysis_data.get('pacing_analysis', {}),
                transition_quality=analysis_data.get('transition_quality', 5.0)
            )
            
        except Exception as e:
            print(f"[ERROR] AI structure analysis failed: {e}")
            return self._fallback_structure_analysis(script)
    
    def _fallback_structure_analysis(self, script: str) -> StructureAnalysis:
        """Fallback structure analysis without AI"""
        word_count = len(script.split())
        
        # Basic heuristics
        hook_score = 8.0 if len(script[:200].split()) > 20 else 5.0
        flow_score = 7.0 if word_count > 1000 else 5.0
        engagement_score = 6.0  # Default
        conclusion_score = 7.0 if script[-200:].strip() else 4.0
        
        return StructureAnalysis(
            overall_score=6.5,
            hook_effectiveness=hook_score,
            flow_quality=flow_score,
            engagement_level=engagement_score,
            conclusion_strength=conclusion_score,
            pacing_analysis={"overall_pacing": "moderate"},
            transition_quality=6.0
        )
    
    def _calculate_readability(self, script: str) -> ReadabilityMetrics:
        """Calculate readability metrics"""
        sentences = re.split(r'[.!?]+', script)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = script.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease
        if len(sentences) > 0 and len(words) > 0:
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            flesch_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            flesch_kincaid = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        else:
            flesch_ease = 50.0
            flesch_kincaid = 8.0
            avg_sentence_length = 0
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Complexity score (1-10, lower is simpler)
        complexity_score = min(10, max(1, (flesch_kincaid / 2) + (avg_word_length * 0.5)))
        
        return ReadabilityMetrics(
            flesch_reading_ease=max(0, min(100, flesch_ease)),
            flesch_kincaid_grade=max(0, flesch_kincaid),
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            complexity_score=complexity_score
        )
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simple heuristic)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _detect_sections(self, script: str) -> List[DetectedSection]:
        """Detect sections in the script"""
        sections = []
        words = script.split()
        
        # Simple section detection based on length
        if len(words) < 500:
            # Short script - treat as single section
            sections.append(DetectedSection(
                title="Main Content",
                content=script,
                word_count=len(words),
                start_position=0,
                end_position=len(script),
                section_type="body",
                engagement_score=6.0
            ))
        else:
            # Longer script - split into intro, body, conclusion
            intro_end = len(script) // 10  # First 10%
            conclusion_start = len(script) * 9 // 10  # Last 10%
            
            sections.extend([
                DetectedSection(
                    title="Introduction",
                    content=script[:intro_end],
                    word_count=len(script[:intro_end].split()),
                    start_position=0,
                    end_position=intro_end,
                    section_type="intro",
                    engagement_score=7.0
                ),
                DetectedSection(
                    title="Main Content",
                    content=script[intro_end:conclusion_start],
                    word_count=len(script[intro_end:conclusion_start].split()),
                    start_position=intro_end,
                    end_position=conclusion_start,
                    section_type="body",
                    engagement_score=6.0
                ),
                DetectedSection(
                    title="Conclusion",
                    content=script[conclusion_start:],
                    word_count=len(script[conclusion_start:].split()),
                    start_position=conclusion_start,
                    end_position=len(script),
                    section_type="conclusion",
                    engagement_score=6.5
                )
            ])
        
        return sections
    
    async def _generate_suggestions(self, script: str, structure: StructureAnalysis, readability: ReadabilityMetrics) -> List[ContentSuggestion]:
        """Generate improvement suggestions"""
        suggestions = []
        
        # Structure-based suggestions
        if structure.hook_effectiveness < 6.0:
            suggestions.append(ContentSuggestion(
                type="structure",
                priority="high",
                title="Improve Opening Hook",
                description="The opening could be more compelling to grab viewer attention immediately",
                specific_action="Start with a surprising fact, question, or bold statement",
                estimated_impact="high"
            ))
        
        if structure.flow_quality < 6.0:
            suggestions.append(ContentSuggestion(
                type="structure",
                priority="medium",
                title="Enhance Flow Between Sections",
                description="The transitions between ideas could be smoother",
                specific_action="Add transitional phrases and bridge sentences between main points",
                estimated_impact="medium"
            ))
        
        if structure.conclusion_strength < 6.0:
            suggestions.append(ContentSuggestion(
                type="structure",
                priority="medium",
                title="Strengthen Conclusion",
                description="The ending could be more impactful and memorable",
                specific_action="Add a clear call-to-action or memorable final thought",
                estimated_impact="medium"
            ))
        
        # Readability-based suggestions
        if readability.complexity_score > 8.0:
            suggestions.append(ContentSuggestion(
                type="content",
                priority="medium",
                title="Simplify Language",
                description="The script may be too complex for general audience",
                specific_action="Use shorter sentences and simpler vocabulary",
                estimated_impact="medium"
            ))
        
        if readability.avg_sentence_length > 20:
            suggestions.append(ContentSuggestion(
                type="content",
                priority="low",
                title="Shorten Sentences",
                description="Sentences are quite long and may be hard to follow",
                specific_action="Break long sentences into shorter, punchier ones",
                estimated_impact="low"
            ))
        
        # Length-based suggestions
        word_count = len(script.split())
        if word_count < 1000:
            suggestions.append(ContentSuggestion(
                type="length",
                priority="high",
                title="Expand Content",
                description="Script is quite short and may not provide enough value",
                specific_action="Add more examples, details, or supporting points",
                estimated_impact="high"
            ))
        elif word_count > 5000:
            suggestions.append(ContentSuggestion(
                type="length",
                priority="medium",
                title="Consider Shortening",
                description="Script is very long and may lose viewer attention",
                specific_action="Focus on the most important points and remove redundancy",
                estimated_impact="medium"
            ))
        
        # Engagement suggestions
        if structure.engagement_level < 7.0:
            suggestions.append(ContentSuggestion(
                type="engagement",
                priority="high",
                title="Increase Engagement Elements",
                description="Script could use more elements to maintain viewer interest",
                specific_action="Add rhetorical questions, surprising facts, or personal anecdotes",
                estimated_impact="high"
            ))
        
        return suggestions
    
    async def _extract_key_points(self, script: str) -> List[str]:
        """Extract key points from the script"""
        if not self.openai_client:
            return self._fallback_key_points(script)
        
        try:
            prompt = f"""
            Extract the 5-7 most important key points from this script. 
            Each key point should be a concise sentence that captures a main idea or takeaway.
            
            Format your response as a JSON array of strings:
            ["Key point 1", "Key point 2", "Key point 3", ...]
            
            Script (first 1500 characters):
            {script[:1500]}...
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            key_points = json.loads(response.choices[0].message.content)
            return key_points[:7]  # Limit to 7 points
            
        except Exception as e:
            print(f"[ERROR] AI key point extraction failed: {e}")
            return self._fallback_key_points(script)
    
    def _fallback_key_points(self, script: str) -> List[str]:
        """Fallback key point extraction without AI"""
        # Simple extraction based on sentence structure
        sentences = re.split(r'[.!?]+', script)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
        
        # Take first few sentences as key points
        key_points = sentences[:5]
        
        if not key_points:
            key_points = ["Content analysis pending", "Key points will be extracted during processing"]
        
        return key_points

# Global analyzer instance
script_analyzer = ScriptAnalyzer() 