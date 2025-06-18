# AI Video Slicer Prompts

This file contains all the prompts used by the AI Video Slicer application.

## Default Script Generation Prompt

```
I'm learning YouTube automation, and I need your help creating high-retention scripts that keep viewers interested and clicking. I want the scripts to have a natural flow and avoid anything that feels repetitive, robotic, or slow. Here's how we'll do it - I will send you the video transcript of one of my competitors:

CRITICAL INSTRUCTIONS: You MUST complete ALL steps in this single response. Do not stop after any step - continue until you have written the complete 20,000+ character script.

YOUR EXACT PROCESS (Complete ALL steps in order):

STEP 1: Create 10 Detailed Bullet Points
Create exactly 10 bullet points that outline the script structure. Each bullet point must include:
- The section title
- 2-3 sentences describing what content/details that section will cover
- Key talking points and elements to include
- The emotional tone and engagement strategy for that section

STEP 2-11: Write Each Script Section Immediately
After creating the bullet points, immediately write each section:
- Write "=== POINT 1: [Title] ===" then write that complete section
- Write "=== POINT 2: [Title] ===" then write that complete section  
- Continue for ALL 10 points without stopping

STEP 12: Create Final TTS-Ready Script
After writing all 10 sections, combine them into one continuous paragraph script WITHOUT section titles or headers. This final script will be sent to ElevenLabs TTS service, so it must be clean paragraph format with smooth transitions between sections.

CONTENT REQUIREMENTS:
- Point 1 (Intro): 300-500 characters - strong hook, mysterious opening
- Points 2-10: Each 2,000-2,500 characters - detailed content with engagement
- Total Target: 20,000-25,000 characters minimum
- Tone: Engaging, mysterious, conversational like sharing secrets with a friend
- Structure: 10 distinct sections that flow naturally together

FORMAT REQUIREMENTS:
- Write in paragraph form, no stage directions
- Use dramatic language: "shocking," "exposed," "revealed," "untold truth"
- Include smooth transitions between sections  
- Censor sensitive content appropriately
- Keep sentences short and impactful
- Add suspense hooks at end of each section

ENGAGEMENT TECHNIQUES:
- Focus on controversial, shocking, or unknown elements
- Include recent rumors, controversies, and speculations
- Highlight challenges, untold stories, conflicts viewers don't know
- Use varied dramatic language to avoid repetition
- Make each section introduce new, interesting information
- Build curiosity throughout the entire script

MANDATORY COMPLETION: You must write the complete script for all 10 sections. Do not stop after bullet points or after just a few sections. Complete the entire 20,000+ character script in this single response.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

[DETAILED BULLET POINTS - 10 Points with Descriptions]
1. [Section Title]: [2-3 sentences describing content, key points, tone]
2. [Section Title]: [2-3 sentences describing content, key points, tone]
...continue for all 10 points...

[SECTION-BY-SECTION WRITING]

=== POINT 1: [Title] ===
[Write complete intro section - 300-500 characters]

=== POINT 2: [Title] ===
[Write complete section - 2,000-2,500 characters]

=== POINT 3: [Title] ===
[Write complete section - 2,000-2,500 characters]

...continue for ALL 10 points without stopping...

=== POINT 10: [Title] ===
[Write complete conclusion section - 2,000-2,500 characters]

[FINAL TTS-READY SCRIPT - PARAGRAPH FORMAT]
[Combine all 10 sections into one continuous paragraph script WITHOUT section titles, headers, or separators. This is the final script that will be sent to ElevenLabs TTS service. Ensure smooth transitions between sections so it flows as one cohesive narrative. Must be 20,000+ characters.]

REMEMBER: Complete ALL sections AND provide the final paragraph-format script for TTS. The script must reach 20,000+ characters total.
```

## Scene Analysis Prompt

```
You are a video editor creating a script for recomposing video scenes. Analyze the available scenes and create a narrative that flows naturally while maintaining visual coherence.
```

## Additional Prompts

Add more prompts here as needed for future features.
