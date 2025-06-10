# 📚 implementation-plan.md

## Step-by-Step Action Plan

### Phase 1: MVP Core Features

1. **Set up project base**

   * Next.js frontend (with ShadCN UI, Tailwind)
   * Backend API scaffold (FastAPI or Express)

2. **Script Panel**

   * Input for YouTube URL
   * Prompt toggle + prompt textarea
   * Button to trigger transcription and rewrite
   * Connect Whisper + OpenAI/Anthropic logic

3. **Video Panel**

   * Upload (via uploadthing or react-dropzone) or enter 2 YouTube links
   * Temporary file storage
   * Connect scene extraction (PySceneDetect or custom ffmpeg logic)

4. **Scene-to-Script Matching Engine**

   * Basic keyword alignment from rewritten script to scene metadata
   * Match scenes and create mapping

5. **Video Reassembly (FFmpeg)**

   * Compose new video using selected scenes
   * Add transitions (if any)
   * Export final output file

6. **Preview + Download UI**

   * Video preview player (video.js or mux-player)
   * Download button

### Phase 2 (Optional/Next)

* Add captions overlay from rewritten script
* Add timeline scrubber
* Export .srt/.vtt for subtitles
* In-app onboarding or tooltips

## Timeline/Phases

* **Week 1**: UI scaffolding + Whisper/OpenAI integration
* **Week 2**: Upload + video parsing logic (scene splitting)
* **Week 3**: Script-matching engine + FFmpeg automation
* **Week 4**: Final assembly + testing + preview UI

## Team Setup Recommendations

* Solo builder (you) for now
* In future: bring in support dev for backend (if scaling cloud-side)

## Optional Tasks / Integrations

* Cloudflare R2 or S3 for persistent storage (Phase 3+)
* Web workers or queue system for long processing jobs
* Add presets for rewriting ("make funnier," "shorten to 2 min")
