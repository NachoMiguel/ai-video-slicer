@echo off
echo Testing YouTube URL processing...
curl -X POST -d "youtube_url=https://www.youtube.com/watch?v=1bSmC_aO2bI" -d "use_default_prompt=true" -d "save_script=true" http://localhost:8000/api/generate-script
pause 