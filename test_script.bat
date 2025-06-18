@echo off
echo Testing script generation...
curl.exe -X POST "http://127.0.0.1:8000/api/generate-script" -F "youtube_url=https://www.youtube.com/watch?v=1bSmC_aO2bI" -F "use_default_prompt=true" -F "save_script=true"
pause 