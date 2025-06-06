import os
import tempfile
import json
import uuid
import asyncio
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from scenedetect import detect, ContentDetector
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import openai
import whisper
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/api/process")
async def process_videos(
    videos: List[UploadFile] = File(...),
    prompt: str = Form(...)
):
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded videos
            video_paths = []
            for video in videos:
                temp_path = os.path.join(temp_dir, video.filename)
                with open(temp_path, "wb") as f:
                    f.write(await video.read())
                video_paths.append(temp_path)

            # Detect scenes in each video
            scenes = []
            for video_path in video_paths:
                scene_list = detect(video_path, ContentDetector())
                scenes.extend([(video_path, scene) for scene in scene_list])

            # Generate script using OpenAI
            script = generate_script(prompt, scenes)

            # TODO: Implement video recomposition based on script
            # This is a placeholder for the actual implementation
            output_path = os.path.join(temp_dir, "output.mp4")
            
            # For now, just concatenate the videos
            clips = [VideoFileClip(path) for path in video_paths]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path)

            # Read the output file
            with open(output_path, "rb") as f:
                output_data = f.read()

            return {
                "status": "success",
                "message": "Video processing completed",
                "data": output_data
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def generate_script(prompt: str, scenes: List[tuple]) -> str:
    """Generate a script using OpenAI based on the scenes and prompt."""
    try:
        # Create a description of the scenes
        scene_descriptions = []
        for video_path, scene in scenes:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scene_descriptions.append(
                f"Scene from {os.path.basename(video_path)}: {start_time:.2f}s to {end_time:.2f}s"
            )

        # Generate script using OpenAI
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a video editor creating a script for recomposing video scenes."},
                {"role": "user", "content": f"Prompt: {prompt}\n\nAvailable scenes:\n" + "\n".join(scene_descriptions)}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating script: {e}")
        return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 