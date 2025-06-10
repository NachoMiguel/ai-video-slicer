# AI Video Slicer & Recomposer

An AI-powered video editor that automatically slices and recomposes videos based on prompts.

## Features

- Upload 2-3 videos via drag-and-drop
- Automatic scene detection
- AI-powered script generation
- Video recomposition with transitions and captions
- Modern, responsive UI

## Tech Stack

- Next.js 14 (React + TypeScript)
- Tailwind CSS + ShadCN UI
- React Dropzone for file uploads
- Whisper for transcription
- OpenAI/Anthropic for script generation
- PySceneDetect for scene detection
- FFmpeg/MoviePy for video processing

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Development

This is an MVP version with the following limitations:
- No database integration
- All processing happens in-memory
- No persistent storage
- Limited to 2-3 video uploads

## License

This project is intended for private MVP prototyping and internal testing. 