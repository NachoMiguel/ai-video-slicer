# 📝 app-flow-pages-and-roles.md

## Core Pages

### 1. **Script Panel Page**

* Paste YouTube video URL
* Toggle between default/custom prompt
* Textarea for user-defined prompt
* Button: "Transcribe & Rewrite"
* Displays rewritten script below

### 2. **Video Input Panel**

* Upload 2 local video files *or* enter 2 YouTube links
* Displays thumbnails or file names
* Button: "Generate Edited Video"
* Background processing of scenes

### 3. **Output Preview Page**

* Embedded video player to preview result
* Button: "Download Final Video"
* Optional: list of matched scenes with timestamps

## User Roles and Access Rights

* **Editor (you)**: full access to all panels and features
* No user authentication or multi-role logic in MVP

## How Users Move Through the App

1. Open the app (desktop only)
2. Paste a YouTube link in the Script Panel
3. Choose default or custom prompt → rewrite script
4. Upload or paste 2 video sources in Video Panel
5. Click "Generate Edited Video"
6. Wait for processing → see output preview
7. Click download to save the result

Each step flows linearly with visual feedback (loading spinners, progress bars, etc.)
