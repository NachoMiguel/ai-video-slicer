try:
    import moviepy
    print(f"MoviePy version: {moviepy.__version__}")
    from moviepy.editor import VideoFileClip
    print("Successfully imported VideoFileClip")
except Exception as e:
    print(f"Error: {str(e)}")