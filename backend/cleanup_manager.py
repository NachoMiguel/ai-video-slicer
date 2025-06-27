#!/usr/bin/env python3
"""
Cleanup Manager for AI Video Slicer
Handles temporary file cleanup and prevents accumulation of segment files
"""

import os
import glob
import tempfile
import shutil
from typing import List, Optional
from pathlib import Path
import logging

try:
    from unicode_safe_logger import safe_print
except ImportError:
    def safe_print(*args, **kwargs):
        print(*args, **kwargs)

class CleanupManager:
    """Manages temporary files and cleanup operations"""
    
    def __init__(self):
        self.temp_dirs: List[str] = []
        self.temp_files: List[str] = []
        self.segment_files: List[str] = []
        
    def create_temp_dir(self, prefix: str = "ai_video_") -> str:
        """Create a temporary directory and track it for cleanup"""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.temp_dirs.append(temp_dir)
        safe_print(f"[CLEANUP] Created temp directory: {temp_dir}")
        return temp_dir
    
    def track_segment_file(self, file_path: str):
        """Track a segment file for cleanup"""
        if file_path and os.path.exists(file_path):
            self.segment_files.append(file_path)
            safe_print(f"[CLEANUP] Tracking segment file: {file_path}")
    
    def track_temp_file(self, file_path: str):
        """Track a temporary file for cleanup"""
        if file_path and os.path.exists(file_path):
            self.temp_files.append(file_path)
            safe_print(f"[CLEANUP] Tracking temp file: {file_path}")
    
    def cleanup_segments(self):
        """Clean up tracked segment files"""
        cleaned_count = 0
        for file_path in self.segment_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    safe_print(f"[CLEANUP] Removed segment: {file_path}")
                    cleaned_count += 1
            except Exception as e:
                safe_print(f"[CLEANUP] Failed to remove segment {file_path}: {e}")
        
        self.segment_files.clear()
        safe_print(f"[CLEANUP] Cleaned {cleaned_count} segment files")
        return cleaned_count
    
    def cleanup_temp_files(self):
        """Clean up tracked temporary files"""
        cleaned_count = 0
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    safe_print(f"[CLEANUP] Removed temp file: {file_path}")
                    cleaned_count += 1
            except Exception as e:
                safe_print(f"[CLEANUP] Failed to remove temp file {file_path}: {e}")
        
        self.temp_files.clear()
        safe_print(f"[CLEANUP] Cleaned {cleaned_count} temp files")
        return cleaned_count
    
    def cleanup_temp_dirs(self):
        """Clean up tracked temporary directories"""
        cleaned_count = 0
        for dir_path in self.temp_dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    safe_print(f"[CLEANUP] Removed temp directory: {dir_path}")
                    cleaned_count += 1
            except Exception as e:
                safe_print(f"[CLEANUP] Failed to remove temp directory {dir_path}: {e}")
        
        self.temp_dirs.clear()
        safe_print(f"[CLEANUP] Cleaned {cleaned_count} temp directories")
        return cleaned_count
    
    def cleanup_all(self):
        """Clean up all tracked files and directories"""
        safe_print("[CLEANUP] Starting complete cleanup...")
        
        segment_count = self.cleanup_segments()
        file_count = self.cleanup_temp_files()
        dir_count = self.cleanup_temp_dirs()
        
        total_cleaned = segment_count + file_count + dir_count
        safe_print(f"[CLEANUP] Complete cleanup finished: {total_cleaned} items removed")
        return total_cleaned
    
    @staticmethod
    def cleanup_orphaned_segments(directory: str = ".") -> int:
        """Clean up orphaned segment files in a directory"""
        safe_print(f"[CLEANUP] Scanning for orphaned segments in: {directory}")
        
        # Find all segment files with the MoviePy pattern
        patterns = [
            "segment_*TEMP_MPY*.mp4",
            "segment_*.mp4"
        ]
        
        orphaned_files = []
        for pattern in patterns:
            orphaned_files.extend(glob.glob(os.path.join(directory, pattern)))
        
        cleaned_count = 0
        for file_path in orphaned_files:
            try:
                file_size = os.path.getsize(file_path)
                # Remove files that are suspiciously small (likely failed extractions)
                if file_size < 1024:  # Less than 1KB
                    os.remove(file_path)
                    safe_print(f"[CLEANUP] Removed orphaned segment: {file_path} ({file_size} bytes)")
                    cleaned_count += 1
                else:
                    safe_print(f"[CLEANUP] Keeping segment with content: {file_path} ({file_size} bytes)")
            except Exception as e:
                safe_print(f"[CLEANUP] Failed to process {file_path}: {e}")
        
        safe_print(f"[CLEANUP] Orphaned segment cleanup complete: {cleaned_count} files removed")
        return cleaned_count
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatic cleanup"""
        self.cleanup_all()

# Global cleanup instance for the video processing pipeline
_global_cleanup_manager = None

def get_cleanup_manager() -> CleanupManager:
    """Get the global cleanup manager instance"""
    global _global_cleanup_manager
    if _global_cleanup_manager is None:
        _global_cleanup_manager = CleanupManager()
    return _global_cleanup_manager

def cleanup_orphaned_files():
    """Utility function to clean up orphaned files"""
    return CleanupManager.cleanup_orphaned_segments() 