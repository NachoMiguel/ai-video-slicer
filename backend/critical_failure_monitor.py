#!/usr/bin/env python3
"""
Critical Failure Monitor - Automatically shuts down servers when critical failures are detected
"""
import os
import sys
import signal
import logging
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Critical failure patterns that should trigger immediate shutdown
CRITICAL_FAILURE_PATTERNS = [
    # Video processing failures
    "'NoneType' object has no attribute 'get_frame'",
    "No valid video clips after validation",
    "Simple assembly failed",
    "All video write attempts failed",
    
    # Unicode encoding crashes
    "'charmap' codec can't encode character",
    "UnicodeEncodeError",
    
    # Memory/resource failures
    "MemoryError",
    "Out of memory",
    "Disk space",
    
    # API rate limiting/credit exhaustion
    "Rate limit exceeded",
    "Quota exceeded",
    "Credits exhausted",
    "Too many requests"
]

# Failure tracking
failure_counts = {}
last_failure_time = {}
shutdown_triggered = False

class CriticalFailureMonitor:
    def __init__(self, max_failures: int = 3, time_window_minutes: int = 5):
        self.max_failures = max_failures
        self.time_window = timedelta(minutes=time_window_minutes)
        self.logger = logging.getLogger(__name__)
        
    def check_for_critical_failure(self, log_message: str, error: Exception = None) -> bool:
        """
        Check if a log message or error indicates a critical failure
        Returns True if server should be shut down
        """
        global shutdown_triggered
        
        if shutdown_triggered:
            return True
            
        # Convert error to string if provided
        if error:
            error_str = str(error)
            log_message = f"{log_message} - {error_str}"
        
        # Check for critical failure patterns
        for pattern in CRITICAL_FAILURE_PATTERNS:
            if pattern.lower() in log_message.lower():
                self.logger.error(f"[CRITICAL] Critical failure pattern detected: {pattern}")
                self.logger.error(f"[CRITICAL] Full message: {log_message}")
                
                # Track failure count
                current_time = datetime.now()
                
                if pattern not in failure_counts:
                    failure_counts[pattern] = 0
                    last_failure_time[pattern] = current_time
                
                # Reset count if outside time window
                if current_time - last_failure_time[pattern] > self.time_window:
                    failure_counts[pattern] = 0
                
                failure_counts[pattern] += 1
                last_failure_time[pattern] = current_time
                
                self.logger.error(f"[CRITICAL] Failure count for '{pattern}': {failure_counts[pattern]}/{self.max_failures}")
                
                # Trigger shutdown if threshold exceeded
                if failure_counts[pattern] >= self.max_failures:
                    self.logger.error(f"[CRITICAL] Maximum failures reached for pattern: {pattern}")
                    self.trigger_emergency_shutdown(f"Critical failure: {pattern}")
                    return True
                    
                # For certain patterns, shut down immediately
                immediate_shutdown_patterns = [
                    "'NoneType' object has no attribute 'get_frame'",
                    "Simple assembly failed",
                    "No valid video clips after validation"
                ]
                
                if any(immediate in pattern for immediate in immediate_shutdown_patterns):
                    self.logger.error(f"[CRITICAL] Immediate shutdown triggered for: {pattern}")
                    self.trigger_emergency_shutdown(f"Immediate critical failure: {pattern}")
                    return True
        
        return False
    
    def trigger_emergency_shutdown(self, reason: str):
        """
        Trigger emergency shutdown of the server
        """
        global shutdown_triggered
        
        if shutdown_triggered:
            return
            
        shutdown_triggered = True
        
        self.logger.error(f"[EMERGENCY SHUTDOWN] Reason: {reason}")
        self.logger.error(f"[EMERGENCY SHUTDOWN] Timestamp: {datetime.now().isoformat()}")
        self.logger.error(f"[EMERGENCY SHUTDOWN] Shutting down server to prevent credit waste...")
        
        # Log shutdown reason to file
        try:
            with open("emergency_shutdown.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - EMERGENCY SHUTDOWN: {reason}\n")
        except:
            pass
        
        # Give a moment for logs to flush
        import time
        time.sleep(1)
        
        # Force shutdown
        try:
            # Try graceful shutdown first
            os.kill(os.getpid(), signal.SIGTERM)
        except:
            try:
                # Force kill if graceful fails
                os.kill(os.getpid(), signal.SIGKILL)
            except:
                # Last resort
                sys.exit(1)

# Global monitor instance
monitor = CriticalFailureMonitor()

def check_critical_failure(message: str, error: Exception = None) -> bool:
    """
    Convenience function to check for critical failures
    """
    return monitor.check_for_critical_failure(message, error)

def emergency_shutdown(reason: str):
    """
    Convenience function to trigger emergency shutdown
    """
    monitor.trigger_emergency_shutdown(reason)

# Enhanced logging handler that checks for critical failures
class CriticalFailureHandler(logging.Handler):
    def emit(self, record):
        try:
            message = self.format(record)
            
            # Check for critical failures in log messages
            if record.levelno >= logging.ERROR:
                exception = getattr(record, 'exc_info', None)
                error = exception[1] if exception and len(exception) > 1 else None
                
                monitor.check_for_critical_failure(message, error)
                
        except Exception:
            pass  # Don't let monitoring interfere with normal logging

def setup_critical_failure_monitoring():
    """
    Setup critical failure monitoring on the root logger
    """
    handler = CriticalFailureHandler()
    handler.setLevel(logging.ERROR)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    logging.info("[MONITOR] Critical failure monitoring enabled")

if __name__ == "__main__":
    # Test the monitor
    setup_critical_failure_monitoring()
    
    # Simulate some failures
    logging.error("Test error: 'NoneType' object has no attribute 'get_frame'")
    logging.error("Test error: Simple assembly failed: No valid video clips after validation") 