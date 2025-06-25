import logging
import sys
import re
from typing import Any

# Unicode to ASCII mapping for all characters used in the app
UNICODE_REPLACEMENTS = {
    # Checkmarks and validation
    '✓': '[OK]',
    '✅': '[SUCCESS]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    
    # Emojis used in logging
    '🔍': '[SEARCH]',
    '📊': '[STATS]',
    '💳': '[PAID]',
    '🆓': '[FREE]',
    '🚫': '[BLOCKED]',
    '💤': '[INACTIVE]',
    '💎': '[CREDITS]',
    '🧪': '[TEST]',
    '📧': '[EMAIL]',
    '🔑': '[KEY]',
    '📈': '[RATE]',
    '🔄': '[ROTATE]',
    '🚀': '[START]',
    '💥': '[CRASH]',
    '🔓': '[UNLOCK]',
    '📉': '[DOWN]',
    '💡': '[TIP]',
    '🎉': '[DONE]',
    '💰': '[COST]',
    '🎯': '[TARGET]',
    '🔒': '[SECURE]',
    '💪': '[POWER]',
    
    # Video processing emojis
    '🎬': '[VIDEO]',
    '🎨': '[EDIT]',
    '🔊': '[AUDIO]',
    '💾': '[SAVE]',
    '📁': '[FOLDER]',
    '📐': '[SIZE]',
    '⭐': '[QUALITY]',
    '🔗': '[CONCAT]',
    
    # Other symbols
    '•': '-',
    '×': 'x',
    '⚡': '[FAST]',
    '🤖': '[AI]',
    '🔧': '[TOOL]',
    '📹': '[CAM]',
    '✕': 'X',
}

def safe_unicode_replace(text: str) -> str:
    """
    Replace all Unicode characters with ASCII equivalents.
    This prevents 'charmap' codec errors on Windows.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Replace known Unicode characters
    for unicode_char, ascii_replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, ascii_replacement)
    
    # Replace any remaining non-ASCII characters with [?]
    text = re.sub(r'[^\x00-\x7F]', '[?]', text)
    
    return text

def safe_print(*args, **kwargs):
    """
    Unicode-safe print function that converts all Unicode to ASCII.
    """
    safe_args = []
    for arg in args:
        if isinstance(arg, str):
            safe_args.append(safe_unicode_replace(arg))
        else:
            safe_args.append(safe_unicode_replace(str(arg)))
    
    print(*safe_args, **kwargs)

def safe_log(logger: logging.Logger, level: str, message: str, *args, **kwargs):
    """
    Unicode-safe logging function.
    """
    safe_message = safe_unicode_replace(message)
    getattr(logger, level.lower())(safe_message, *args, **kwargs)

class UnicodeFilter(logging.Filter):
    """
    Logging filter that converts Unicode characters to ASCII.
    """
    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = safe_unicode_replace(record.msg)
        if hasattr(record, 'args') and record.args:
            safe_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    safe_args.append(safe_unicode_replace(arg))
                else:
                    safe_args.append(arg)
            record.args = tuple(safe_args)
        return True

def setup_unicode_safe_logging():
    """
    Setup logging with Unicode filtering for Windows compatibility.
    """
    # Add Unicode filter to all loggers
    unicode_filter = UnicodeFilter()
    
    # Get root logger and add filter
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(unicode_filter)
    
    # Also add to any existing loggers
    for logger_name in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.addFilter(unicode_filter)

# Convenience functions for common logging patterns
def log_success(message: str):
    safe_print(f"[SUCCESS] {message}")

def log_error(message: str):
    safe_print(f"[ERROR] {message}")

def log_warning(message: str):
    safe_print(f"[WARNING] {message}")

def log_info(message: str):
    safe_print(f"[INFO] {message}")

def log_debug(message: str):
    safe_print(f"[DEBUG] {message}") 