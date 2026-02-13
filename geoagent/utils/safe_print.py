"""
Safe printing utility for handling Unicode characters in Windows console output.

This module provides replacement functions for print() and logging that gracefully
handle Unicode characters that can't be encoded in Windows cp1252 console encoding.

Usage:
    from utils.safe_print import safe_print

    # Instead of print()
    safe_print("✅ Success message")

    # Or replace print globally in a module
    from utils.safe_print import safe_print as print
"""

import sys


def _sanitize_unicode(text):
    """
    Replace Unicode emoji/symbols with ASCII equivalents for Windows console.

    Args:
        text: String that may contain Unicode characters

    Returns:
        String with Unicode characters replaced by ASCII equivalents
    """
    replacements = {
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '⚠': '[WARNING]',
        '✓': '[OK]',
        '✗': '[FAIL]',
        '►': '>',
        '◄': '<',
        '▲': '^',
        '▼': 'v',
        '●': '*',
        '○': 'o',
        '■': '#',
        '□': '-',
        '├': '|',
        '└': '\\',
        '│': '|',
        '─': '-',
        '┌': '+',
        '┐': '+',
        '┘': '+',
        '┴': '+',
        '┬': '+',
        '┤': '+',
        '🔍': '[SEARCH]',
        '🔧': '[CONFIG]',
        '🎯': '[TARGET]',
        '💡': '[INFO]',
        '📊': '[DATA]',
        '🚨': '[ALERT]',
        '⏳': '[WAIT]',
        '🔄': '[PROCESS]',
    }

    result = str(text)
    for unicode_char, ascii_char in replacements.items():
        result = result.replace(unicode_char, ascii_char)

    return result


def safe_print(*args, **kwargs):
    """
    Safe replacement for print() that handles Unicode encoding errors.

    Automatically sanitizes Unicode characters that can't be encoded by the
    console's encoding (typically cp1252 on Windows).

    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print()
    """
    # Get the output file (default is stdout)
    file = kwargs.get('file', sys.stdout)

    try:
        # Try normal print first
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # If encoding fails, sanitize and retry
        sanitized_args = [_sanitize_unicode(arg) for arg in args]
        print(*sanitized_args, **kwargs)


def safe_format(text):
    """
    Safely format a string for console output.

    Args:
        text: String that may contain Unicode characters

    Returns:
        String safe for console output
    """
    return _sanitize_unicode(text)


class SafeLogger:
    """
    Logger wrapper that handles Unicode encoding gracefully.

    Usage:
        logger = SafeLogger()
        logger.info("✅ Operation successful")
        logger.warning("⚠️ Potential issue")
        logger.error("❌ Operation failed")
    """

    def __init__(self, prefix=""):
        self.prefix = prefix

    def _log(self, level, message, *args):
        """Internal logging method"""
        if args:
            message = message % args

        full_message = f"{self.prefix}{level}: {message}" if self.prefix else f"{level}: {message}"
        safe_print(full_message)

    def info(self, message, *args):
        """Log info message"""
        self._log("INFO", message, *args)

    def success(self, message, *args):
        """Log success message"""
        self._log("SUCCESS", message, *args)

    def warning(self, message, *args):
        """Log warning message"""
        self._log("WARNING", message, *args)

    def error(self, message, *args):
        """Log error message"""
        self._log("ERROR", message, *args)

    def debug(self, message, *args):
        """Log debug message"""
        self._log("DEBUG", message, *args)


# Create a default logger instance
default_logger = SafeLogger()


# Convenience functions
def log_success(message, *args):
    """Log a success message"""
    default_logger.success(message, *args)


def log_warning(message, *args):
    """Log a warning message"""
    default_logger.warning(message, *args)


def log_error(message, *args):
    """Log an error message"""
    default_logger.error(message, *args)


def log_info(message, *args):
    """Log an info message"""
    default_logger.info(message, *args)
