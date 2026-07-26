"""
utils.py
---------
Common utility functions used throughout VisionPro Studio.
"""

from pathlib import Path
from datetime import datetime
import cv2
import sys


# ==============================================================
# Supported Extensions
# ==============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".gif"
}


# ==============================================================
# Directory & File Utilities
# ==============================================================

def ensure_directory(directory):
    """
    Create directory if it doesn't exist.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def is_supported_file(filepath):
    """
    Check whether file extension is supported.
    """
    return Path(filepath).suffix.lower() in SUPPORTED_EXTENSIONS


def save_image(image, filepath):
    """
    Save image safely.
    """
    ensure_directory(Path(filepath).parent)
    cv2.imwrite(str(filepath), image)


# ==============================================================
# Image Utilities
# ==============================================================

def resize_keep_aspect(image, width=None, height=None):
    """
    Resize image while maintaining aspect ratio.
    """
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None:
        ratio = width / w
        dimension = (width, int(h * ratio))
    else:
        ratio = height / h
        dimension = (int(w * ratio), height)

    return cv2.resize(image, dimension)


# ==============================================================
# Time Utilities
# ==============================================================

def timestamp():
    """
    Current timestamp string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_filename():
    """
    Timestamp string safe for filenames.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ==============================================================
# Console Display Utilities
# ==============================================================

class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def print_header(title):
    """
    Display formatted console header.
    """
    width = 60
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title.center(width)}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * width}{Colors.RESET}")


def _safe_print(text):
    """Print with fallback for Windows encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Strip ANSI codes and use plain text
        import re
        clean = re.sub(r'\033\[[0-9;]*m', '', text)
        print(clean.encode('ascii', 'replace').decode('ascii'))


def print_success(message):
    """
    Print a success message with green checkmark.
    """
    _safe_print(f"{Colors.GREEN}[+] {message}{Colors.RESET}")


def print_error(message):
    """
    Print an error message with red cross.
    """
    _safe_print(f"{Colors.RED}[x] {message}{Colors.RESET}")


def print_warning(message):
    """
    Print a warning message.
    """
    _safe_print(f"{Colors.YELLOW}[!] {message}{Colors.RESET}")


def print_info(message):
    """
    Print an info message.
    """
    _safe_print(f"{Colors.BLUE}[*] {message}{Colors.RESET}")


def print_stage(stage_name):
    """
    Print a processing stage header.
    """
    _safe_print(f"\n{Colors.MAGENTA}> {stage_name}...{Colors.RESET}")


# ==============================================================
# File Size Formatting
# ==============================================================

def format_file_size(size_bytes):
    """
    Format file size in human-readable form.

    Parameters
    ----------
    size_bytes : int
        File size in bytes.

    Returns
    -------
    str
        Formatted size string (e.g., '1.5 MB').
    """

    if size_bytes < 1024:
        return f"{size_bytes} B"

    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"

    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"

    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


# ==============================================================
# Progress Bar
# ==============================================================

def progress_bar(current, total, prefix="", bar_length=40):
    """
    Display a simple progress bar in the console.

    Parameters
    ----------
    current : int
        Current progress value.
    total : int
        Total value.
    prefix : str
        Text to display before the bar.
    bar_length : int
        Length of the progress bar.
    """

    if total == 0:
        return

    fraction = current / total

    filled = int(bar_length * fraction)

    bar = "#" * filled + "-" * (bar_length - filled)

    percent = f"{fraction * 100:.1f}%"

    line = f"\r{prefix} |{bar}| {percent} [{current}/{total}]"

    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        sys.stdout.write(line.encode('ascii', 'replace').decode('ascii'))

    if current == total:
        sys.stdout.write("\n")

    sys.stdout.flush()


# ==============================================================
# Table Formatting
# ==============================================================

def format_table(data, headers=None, col_width=25):
    """
    Format a dictionary or list of tuples as a simple table.

    Parameters
    ----------
    data : dict or list
        Data to format.
    headers : tuple or None
        Column headers.
    col_width : int
        Column width.

    Returns
    -------
    str
        Formatted table string.
    """

    lines = []

    if headers:
        header_line = "  ".join(
            h.ljust(col_width) for h in headers
        )
        lines.append(header_line)
        lines.append("-" * len(header_line))

    if isinstance(data, dict):
        for key, value in data.items():
            lines.append(
                f"{str(key):<{col_width}}  {str(value)}"
            )

    elif isinstance(data, list):
        for row in data:
            if isinstance(row, (list, tuple)):
                lines.append(
                    "  ".join(
                        str(v).ljust(col_width)
                        for v in row
                    )
                )
            else:
                lines.append(str(row))

    return "\n".join(lines)