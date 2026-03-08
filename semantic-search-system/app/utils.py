"""
utils.py
--------
Shared utilities: logging configuration and text cleaning.

Design decisions:
- A single module-level logger is configured here and imported by all other
  modules, so log format and level are consistent across the entire app.
- Text cleaning is deliberately aggressive for the 20-Newsgroups corpus, whose
  posts contain noisy headers/footers, quoted lines, email addresses etc that
  would pollute embeddings and cluster topics.
"""

import logging
import re
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return the root application logger.

    Uses a single StreamHandler writing to stdout so that log lines are
    visible in the uvicorn terminal without file rotation complexity.
    """
    logger = logging.getLogger("semantic_search")
    if logger.handlers:
        # Avoid adding duplicate handlers on module reload (e.g. uvicorn --reload)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Module-level singleton logger — import this everywhere.
logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

# Pre-compile patterns once at import time for efficiency.
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_NEWSGROUP_QUOTE = re.compile(r"^>.*$", re.MULTILINE)        # quoted reply lines
_RE_REPEATED_PUNCT = re.compile(r"[.\-_=*~]{3,}")               # -----, ===== etc.
_RE_WHITESPACE = re.compile(r"\s+")
_RE_NON_ASCII = re.compile(r"[^\x00-\x7F]+")                    # strip non-ASCII


def clean_text(text: str) -> str:
    """
    Clean a raw 20-Newsgroups document for embedding.

    Steps applied in order:
    1. Remove email addresses (privacy + noise).
    2. Remove URLs.
    3. Strip quoted-reply lines that start with '>'.
    4. Remove repeated punctuation / separator lines.
    5. Strip non-ASCII characters.
    6. Collapse excess whitespace.

    Args:
        text: Raw document string.

    Returns:
        Cleaned string, or empty string if nothing meaningful remains.
    """
    if not text:
        return ""

    text = _RE_EMAIL.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_NEWSGROUP_QUOTE.sub(" ", text)
    text = _RE_REPEATED_PUNCT.sub(" ", text)
    text = _RE_NON_ASCII.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def truncate_text(text: str, max_chars: int = 512) -> str:
    """
    Truncate text to *max_chars* characters, breaking at a word boundary.

    SentenceTransformers internally handles token limits, but pre-truncating
    avoids passing enormous strings to the tokeniser when documents are very
    long (e.g. some 20-Newsgroups posts with appended binaries).

    Args:
        text: Input string.
        max_chars: Maximum character count (default 512).

    Returns:
        Possibly truncated string.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to not cut mid-word
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated
