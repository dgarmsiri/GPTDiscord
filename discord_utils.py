"""
utils/discord_utils.py
Utility helpers for message formatting and command parsing used
throughout the GPTDiscord cogs.
"""

import re

MAX_EMBED_DESCRIPTION = 4096
DISCORD_MAX_MESSAGE   = 2000


def is_valid_prompt(text):
    """Return True if text is a non-empty, non-whitespace-only string."""
    return isinstance(text, str) and bool(text.strip())


def clean_discord_mention(text):
    """Remove Discord user/role/channel mention syntax (<@123>, <#123>, etc.)
    and return the cleaned string."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"<[@#!&][0-9]+>", "", text)
    return re.sub(r" +", " ", cleaned).strip()


def extract_command_args(content, prefix="/"):
    """Split a raw Discord message into (command, args_string).
    E.g. '/gpt ask hello world' → ('gpt ask', 'hello world').
    Returns (None, None) if the message doesn't start with the prefix."""
    if not isinstance(content, str) or not content.startswith(prefix):
        return None, None
    body = content[len(prefix):]
    parts = body.split(" ", 2)
    if len(parts) == 1:
        return parts[0], ""
    return " ".join(parts[:2]), parts[2] if len(parts) > 2 else ""


def format_error_message(error):
    """Return a user-facing error string with consistent formatting."""
    return f"⚠️ An error occurred: {str(error)}"


def chunk_text(text, max_size=DISCORD_MAX_MESSAGE):
    """Split text into a list of strings each at most max_size characters.
    Prefers splitting on whitespace boundaries."""
    if not text:
        return []
    if len(text) <= max_size:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_size:
            chunks.append(text)
            break
        cut = text.rfind(" ", 0, max_size)
        if cut == -1:
            cut = max_size
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    return chunks


def is_within_token_budget(estimated_tokens, budget):
    """Return True if estimated_tokens is within the given budget."""
    return isinstance(estimated_tokens, int) and estimated_tokens <= budget
