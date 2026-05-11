"""
models/openai_model.py
Core data model and utility functions for GPTDiscord.
Handles conversation history, token estimation, message building,
parameter validation, and prompt truncation.
"""

import re

# ── Constants ────────────────────────────────────────────────────────────────

MAX_TOKENS = 4096          # Hard cap for GPT-3.5-turbo context window
MAX_CONVERSATION_LENGTH = 100   # Max messages stored in a conversation
VALID_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0
TOP_P_MIN = 0.0
TOP_P_MAX = 1.0
MAX_DISCORD_MESSAGE_LENGTH = 2000


# ── Parameter validation ──────────────────────────────────────────────────────

def validate_temperature(value):
    """Return True if value is a valid temperature (float between 0.0 and 2.0)."""
    try:
        f = float(value)
        return TEMPERATURE_MIN <= f <= TEMPERATURE_MAX
    except (TypeError, ValueError):
        return False


def validate_top_p(value):
    """Return True if value is a valid top_p (float between 0.0 and 1.0)."""
    try:
        f = float(value)
        return TOP_P_MIN <= f <= TOP_P_MAX
    except (TypeError, ValueError):
        return False


def validate_model(model_name):
    """Return True if model_name is one of the supported model identifiers."""
    return isinstance(model_name, str) and model_name in VALID_MODELS


def validate_max_tokens(value):
    """Return True if value is a positive integer within the context window."""
    try:
        n = int(value)
        return 1 <= n <= MAX_TOKENS
    except (TypeError, ValueError):
        return False


# ── Token estimation ──────────────────────────────────────────────────────────

def estimate_tokens(text):
    """Rough token count: ~4 characters per token (OpenAI approximation).
    Returns 0 for empty or non-string input."""
    if not isinstance(text, str) or not text:
        return 0
    return max(1, len(text) // 4)


def estimate_conversation_tokens(messages):
    """Sum estimated tokens across a list of message dicts.
    Each message is {"role": str, "content": str}."""
    if not isinstance(messages, list):
        return 0
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


# ── Message construction ──────────────────────────────────────────────────────

def build_message(role, content):
    """Return a single OpenAI chat message dict.
    Raises ValueError for invalid role or empty content."""
    valid_roles = {"system", "user", "assistant"}
    if role not in valid_roles:
        raise ValueError(f"Invalid role '{role}'. Must be one of {valid_roles}.")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content must be a non-empty string.")
    return {"role": role, "content": content}


def build_system_prompt(instruction):
    """Wrap a plain instruction string into a system-role message dict."""
    return build_message("system", instruction)


# ── Conversation management ───────────────────────────────────────────────────

def add_message_to_history(history, role, content):
    """Append a new message to the conversation history list.
    Enforces MAX_CONVERSATION_LENGTH by dropping the oldest non-system
    message when the limit is exceeded.
    Returns the updated history."""
    if not isinstance(history, list):
        raise TypeError("history must be a list.")
    msg = build_message(role, content)
    history.append(msg)
    # Trim oldest non-system message if over limit
    while len(history) > MAX_CONVERSATION_LENGTH:
        for i, m in enumerate(history):
            if m["role"] != "system":
                history.pop(i)
                break
    return history


def clear_conversation(history):
    """Remove all non-system messages from history, preserving system prompts."""
    if not isinstance(history, list):
        raise TypeError("history must be a list.")
    return [m for m in history if m["role"] == "system"]


def get_last_message(history):
    """Return the last message dict in history, or None if history is empty."""
    if not history:
        return None
    return history[-1]


# ── Prompt / response utilities ───────────────────────────────────────────────

def truncate_to_token_limit(text, token_limit):
    """Truncate text so that estimate_tokens(result) <= token_limit.
    Returns the truncated string (may be shorter than the limit)."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if token_limit <= 0:
        return ""
    # Each token ≈ 4 chars
    char_limit = token_limit * 4
    return text[:char_limit]


def split_into_discord_messages(text, max_length=MAX_DISCORD_MESSAGE_LENGTH):
    """Split a long string into a list of chunks, each at most max_length chars.
    Attempts to split on newlines first; falls back to hard splits."""
    if not text:
        return []
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        split_pos = text.rfind("\n", 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


def sanitize_prompt(prompt):
    """Strip leading/trailing whitespace and collapse multiple internal
    spaces/newlines into a single space. Returns the cleaned string."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")
    return re.sub(r"\s+", " ", prompt).strip()


def format_model_response(response_text, user_name=None):
    """Prepend an optional mention and strip surrounding whitespace
    from the model's response text."""
    text = response_text.strip() if isinstance(response_text, str) else ""
    if user_name:
        return f"**{user_name}**: {text}"
    return text
