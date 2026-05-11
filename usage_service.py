"""
services/usage_service.py
Tracks token usage per user and provides settings management helpers.
"""

# ── Token usage tracking ──────────────────────────────────────────────────────

class UsageTracker:
    """Per-user token usage tracker."""

    def __init__(self):
        self._usage = {}  # {user_id: int}

    def add_usage(self, user_id, tokens):
        """Add tokens to a user's running total. tokens must be >= 0."""
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("tokens must be a non-negative integer.")
        self._usage[user_id] = self._usage.get(user_id, 0) + tokens

    def get_usage(self, user_id):
        """Return total tokens used by user_id (0 if never seen)."""
        return self._usage.get(user_id, 0)

    def reset_usage(self, user_id):
        """Reset a user's token count to 0."""
        self._usage[user_id] = 0

    def get_all_usage(self):
        """Return a copy of the full usage dict."""
        return dict(self._usage)


# ── Settings helpers ──────────────────────────────────────────────────────────

DEFAULT_SETTINGS = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.9,
    "top_p": 1.0,
    "max_tokens": 1500,
    "low_usage_mode": False,
}


def get_default_settings():
    """Return a fresh copy of the default settings dict."""
    return dict(DEFAULT_SETTINGS)


def apply_setting(settings, key, value):
    """Update settings[key] = value if key is a known setting.
    Raises KeyError for unknown keys."""
    if key not in DEFAULT_SETTINGS:
        raise KeyError(f"Unknown setting: '{key}'.")
    settings[key] = value
    return settings


def is_low_usage_mode(settings):
    """Return the boolean value of the low_usage_mode setting."""
    return bool(settings.get("low_usage_mode", False))


def get_model_for_settings(settings):
    """Return 'gpt-3.5-turbo' when low_usage_mode is on, otherwise the
    configured model — mirroring GPTDiscord's /system settings low_usage_mode."""
    if is_low_usage_mode(settings):
        return "gpt-3.5-turbo"
    return settings.get("model", "gpt-3.5-turbo")
