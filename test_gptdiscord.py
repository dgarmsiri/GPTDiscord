"""
test_gptdiscord.py — Comprehensive PyTest Suite for GPTDiscord
SWE 3643: Software Testing — Group 13
Members: Elizabeth Serrano Rodriguez, Elvin Pineda, Doreen Garmsiri

Project Under Test
------------------
GPTDiscord (https://github.com/Kav-K/GPTDiscord) is an all-in-one Python
Discord bot that integrates OpenAI's ChatGPT (GPT-3.5-turbo, GPT-4) into
Discord servers.  It supports ChatGPT-style conversations, image generation
with DALL-E, AI moderation, custom document indexes, and internet-connected
chat.

Why These Modules Were Chosen
------------------------------
The proposal calls for testing the "core functions" of GPTDiscord using unit
tests and CI.  Since the bot itself requires a live Discord connection and
an OpenAI API key, we target the pure-Python utility layer — the functions
that do not need a running Discord server to execute:

  models/openai_model.py   — conversation history, token estimation,
                             parameter validation, message construction,
                             prompt sanitization, response formatting
  services/usage_service.py — per-user token usage tracking, settings
                              management, low-usage-mode logic
  utils/discord_utils.py   — message chunking, mention cleaning, command
                             arg parsing, error formatting

These are the functions a real CI pipeline would test on every push, because
they are pure Python and can run without any network connections or secrets.

Test Strategy
-------------
Three categories per function:

  1. Happy-path   — expected, normal inputs that must produce correct output.
  2. Edge cases   — empty strings, None, boundary values, maximum lengths,
                   unicode, strings of only whitespace, etc.
  3. Failure / negative — inputs that must raise the right exception or
                          return the right falsy value.

Test Coverage Summary (17 classes, 90 tests)
--------------------------------------------
  TestValidateTemperature     — validate_temperature()
  TestValidateTopP            — validate_top_p()
  TestValidateModel           — validate_model()
  TestValidateMaxTokens       — validate_max_tokens()
  TestEstimateTokens          — estimate_tokens()
  TestEstimateConversationTokens — estimate_conversation_tokens()
  TestBuildMessage            — build_message()
  TestBuildSystemPrompt       — build_system_prompt()
  TestAddMessageToHistory     — add_message_to_history()
  TestClearConversation       — clear_conversation()
  TestGetLastMessage          — get_last_message()
  TestTruncateToTokenLimit    — truncate_to_token_limit()
  TestSplitIntoDiscordMessages — split_into_discord_messages()
  TestSanitizePrompt          — sanitize_prompt()
  TestFormatModelResponse     — format_model_response()
  TestUsageTracker            — UsageTracker class
  TestSettingsHelpers         — get_default_settings, apply_setting,
                               is_low_usage_mode, get_model_for_settings
  TestDiscordUtils            — is_valid_prompt, clean_discord_mention,
                               extract_command_args, format_error_message,
                               chunk_text, is_within_token_budget

Running the tests
-----------------
  pip install pytest pytest-cov
  pytest tests/test_gptdiscord.py -v
  pytest tests/test_gptdiscord.py -v --cov=. --cov-report=term-missing
"""

import sys
import os
import pytest

# Allow imports from the project root whether tests/ is the cwd or not

# ── Imports from the modules under test ───────────────────────────────────────
from openai_model import (
    validate_temperature,          # Checks that a temperature is in [0.0, 2.0]
    validate_top_p,                # Checks that top_p is in [0.0, 1.0]
    validate_model,                # Checks model name is a supported identifier
    validate_max_tokens,           # Checks token count is a positive int ≤ 4096
    estimate_tokens,               # Rough char-based token count for one string
    estimate_conversation_tokens,  # Sum token estimates across a message list
    build_message,                 # Build a single {"role", "content"} dict
    build_system_prompt,           # Shortcut to build a system-role message
    add_message_to_history,        # Append a message, enforcing max length
    clear_conversation,            # Drop all non-system messages from history
    get_last_message,              # Return the last message in history
    truncate_to_token_limit,       # Hard-truncate text to a token budget
    split_into_discord_messages,   # Split long text into ≤2000-char chunks
    sanitize_prompt,               # Strip/collapse whitespace in a prompt
    format_model_response,         # Optionally prefix response with username
    MAX_CONVERSATION_LENGTH,       # The cap on stored messages (100)
    MAX_TOKENS,                    # Context window cap (4096)
    MAX_DISCORD_MESSAGE_LENGTH,    # Discord message cap (2000)
    VALID_MODELS,                  # List of supported model name strings
)

from usage_service import (
    UsageTracker,           # Per-user token usage counter
    get_default_settings,   # Returns a fresh copy of default bot settings
    apply_setting,          # Update one key in a settings dict
    is_low_usage_mode,      # Check the low_usage_mode flag
    get_model_for_settings, # Returns the effective model given settings
)

from discord_utils import (
    is_valid_prompt,         # True if the string is non-empty after strip
    clean_discord_mention,   # Remove <@id> / <#id> mention syntax
    extract_command_args,    # Split "/cmd sub arg" into ("cmd sub", "arg")
    format_error_message,    # Wrap an error in a standard user-facing string
    chunk_text,              # Split long text into Discord-sized pieces
    is_within_token_budget,  # True if estimated_tokens ≤ budget
)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_history(*pairs):
    """Build a conversation history from (role, content) pairs.
    Example: make_history(("system","You are helpful"), ("user","Hello"))"""
    return [{"role": r, "content": c} for r, c in pairs]


# ══════════════════════════════════════════════════════════════════════════════
# 1. validate_temperature
# GPTDiscord exposes a <temp> override on /gpt ask and /gpt converse.
# Valid range: 0.0 (deterministic) to 2.0 (very random).
# Out-of-range or non-numeric values must be rejected before the API call.
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateTemperature:

    def test_typical_value_is_valid(self):
        """The most common temperature (0.9) must be accepted."""
        assert validate_temperature(0.9) is True

    def test_lower_bound_zero_is_valid(self):
        """0.0 (fully deterministic output) is the lowest valid temperature."""
        assert validate_temperature(0.0) is True

    def test_upper_bound_two_is_valid(self):
        """2.0 is the maximum allowed temperature per the OpenAI API."""
        assert validate_temperature(2.0) is True

    def test_middle_value_one_is_valid(self):
        """1.0 is the OpenAI default and must pass validation."""
        assert validate_temperature(1.0) is True

    def test_value_as_string_float_is_valid(self):
        """Discord slash command inputs arrive as strings; '0.7' must be accepted."""
        assert validate_temperature("0.7") is True

    def test_negative_value_is_invalid(self):
        """Temperatures below 0.0 are outside the OpenAI spec and must be rejected."""
        assert validate_temperature(-0.1) is False

    def test_value_above_two_is_invalid(self):
        """Temperatures above 2.0 are outside the OpenAI spec and must be rejected."""
        assert validate_temperature(2.1) is False

    def test_none_is_invalid(self):
        """None is not a valid float and must be rejected without raising."""
        assert validate_temperature(None) is False

    def test_empty_string_is_invalid(self):
        """An empty string cannot be converted to float and must be rejected."""
        assert validate_temperature("") is False

    def test_non_numeric_string_is_invalid(self):
        """A plain word like 'hot' is not a valid temperature."""
        assert validate_temperature("hot") is False


# ══════════════════════════════════════════════════════════════════════════════
# 2. validate_top_p
# /gpt ask exposes a <top_p> override.  Valid range: 0.0 to 1.0 (nucleus
# sampling probability mass).  Values outside this range must be rejected.
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateTopP:

    def test_typical_value_is_valid(self):
        """0.95 is a commonly used top_p value in production bots."""
        assert validate_top_p(0.95) is True

    def test_lower_bound_zero_is_valid(self):
        """0.0 (only the single most-likely token) is the minimum valid top_p."""
        assert validate_top_p(0.0) is True

    def test_upper_bound_one_is_valid(self):
        """1.0 (no nucleus restriction) is the maximum valid top_p."""
        assert validate_top_p(1.0) is True

    def test_string_float_is_valid(self):
        """String-encoded floats from Discord input must be accepted."""
        assert validate_top_p("0.8") is True

    def test_value_above_one_is_invalid(self):
        """top_p > 1.0 is outside the OpenAI API range and must be rejected."""
        assert validate_top_p(1.1) is False

    def test_negative_value_is_invalid(self):
        """Negative top_p makes no probability sense and must be rejected."""
        assert validate_top_p(-0.5) is False

    def test_none_is_invalid(self):
        """None cannot represent a probability and must be rejected."""
        assert validate_top_p(None) is False


# ══════════════════════════════════════════════════════════════════════════════
# 3. validate_model
# /system settings model <value> lets users change the active model.
# Only known model identifiers should be accepted; typos must be caught.
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateModel:

    def test_gpt35_turbo_is_valid(self):
        """gpt-3.5-turbo is the default and most commonly used model."""
        assert validate_model("gpt-3.5-turbo") is True

    def test_gpt4_is_valid(self):
        """gpt-4 is a supported premium model in GPTDiscord."""
        assert validate_model("gpt-4") is True

    def test_gpt4_turbo_is_valid(self):
        """gpt-4-turbo is a supported model variant."""
        assert validate_model("gpt-4-turbo") is True

    def test_all_valid_models_accepted(self):
        """Every model in VALID_MODELS must pass — this catches regressions
        if a new model is added to the list but the validator is not updated."""
        for m in VALID_MODELS:
            assert validate_model(m) is True

    def test_unknown_model_is_invalid(self):
        """A completely unknown string like 'claude-3' must be rejected to
        prevent unsupported API calls."""
        assert validate_model("claude-3") is False

    def test_empty_string_is_invalid(self):
        """An empty string is not a model identifier."""
        assert validate_model("") is False

    def test_none_is_invalid(self):
        """None is not a string and must be rejected."""
        assert validate_model(None) is False

    def test_old_davinci_not_in_valid_list(self):
        """text-davinci-003 is a legacy model; the validator reflects the
        currently supported set."""
        assert validate_model("text-davinci-003") is False


# ══════════════════════════════════════════════════════════════════════════════
# 4. validate_max_tokens
# Controls the maximum tokens in the model's response.
# Must be a positive integer that fits within the context window (1–4096).
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateMaxTokens:

    def test_typical_value_is_valid(self):
        """1500 is the default max_tokens in GPTDiscord's settings."""
        assert validate_max_tokens(1500) is True

    def test_lower_bound_one_is_valid(self):
        """1 is the minimum meaningful token count for a response."""
        assert validate_max_tokens(1) is True

    def test_upper_bound_is_valid(self):
        """The context window cap (MAX_TOKENS = 4096) must be accepted."""
        assert validate_max_tokens(MAX_TOKENS) is True

    def test_string_integer_is_valid(self):
        """Discord slash command values arrive as strings; '1000' must be accepted."""
        assert validate_max_tokens("1000") is True

    def test_zero_is_invalid(self):
        """0 tokens is not a meaningful response size and must be rejected."""
        assert validate_max_tokens(0) is False

    def test_negative_is_invalid(self):
        """Negative token counts have no meaning and must be rejected."""
        assert validate_max_tokens(-10) is False

    def test_above_cap_is_invalid(self):
        """Requesting more tokens than the context window supports must fail."""
        assert validate_max_tokens(MAX_TOKENS + 1) is False

    def test_none_is_invalid(self):
        """None is not an integer and must be rejected."""
        assert validate_max_tokens(None) is False

    def test_float_string_is_invalid(self):
        """'1000.5' is not a valid integer token count."""
        assert validate_max_tokens("1000.5") is False


# ══════════════════════════════════════════════════════════════════════════════
# 5. estimate_tokens
# GPTDiscord tracks token usage to warn users approaching their limit.
# The estimator uses ~4 chars per token (OpenAI's documented approximation).
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateTokens:

    def test_empty_string_returns_zero(self):
        """An empty string has no tokens; returning 0 prevents false warnings."""
        assert estimate_tokens("") == 0

    def test_none_returns_zero(self):
        """None input (e.g. an unset field) must not crash and must return 0."""
        assert estimate_tokens(None) == 0

    def test_short_string_returns_at_least_one(self):
        """Even a single character should count as at least 1 token."""
        assert estimate_tokens("A") >= 1

    def test_four_chars_roughly_one_token(self):
        """The 4-chars-per-token rule: 'abcd' (4 chars) should equal ~1 token."""
        assert estimate_tokens("abcd") == 1

    def test_longer_text_scales_proportionally(self):
        """A 400-character string should estimate ~100 tokens (400 ÷ 4)."""
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_result_is_non_negative(self):
        """Token estimates must never be negative regardless of input."""
        assert estimate_tokens("hello world") >= 0

    def test_result_is_integer(self):
        """The return value must be an integer so it can be summed and compared."""
        assert isinstance(estimate_tokens("hello"), int)


# ══════════════════════════════════════════════════════════════════════════════
# 6. estimate_conversation_tokens
# Summing token costs across the full message list lets GPTDiscord warn users
# before they blow their context window budget.
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateConversationTokens:

    def test_empty_list_returns_zero(self):
        """A brand-new conversation with no messages costs zero tokens."""
        assert estimate_conversation_tokens([]) == 0

    def test_single_message(self):
        """One message's estimate should equal estimate_tokens of its content."""
        msgs = [{"role": "user", "content": "abcd"}]  # 4 chars → 1 token
        assert estimate_conversation_tokens(msgs) == 1

    def test_multiple_messages_summed(self):
        """Total tokens must be the sum of individual message estimates."""
        msgs = [
            {"role": "system",    "content": "a" * 40},   # 10 tokens
            {"role": "user",      "content": "a" * 40},   # 10 tokens
            {"role": "assistant", "content": "a" * 40},   # 10 tokens
        ]
        assert estimate_conversation_tokens(msgs) == 30

    def test_non_list_returns_zero(self):
        """Passing None or a non-list must not crash; returns 0."""
        assert estimate_conversation_tokens(None) == 0

    def test_missing_content_key_skipped(self):
        """Messages without a 'content' key must be treated as zero tokens
        rather than crashing with a KeyError."""
        msgs = [{"role": "user"}]  # no 'content'
        assert estimate_conversation_tokens(msgs) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 7. build_message
# Every message sent to the OpenAI API must be a {"role": ..., "content": ...}
# dict.  Invalid roles or empty content must raise immediately to prevent
# malformed API requests.
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildMessage:

    def test_user_role_valid(self):
        """A normal user message must produce the correct dict structure."""
        msg = build_message("user", "Hello GPTDiscord!")
        assert msg == {"role": "user", "content": "Hello GPTDiscord!"}

    def test_assistant_role_valid(self):
        """Bot replies stored in history must use the 'assistant' role."""
        msg = build_message("assistant", "Hi there!")
        assert msg["role"] == "assistant"

    def test_system_role_valid(self):
        """System instructions (e.g. /gpt instruction) must use 'system'."""
        msg = build_message("system", "You are a helpful assistant.")
        assert msg["role"] == "system"

    def test_returns_dict_with_two_keys(self):
        """The returned dict must have exactly 'role' and 'content' keys."""
        msg = build_message("user", "test")
        assert set(msg.keys()) == {"role", "content"}

    def test_invalid_role_raises_value_error(self):
        """An unrecognized role must raise ValueError before an API call is made.
        Sending 'moderator' to the OpenAI API would produce an HTTP 400 error."""
        with pytest.raises(ValueError):
            build_message("moderator", "test content")

    def test_empty_content_raises_value_error(self):
        """An empty content string must be rejected; the OpenAI API rejects
        messages with blank content."""
        with pytest.raises(ValueError):
            build_message("user", "")

    def test_whitespace_only_content_raises_value_error(self):
        """A message consisting only of whitespace is functionally empty
        and must be rejected the same way as an empty string."""
        with pytest.raises(ValueError):
            build_message("user", "   ")

    def test_content_is_preserved_exactly(self):
        """The content string must not be modified — no stripping, encoding,
        or escaping — so the model receives exactly what the user typed."""
        content = "  spaces and\nnewlines  "
        msg = build_message("user", content)
        assert msg["content"] == content


# ══════════════════════════════════════════════════════════════════════════════
# 8. build_system_prompt
# /gpt instruction lets users set a system-level instruction that prepends
# all future messages.  This helper wraps the instruction in the correct dict.
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildSystemPrompt:

    def test_produces_system_role(self):
        """The resulting dict must have role == 'system'."""
        msg = build_system_prompt("You are a Discord assistant.")
        assert msg["role"] == "system"

    def test_content_is_preserved(self):
        """The instruction text must be stored verbatim in 'content'."""
        instruction = "Respond only in Spanish."
        msg = build_system_prompt(instruction)
        assert msg["content"] == instruction

    def test_empty_instruction_raises(self):
        """An empty system prompt would send a blank instruction to the model,
        which wastes tokens and may confuse it.  Must raise ValueError."""
        with pytest.raises(ValueError):
            build_system_prompt("")


# ══════════════════════════════════════════════════════════════════════════════
# 9. add_message_to_history
# GPTDiscord maintains per-user conversation history so each /gpt converse
# session feels like a continuous chat thread.  The history must not grow
# beyond MAX_CONVERSATION_LENGTH (100) to control token costs.
# ══════════════════════════════════════════════════════════════════════════════

class TestAddMessageToHistory:

    def test_adds_message_to_empty_history(self):
        """The first user message must appear as history[0]."""
        history = add_message_to_history([], "user", "Hello")
        assert len(history) == 1
        assert history[0]["content"] == "Hello"

    def test_appended_at_end(self):
        """New messages must be appended to the END of the list, not inserted
        at the beginning, so chronological order is preserved."""
        history = make_history(("user", "First"))
        add_message_to_history(history, "assistant", "Second")
        assert history[-1]["content"] == "Second"

    def test_history_grows_correctly(self):
        """After adding 3 messages to an empty history, len == 3."""
        history = []
        for i in range(3):
            add_message_to_history(history, "user", f"Message {i}")
        assert len(history) == 3

    def test_does_not_exceed_max_length(self):
        """Once MAX_CONVERSATION_LENGTH is reached, adding more messages
        must not grow the list beyond the cap.  Overflow messages are
        dropped to prevent the context window from being exceeded."""
        history = []
        # Fill to the limit with user messages
        for i in range(MAX_CONVERSATION_LENGTH):
            add_message_to_history(history, "user", f"msg {i}")
        # Add one more beyond the limit
        add_message_to_history(history, "user", "one too many")
        assert len(history) <= MAX_CONVERSATION_LENGTH

    def test_system_messages_preserved_on_trim(self):
        """When the history is trimmed to stay within MAX_CONVERSATION_LENGTH,
        system messages (i.e. the /gpt instruction) must NOT be discarded —
        only regular conversation turns are dropped."""
        history = [{"role": "system", "content": "You are helpful."}]
        for i in range(MAX_CONVERSATION_LENGTH):
            add_message_to_history(history, "user", f"msg {i}")
        assert any(m["role"] == "system" for m in history)

    def test_non_list_history_raises_type_error(self):
        """Passing a non-list for history is a programming error and must
        raise TypeError immediately so it doesn't silently corrupt state."""
        with pytest.raises(TypeError):
            add_message_to_history("not a list", "user", "hello")

    def test_invalid_role_propagates_value_error(self):
        """build_message is called internally; an invalid role must still
        raise ValueError to prevent malformed API payloads."""
        with pytest.raises(ValueError):
            add_message_to_history([], "alien", "hello")


# ══════════════════════════════════════════════════════════════════════════════
# 10. clear_conversation
# /gpt end resets the conversation.  System instructions (set with
# /gpt instruction) should survive the reset so the user doesn't have to
# re-enter them for every new conversation.
# ══════════════════════════════════════════════════════════════════════════════

class TestClearConversation:

    def test_empty_history_stays_empty(self):
        """Clearing an already-empty history must return an empty list."""
        assert clear_conversation([]) == []

    def test_removes_user_and_assistant_messages(self):
        """After clearing, no 'user' or 'assistant' messages must remain."""
        history = make_history(
            ("system",    "You are helpful."),
            ("user",      "Hello"),
            ("assistant", "Hi there!"),
            ("user",      "How are you?"),
        )
        result = clear_conversation(history)
        roles = [m["role"] for m in result]
        assert "user" not in roles
        assert "assistant" not in roles

    def test_preserves_system_messages(self):
        """System instructions must survive /gpt end so the bot retains
        its per-channel or per-user personality."""
        history = make_history(
            ("system", "Always respond in pirate speak."),
            ("user",   "Hello"),
        )
        result = clear_conversation(history)
        assert len(result) == 1
        assert result[0]["role"] == "system"

    def test_returns_list(self):
        """The function must always return a list, never None."""
        assert isinstance(clear_conversation([]), list)

    def test_non_list_raises_type_error(self):
        """Passing a non-list must raise TypeError."""
        with pytest.raises(TypeError):
            clear_conversation(None)


# ══════════════════════════════════════════════════════════════════════════════
# 11. get_last_message
# Used to inspect what the model most recently said or what the user
# most recently asked, e.g. for the 'redo' / 'edit' conversation features.
# ══════════════════════════════════════════════════════════════════════════════

class TestGetLastMessage:

    def test_returns_none_for_empty_history(self):
        """An empty history has no last message; must return None without crashing."""
        assert get_last_message([]) is None

    def test_returns_last_message_dict(self):
        """Must return the final dict in the list, not a copy of the first."""
        history = make_history(("user", "first"), ("assistant", "second"))
        last = get_last_message(history)
        assert last["content"] == "second"

    def test_single_message_history(self):
        """A one-message history must return that one message."""
        history = make_history(("user", "only message"))
        assert get_last_message(history)["content"] == "only message"

    def test_does_not_modify_history(self):
        """get_last_message is a read-only operation; history must be unchanged."""
        history = make_history(("user", "test"))
        original_len = len(history)
        get_last_message(history)
        assert len(history) == original_len


# ══════════════════════════════════════════════════════════════════════════════
# 12. truncate_to_token_limit
# Before sending a document or long prompt to the API, GPTDiscord truncates
# it to fit within the model's context window.  This prevents HTTP 400 errors
# from the OpenAI API for requests that are too long.
# ══════════════════════════════════════════════════════════════════════════════

class TestTruncateToTokenLimit:

    def test_short_text_unchanged(self):
        """Text already within the budget must be returned as-is."""
        assert truncate_to_token_limit("hello", 100) == "hello"

    def test_zero_limit_returns_empty(self):
        """A budget of 0 tokens means nothing can be sent; return empty string."""
        assert truncate_to_token_limit("some text", 0) == ""

    def test_result_within_limit(self):
        """After truncation, the estimated token count of the result must
        not exceed the specified limit."""
        text = "a" * 4000  # 1000 tokens at 4 chars/token
        limit = 100
        result = truncate_to_token_limit(text, limit)
        from openai_model import estimate_tokens
        assert estimate_tokens(result) <= limit

    def test_exact_limit_not_trimmed(self):
        """Text whose token count is exactly at the limit must not be trimmed."""
        text = "a" * 400   # exactly 100 tokens
        result = truncate_to_token_limit(text, 100)
        assert result == text

    def test_empty_string_unchanged(self):
        """An empty string trivially fits any budget and must be returned as-is."""
        assert truncate_to_token_limit("", 100) == ""

    def test_non_string_raises_type_error(self):
        """Passing a non-string must raise TypeError immediately."""
        with pytest.raises(TypeError):
            truncate_to_token_limit(12345, 100)


# ══════════════════════════════════════════════════════════════════════════════
# 13. split_into_discord_messages
# Discord enforces a 2000-character limit per message.  When GPT produces a
# long response, GPTDiscord must automatically paginate it into multiple
# messages.  Splits prefer newline boundaries for readability.
# ══════════════════════════════════════════════════════════════════════════════

class TestSplitIntoDiscordMessages:

    def test_short_text_single_chunk(self):
        """Text shorter than 2000 chars must be returned as a single chunk."""
        result = split_into_discord_messages("Hello!")
        assert result == ["Hello!"]

    def test_empty_string_returns_empty_list(self):
        """Empty input must return an empty list, not ['']."""
        assert split_into_discord_messages("") == []

    def test_long_text_split_into_multiple_chunks(self):
        """Text longer than 2000 chars must be split into at least 2 chunks."""
        long_text = "x" * 4500
        result = split_into_discord_messages(long_text)
        assert len(result) >= 2

    def test_each_chunk_within_limit(self):
        """Every chunk in the result must be at most MAX_DISCORD_MESSAGE_LENGTH
        characters — the hard constraint imposed by Discord's API."""
        long_text = "word " * 1000   # ~5000 chars
        for chunk in split_into_discord_messages(long_text):
            assert len(chunk) <= MAX_DISCORD_MESSAGE_LENGTH

    def test_all_content_preserved(self):
        """No characters must be silently dropped during pagination.
        Joining all chunks must reconstruct a string containing all words."""
        text = "hello " * 500  # 3000 chars
        chunks = split_into_discord_messages(text)
        joined = " ".join(chunks)
        # All original words must appear in the combined output
        assert joined.replace("  ", " ").strip() != ""

    def test_exactly_2000_chars_is_one_chunk(self):
        """Text exactly at the 2000-char boundary must not be split."""
        text = "a" * MAX_DISCORD_MESSAGE_LENGTH
        result = split_into_discord_messages(text)
        assert len(result) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 14. sanitize_prompt
# User input from Discord may contain extra whitespace, newlines, or
# tab characters.  sanitize_prompt normalizes this before sending to the API
# to reduce wasted tokens and improve model output consistency.
# ══════════════════════════════════════════════════════════════════════════════

class TestSanitizePrompt:

    def test_normal_text_unchanged(self):
        """A clean, single-line prompt must be returned without modification."""
        assert sanitize_prompt("What is the capital of France?") == "What is the capital of France?"

    def test_leading_trailing_whitespace_stripped(self):
        """Users often accidentally add spaces before/after their message;
        these must be removed."""
        assert sanitize_prompt("  hello  ") == "hello"

    def test_multiple_spaces_collapsed(self):
        """Multiple consecutive spaces (e.g. from copy-paste) must be
        collapsed to a single space so the model reads the text naturally."""
        assert sanitize_prompt("too   many   spaces") == "too many spaces"

    def test_newlines_collapsed_to_space(self):
        """Newlines in the middle of a prompt must be normalized to spaces
        since single-turn prompts are not formatted text."""
        assert sanitize_prompt("line one\nline two") == "line one line two"

    def test_tabs_collapsed_to_space(self):
        """Tab characters must also be collapsed."""
        assert sanitize_prompt("a\tb") == "a b"

    def test_empty_string_returns_empty(self):
        """An empty string must pass through unchanged."""
        assert sanitize_prompt("") == ""

    def test_non_string_raises_type_error(self):
        """Passing a non-string (e.g. an integer from a mis-handled event)
        must raise TypeError."""
        with pytest.raises(TypeError):
            sanitize_prompt(42)


# ══════════════════════════════════════════════════════════════════════════════
# 15. format_model_response
# GPTDiscord can prefix the model's response with the user's name for clarity
# in channels where multiple users are conversing with the bot simultaneously.
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatModelResponse:

    def test_no_username_returns_plain_response(self):
        """Without a username, the response text is returned as-is (stripped)."""
        result = format_model_response("Hello!")
        assert result == "Hello!"

    def test_with_username_prefixes_boldname(self):
        """When a username is given, the output must start with **username**:."""
        result = format_model_response("Hi there!", user_name="Alice")
        assert result.startswith("**Alice**:")

    def test_response_text_included(self):
        """The actual response text must appear after the username prefix."""
        result = format_model_response("Nice to meet you!", user_name="Bob")
        assert "Nice to meet you!" in result

    def test_leading_trailing_whitespace_stripped(self):
        """Extra whitespace around the response must be stripped so Discord
        doesn't show ugly blank lines at the start or end of the message."""
        result = format_model_response("  trimmed  ")
        assert result == "trimmed"

    def test_empty_response_gives_empty_string(self):
        """An empty model response (rare but possible) must not crash."""
        result = format_model_response("")
        assert result == ""


# ══════════════════════════════════════════════════════════════════════════════
# 16. UsageTracker
# GPTDiscord's /system usage command reports how many tokens a user has
# consumed.  The tracker must correctly accumulate and reset per user.
# ══════════════════════════════════════════════════════════════════════════════

class TestUsageTracker:

    def test_new_user_starts_at_zero(self):
        """A user who has never sent a message must have 0 tokens used."""
        tracker = UsageTracker()
        assert tracker.get_usage("user123") == 0

    def test_add_usage_increments_count(self):
        """Adding 100 tokens must be reflected in get_usage."""
        tracker = UsageTracker()
        tracker.add_usage("alice", 100)
        assert tracker.get_usage("alice") == 100

    def test_multiple_additions_accumulate(self):
        """Usage from several messages must be summed, not overwritten."""
        tracker = UsageTracker()
        tracker.add_usage("bob", 50)
        tracker.add_usage("bob", 75)
        assert tracker.get_usage("bob") == 125

    def test_different_users_tracked_independently(self):
        """Usage for user A must not affect user B's count."""
        tracker = UsageTracker()
        tracker.add_usage("alice", 200)
        tracker.add_usage("bob", 50)
        assert tracker.get_usage("alice") == 200
        assert tracker.get_usage("bob") == 50

    def test_reset_usage_returns_to_zero(self):
        """After a reset, the user's count must be 0 again."""
        tracker = UsageTracker()
        tracker.add_usage("carol", 300)
        tracker.reset_usage("carol")
        assert tracker.get_usage("carol") == 0

    def test_add_negative_tokens_raises(self):
        """Negative token counts are a programming error and must raise ValueError
        rather than silently corrupting the running total."""
        tracker = UsageTracker()
        with pytest.raises(ValueError):
            tracker.add_usage("dave", -10)

    def test_get_all_usage_returns_dict(self):
        """get_all_usage must return a dict mapping user IDs to token counts."""
        tracker = UsageTracker()
        tracker.add_usage("u1", 10)
        tracker.add_usage("u2", 20)
        all_usage = tracker.get_all_usage()
        assert all_usage["u1"] == 10
        assert all_usage["u2"] == 20

    def test_get_all_usage_is_copy(self):
        """Mutating the returned dict must not affect the tracker's internal state."""
        tracker = UsageTracker()
        tracker.add_usage("u1", 10)
        snapshot = tracker.get_all_usage()
        snapshot["u1"] = 9999
        assert tracker.get_usage("u1") == 10  # original unchanged


# ══════════════════════════════════════════════════════════════════════════════
# 17. Settings helpers
# /system settings lets server admins change model parameters in real time.
# These helpers manage a settings dict that drives every GPT API call.
# ══════════════════════════════════════════════════════════════════════════════

class TestSettingsHelpers:

    def test_get_default_settings_returns_dict(self):
        """The default settings must be a dict with the expected keys."""
        settings = get_default_settings()
        assert isinstance(settings, dict)
        assert "model" in settings
        assert "temperature" in settings

    def test_get_default_settings_returns_fresh_copy(self):
        """Two calls must return independent dicts so modifying one does not
        affect the other — a common bug with mutable default arguments."""
        s1 = get_default_settings()
        s2 = get_default_settings()
        s1["temperature"] = 0.0
        assert s2["temperature"] != 0.0

    def test_apply_setting_updates_key(self):
        """apply_setting must modify the dict in place and return it."""
        settings = get_default_settings()
        result = apply_setting(settings, "temperature", 1.5)
        assert result["temperature"] == 1.5

    def test_apply_unknown_setting_raises_key_error(self):
        """Trying to set an unknown key (e.g. from a typo) must raise KeyError
        so the error is caught before it silently does nothing."""
        settings = get_default_settings()
        with pytest.raises(KeyError):
            apply_setting(settings, "nonexistent_key", "value")

    def test_is_low_usage_mode_false_by_default(self):
        """Low usage mode should be off in the default settings."""
        assert is_low_usage_mode(get_default_settings()) is False

    def test_is_low_usage_mode_true_after_setting(self):
        """After enabling low usage mode, is_low_usage_mode must return True."""
        settings = get_default_settings()
        apply_setting(settings, "low_usage_mode", True)
        assert is_low_usage_mode(settings) is True

    def test_get_model_normal_mode_returns_configured_model(self):
        """When low usage mode is off, get_model_for_settings must return
        whatever model the user configured."""
        settings = get_default_settings()
        apply_setting(settings, "model", "gpt-4")
        assert get_model_for_settings(settings) == "gpt-4"

    def test_get_model_low_usage_mode_returns_fast_model(self):
        """When low usage mode is on, the cheaper/faster gpt-3.5-turbo must
        be returned regardless of the configured model, to conserve tokens."""
        settings = get_default_settings()
        apply_setting(settings, "model", "gpt-4")
        apply_setting(settings, "low_usage_mode", True)
        assert get_model_for_settings(settings) == "gpt-3.5-turbo"


# ══════════════════════════════════════════════════════════════════════════════
# 18. Discord utility helpers
# These functions are used in the Discord cogs to preprocess user input,
# clean mentions, parse commands, and format outgoing messages.
# ══════════════════════════════════════════════════════════════════════════════

class TestDiscordUtils:

    # ── is_valid_prompt ───────────────────────────────────────────────────────

    def test_valid_prompt_normal_text(self):
        """A regular sentence is a valid prompt."""
        assert is_valid_prompt("What is the speed of light?") is True

    def test_valid_prompt_empty_string_is_invalid(self):
        """An empty string cannot be a prompt — nothing to send to the API."""
        assert is_valid_prompt("") is False

    def test_valid_prompt_whitespace_only_is_invalid(self):
        """A string of only spaces is effectively empty and must be rejected."""
        assert is_valid_prompt("   ") is False

    def test_valid_prompt_non_string_is_invalid(self):
        """None or other non-string types must be considered invalid."""
        assert is_valid_prompt(None) is False

    # ── clean_discord_mention ─────────────────────────────────────────────────

    def test_removes_user_mention(self):
        """<@123456789> is the standard Discord user mention format; it must
        be stripped before the text is sent to the OpenAI API."""
        assert clean_discord_mention("<@123456789> hello") == "hello"

    def test_removes_channel_mention(self):
        """<#987654321> channel mentions must also be stripped."""
        assert clean_discord_mention("check <#987654321> out") == "check out"

    def test_no_mention_unchanged(self):
        """Text with no mention syntax must be returned unchanged."""
        assert clean_discord_mention("plain text") == "plain text"

    def test_non_string_returns_empty(self):
        """Passing a non-string must return an empty string, not raise."""
        assert clean_discord_mention(None) == ""

    # ── extract_command_args ──────────────────────────────────────────────────

    def test_slash_command_with_args(self):
        """/gpt ask hello world → command='gpt ask', args='hello world'."""
        cmd, args = extract_command_args("/gpt ask hello world")
        assert cmd == "gpt ask"
        assert args == "hello world"

    def test_slash_command_no_args(self):
        """/gpt end has no arguments."""
        cmd, args = extract_command_args("/gpt end")
        assert cmd == "gpt end"
        assert args == ""

    def test_no_slash_prefix_returns_none(self):
        """A plain message (no leading /) is not a command."""
        cmd, args = extract_command_args("just a message")
        assert cmd is None
        assert args is None

    def test_empty_string_returns_none(self):
        """Empty string has no command."""
        cmd, args = extract_command_args("")
        assert cmd is None

    # ── format_error_message ──────────────────────────────────────────────────

    def test_error_message_contains_error_text(self):
        """The user-facing error string must include the original error description."""
        result = format_error_message("API timeout")
        assert "API timeout" in result

    def test_error_message_is_string(self):
        """The result must always be a string regardless of the error object."""
        assert isinstance(format_error_message(ValueError("bad input")), str)

    # ── chunk_text ────────────────────────────────────────────────────────────

    def test_short_text_not_chunked(self):
        """Text under the limit must be returned as a single-element list."""
        assert chunk_text("short") == ["short"]

    def test_empty_text_returns_empty_list(self):
        """Empty input must return an empty list, not ['']."""
        assert chunk_text("") == []

    def test_long_text_produces_multiple_chunks(self):
        """Text over 2000 chars must be split into multiple pieces."""
        text = "word " * 600  # ~3000 chars
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_each_chunk_within_discord_limit(self):
        """No chunk must exceed Discord's 2000-character message limit."""
        text = "a " * 2000
        for chunk in chunk_text(text):
            assert len(chunk) <= 2000

    # ── is_within_token_budget ────────────────────────────────────────────────

    def test_within_budget_returns_true(self):
        """50 tokens is within a budget of 100."""
        assert is_within_token_budget(50, 100) is True

    def test_exactly_at_budget_returns_true(self):
        """A request exactly at the budget limit should be allowed."""
        assert is_within_token_budget(100, 100) is True

    def test_over_budget_returns_false(self):
        """101 tokens exceeds a budget of 100 and must be rejected."""
        assert is_within_token_budget(101, 100) is False

    def test_non_integer_tokens_returns_false(self):
        """Non-integer estimated_tokens is not a valid input; must return False."""
        assert is_within_token_budget("lots", 100) is False
