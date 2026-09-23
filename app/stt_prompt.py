"""
app/stt_prompt.py — Dynamic initial_prompt builder for Whisper STT.

Rationale for this approach vs alternatives:
  - REJECTED: Static initial_prompt (hardcoded vocabulary list). Requires manual
    maintenance every time the user changes topics or interests. Breaks the
    "permanent, maintenance-free" principle.
  - REJECTED: Larger model (small -> medium). Valid long-term option, but doubles
    latency and RAM usage. Not warranted until dynamic prompting is benchmarked.
  - CHOSEN: Dynamic prompt from conversation history. Whisper's initial_prompt
    parameter biases the decoder toward previously-used vocabulary and topics.
    Since we already track multi-turn history, it costs nothing extra and adapts
    automatically to any topic the user discusses.

Whisper's initial_prompt behavior (from OpenAI documentation):
  - The model treats initial_prompt as "prior context" — decoded tokens that
    preceded the current audio clip.
  - It biases the beam search toward continuations consistent with those tokens.
  - Effective length: ~224 tokens (~800-1000 chars for Spanish). We limit to
    max_chars to stay within that budget without tokenizing explicitly.
  - Only user messages are included: Kelly's responses are TTS-synthesized speech
    the model hasn't "heard", so including them adds noise without benefit.
"""
from __future__ import annotations


def build_initial_prompt(
    history: list[dict[str, str]],
    max_chars: int = 224,
    last_n_user_messages: int = 2,
) -> str:
    """
    Build a dynamic Whisper initial_prompt from recent conversation history.

    Uses only the user's own words — the vocabulary and topics they have already
    used — to bias the decoder for the *next* utterance without hardcoding anything.

    Args:
        history: List of {"role": "user"|"assistant", "content": "..."} dicts.
        max_chars: Maximum prompt length in characters. Defaults to 224 —
            chosen to stay safely within Whisper's ~224-token context window.
            (Spanish: ~4 chars/token average, so 224 chars ≈ 56 tokens, well
            under the limit even with tokenization overhead.)
        last_n_user_messages: Number of recent user messages to include.
            2 is the sweet spot: enough context to anchor vocabulary without
            diluting the prompt with stale topics.

    Returns:
        A string to pass as initial_prompt, or "" if history is empty/has no
        user messages (Whisper ignores empty-string initial_prompt safely).
    """
    if not history:
        return ""

    user_messages = [
        m["content"]
        for m in history
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content")
    ]

    if not user_messages:
        return ""

    recent = " ".join(user_messages[-last_n_user_messages:])

    # Truncate from the left to keep the most recent content
    if len(recent) > max_chars:
        recent = recent[-max_chars:]

    return recent
