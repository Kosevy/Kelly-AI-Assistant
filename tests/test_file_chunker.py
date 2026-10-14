"""
tests/test_file_chunker.py
Pruebas unitarias para app.file_chunker (cálculo de presupuesto de tokens y truncado).
"""
from app.file_chunker import estimate_tokens, truncate_to_token_budget


def test_estimate_tokens():
    text = "a" * 400
    assert estimate_tokens(text, tokens_per_char=0.25) == 100


def test_truncate_within_budget():
    text = "Texto corto dentro del presupuesto"
    truncated, was_truncated = truncate_to_token_budget(text, max_tokens=100, tokens_per_char=0.25)
    assert was_truncated is False
    assert truncated == text


def test_truncate_exceeding_budget():
    # 100 tokens a 0.25 tokens/char = 400 caracteres máximo
    long_text = "x" * 600
    truncated, was_truncated = truncate_to_token_budget(long_text, max_tokens=100, tokens_per_char=0.25)
    assert was_truncated is True
    assert len(truncated) == 400
    assert truncated == "x" * 400


def test_truncate_edge_cases():
    # Texto vacío
    t1, w1 = truncate_to_token_budget("", max_tokens=100)
    assert t1 == ""
    assert w1 is False

    # max_tokens = 0
    t2, w2 = truncate_to_token_budget("algo", max_tokens=0)
    assert t2 == ""
    assert w2 is True
