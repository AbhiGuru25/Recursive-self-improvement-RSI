"""Tests for answer extraction and normalization (oracle-critical)."""

from rsi_plateau.data.answers import answers_match, extract_final_answer, normalize_answer


def test_extract_hash_delimiter():
    assert extract_final_answer("some work\n#### 42") == "42"


def test_extract_hash_with_currency():
    assert extract_final_answer("#### $1,234") == "1234"


def test_extract_boxed():
    assert extract_final_answer(r"therefore \boxed{7}") == "7"


def test_extract_answer_line():
    assert extract_final_answer("The answer is: 15 apples") == "15"


def test_extract_last_number_fallback():
    assert extract_final_answer("we compute 3 + 4 = 7") == "7"


def test_extract_none_on_empty():
    assert extract_final_answer("") is None


def test_normalize_integer_float_equivalence():
    assert normalize_answer("42") == normalize_answer("42.0")


def test_normalize_commas_and_currency():
    assert normalize_answer("$1,000") == "1000"


def test_normalize_negative_decimal():
    assert normalize_answer("-3.50") == "-3.5"


def test_answers_match_positive():
    assert answers_match("42", "42.0")


def test_answers_match_negative():
    assert not answers_match("42", "43")


def test_answers_match_handles_none():
    assert not answers_match(None, "42")
    assert not answers_match("42", None)
