"""Tests for LLM judge parsing and the verifier factory."""

from rsi_plateau.utils.config import Config
from rsi_plateau.verifiers import LLMJudge, build_verifier, parse_judge_output


def test_parse_correct():
    d = parse_judge_output("The reasoning is fine.\nCORRECT\nCONFIDENCE: 0.9")
    assert d.accepted
    assert abs(d.score - 0.9) < 1e-9


def test_parse_incorrect_not_confused_by_correct_substring():
    d = parse_judge_output("INCORRECT\nCONFIDENCE: 0.8")
    assert not d.accepted
    # score is 1 - confidence for rejections
    assert abs(d.score - 0.2) < 1e-9


def test_parse_unparseable_rejects():
    d = parse_judge_output("I am not sure what to say")
    assert not d.accepted
    assert d.score == 0.0


def test_parse_empty():
    d = parse_judge_output("")
    assert not d.accepted


def test_parse_clamps_confidence():
    d = parse_judge_output("CORRECT\nCONFIDENCE: 1.8")
    assert d.score == 1.0


class _StubBackend:
    def __init__(self, text):
        self.text = text

    def generate_texts(self, prompts, *, temperature, max_new_tokens):
        return [self.text for _ in prompts]


def test_llm_judge_uses_backend():
    judge = LLMJudge(_StubBackend("CORRECT\nCONFIDENCE: 0.7"), name="weak")
    d = judge.verify("q", "s")
    assert d.accepted
    assert judge.name == "weak"


def test_factory_oracle():
    cfg = Config({"verifier": {"type": "oracle"}, "model": {}})
    assert build_verifier(cfg).name == "oracle"


def test_factory_judge_with_stub_backend():
    cfg = Config({"verifier": {"type": "same_family"}, "model": {"dtype": "auto", "device": "cpu"}})
    v = build_verifier(cfg, backend=_StubBackend("CORRECT\nCONFIDENCE: 0.6"))
    assert v.name == "same_family"


def test_factory_unknown_type():
    cfg = Config({"verifier": {"type": "nope"}, "model": {}})
    try:
        build_verifier(cfg)
        assert False
    except ValueError:
        pass
