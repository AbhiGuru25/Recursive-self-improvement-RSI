"""End-to-end loop tests with the stub generator/trainer (no model)."""


from rsi_plateau.data.gsm8k import GSM8KExample
from rsi_plateau.generation import StubGenerator
from rsi_plateau.loop import LoopRunner
from rsi_plateau.training import StubTrainer
from rsi_plateau.verifiers import OracleVerifier


def _make_data(n=10):
    ex = [
        GSM8KExample(
            question=f"q{i}",
            gold_solution=f"#### {i}",
            gold_answer=str(i),
            split="train",
        )
        for i in range(n)
    ]
    return ex, ex


def _runner(architecture="star"):
    gen = StubGenerator(seed=0)
    train, test = _make_data()
    gen.set_gold([e.question for e in train], [e.gold_answer for e in train])
    trainer = StubTrainer(gen, gain=0.3, ceiling=0.9, base_p_correct=0.2)
    return LoopRunner(
        config_name="test",
        architecture=architecture,
        generator=gen,
        verifier=OracleVerifier(),
        trainer=trainer,
        rounds=4,
        k_samples=3,
        bootstrap_samples=100,
        seed=0,
    ), train, test, {"p_correct": 0.2}


def test_loop_produces_rounds():
    runner, train, test, handle = _runner()
    result = runner.run(train, test, handle)
    assert len(result.rounds) == 4
    assert all(0.0 <= r.accuracy <= 1.0 for r in result.rounds)


def test_loop_records_diversity_and_rho():
    runner, train, test, handle = _runner()
    result = runner.run(train, test, handle)
    assert "self_bleu" in result.rounds[0].diversity
    # Round 0 is the reference, so rho should be ~1.0 there.
    assert abs(result.rounds[0].rho["self_bleu"] - 1.0) < 1e-6


def test_rest_architecture_resets():
    runner, train, test, handle = _runner(architecture="rest")
    result = runner.run(train, test, handle)
    assert result.architecture == "rest"
    assert len(result.rounds) == 4


def test_plateau_detected():
    runner, train, test, handle = _runner()
    result = runner.run(train, test, handle)
    assert result.plateau is not None
    assert "plateau_round" in result.plateau
