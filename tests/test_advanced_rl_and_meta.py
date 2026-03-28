import numpy as np

from execution.erl_cmdp import ERLConfig, EntropyRegulatedLagrangianAgent
from signal_generation.meta_labeling import DiscreteLongShortMetaLabeler, MetaLabelConfig


def test_erl_cmdp_updates_lagrange_multipliers_on_constraint_violation():
    agent = EntropyRegulatedLagrangianAgent(
        ERLConfig(
            action_count=3,
            learning_rate=0.1,
            lambda_lr=0.05,
            drawdown_limit=0.10,
            leverage_cap=1.0,
            entropy_target=0.9,
        )
    )

    state = ("regime_a", 1)
    action = agent.select_action(state, rng=np.random.default_rng(0))

    out = agent.update(
        state=state,
        action=action,
        reward=0.2,
        drawdown=0.18,
        leverage=1.35,
        market_uncertainty=1.0,
    )

    assert out["c_drawdown"] > 0
    assert out["c_leverage"] > 0
    assert out["c_entropy"] >= 0
    assert out["lambda_drawdown"] > 0
    assert out["lambda_leverage"] > 0


def test_discrete_long_short_meta_labeling_outputs_signed_sizes():
    rng = np.random.default_rng(12)
    n = 240

    x = rng.normal(size=(n, 4))
    side = np.where(np.arange(n) % 2 == 0, 1, -1)

    score = 0.8 * x[:, 0] - 0.4 * x[:, 1] + 0.5 * side
    p = 1.0 / (1.0 + np.exp(-score))
    y = (rng.random(n) < p).astype(int)

    model = DiscreteLongShortMetaLabeler(
        MetaLabelConfig(calibration_method="sigmoid", sizing_method="sops", min_confidence=0.5)
    )
    model.fit(x, y, side)

    sizes = model.predict_position_sizes(x, side)
    assert len(sizes) == n
    assert (sizes[side > 0] >= 0).all()
    assert (sizes[side < 0] <= 0).all()
    assert np.max(np.abs(sizes)) <= 1.0
