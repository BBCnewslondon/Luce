import numpy as np
import pandas as pd

from signal_generation.ensemble import (
    EnsembleConfig,
    fit_ensemble,
    generate_signal_frame,
    lag_feature_columns,
    predict_ensemble,
)


def test_lag_feature_columns_applies_shift():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    out = lag_feature_columns(df, ["a", "b"], lag=1)
    assert np.isnan(out.loc[0, "a"])
    assert out.loc[1, "a"] == 1.0
    assert out.loc[2, "b"] == 5.0


def test_fit_predict_generate_signal_frame_shapes():
    rng = np.random.default_rng(7)
    n = 500
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "target": rng.normal(size=n) * 0.001,
        }
    )

    bundle = fit_ensemble(
        frame=df,
        feature_columns=["f1", "f2"],
        target_column="target",
        config=EnsembleConfig(),
    )
    pred = predict_ensemble(df, bundle)
    sig = generate_signal_frame(df, pred, threshold=0.0)

    assert len(pred) == n
    assert len(sig) == n
    assert set(["prediction", "signal", "confidence"]).issubset(sig.columns)
    assert sig["signal"].isin([-1, 0, 1]).all()
