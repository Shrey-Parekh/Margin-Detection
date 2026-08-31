"""Rule-based trait scoring from margin flags.

Each margin the detector sets contributes its per-trait weight from
margin_trait_weights.csv; the trait is called positive when the weights sum
above that trait's threshold in margin_trait_thresholds.csv.

The weights are graphological priors, not fitted values. Only the seven
thresholds are learned, which is what keeps this usable at n=29 -- fitting the
weights themselves overfits well before it helps (leave-one-out: hand weights
73.4%, logistic regression on the same inputs 62.6%).

Run this module directly to refit the thresholds.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_CSV = os.path.join(HERE, "margin_trait_weights.csv")
THRESHOLDS_CSV = os.path.join(HERE, "margin_trait_thresholds.csv")

TRAITS = ["visionary", "depressive", "anxiety", "respect_for_others",
          "procrastination", "discipline", "truthfulness"]

THRESHOLD_GRID = np.arange(-6.0, 6.5, 0.5)


def load_weights(path=WEIGHTS_CSV):
    """Weight matrix indexed by margin code, columns in TRAITS order."""
    return pd.read_csv(path, index_col="margin_code")[TRAITS]


def load_thresholds(path=THRESHOLDS_CSV):
    """Fitted per-trait thresholds, or all-zero if none have been fitted yet."""
    if not os.path.exists(path):
        return pd.Series(0.0, index=TRAITS)
    return pd.read_csv(path, index_col="trait")["threshold"].reindex(TRAITS)


def score(features, weights=None):
    """Summed trait weights for a frame of 0/1 margin flags.

    Uses the margin columns common to both inputs, so the 12-column
    (features_auto_81) and 18-column (batch_extract_all) feature sets both work
    without a separate code path.
    """
    if weights is None:
        weights = load_weights()
    cols = [c for c in weights.index if c in features.columns]
    return features[cols].dot(weights.loc[cols])


def fit_thresholds(features, labels, weights=None):
    """Per-trait threshold that best reproduces `labels`.

    `labels` columns must be in TRAITS order; the label CSV uses its own
    spellings, so it is matched by position rather than by name.
    """
    s = score(features, weights)
    best = [max(THRESHOLD_GRID,
                key=lambda t: ((s.iloc[:, i] > t) == labels.iloc[:, i]).mean())
            for i in range(len(TRAITS))]
    return pd.Series(best, index=TRAITS)


def predict(features, weights=None, thresholds=None):
    """Binary trait predictions: score above the trait's threshold -> 1."""
    if thresholds is None:
        thresholds = load_thresholds()
    return score(features, weights).gt(thresholds, axis=1).astype(int)


if __name__ == "__main__":
    # Fitted on the expert margin calls, which are the inputs the weights were
    # written against. The detector's own margins do not yet reproduce them.
    margins = pd.read_csv(os.path.join(HERE, "features_manual_29.csv"))
    margins.columns = [c.upper() for c in margins.columns]
    margins = margins.rename(columns={"RLM": "RFLM"})
    labels = pd.read_csv(os.path.join(HERE, "labels_traits_29.csv"))

    th = fit_thresholds(margins, labels)
    th.rename_axis("trait").rename("threshold").to_csv(THRESHOLDS_CSV)
    print(f"fitted thresholds -> {os.path.basename(THRESHOLDS_CSV)}")
    print(th.to_string())
