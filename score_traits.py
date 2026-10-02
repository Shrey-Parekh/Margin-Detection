"""Rule-based trait scoring from margin flags.

Each margin the detector sets contributes its per-trait weight from
margin_trait_weights.csv. The trait is called positive when the summed weight
falls on the positive side of that trait's fitted threshold.

The weights are graphological priors and are not estimated from data. Only the
threshold and its direction are fitted, two parameters per trait, which is what
keeps this usable at n = 29 -- fitting the weights themselves overfits well
before it helps. The direction matters because a prior can be inverted relative
to the data: without it, a trait whose score runs the wrong way is predicted
backwards on every sample rather than merely being uninformative.

The top and bottom margin classes describe distance from the page edge, so they
can only be set from annotations of uncropped pages. Detectors run on
text-cropped scans leave them at zero, which this module treats as undetermined.

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
    """Fitted thresholds and directions, or a zero threshold if none are fitted."""
    if not os.path.exists(path):
        return pd.DataFrame({"threshold": 0.0, "direction": 1}, index=TRAITS)
    return pd.read_csv(path, index_col="trait").reindex(TRAITS)


def score(features, weights=None):
    """Summed trait weights for a frame of 0/1 margin flags.

    Uses the margin columns common to both inputs, so a feature set missing the
    top and bottom classes scores on the remaining ones without a separate path.
    """
    if weights is None:
        weights = load_weights()
    cols = [c for c in weights.index if c in features.columns]
    return features[cols].dot(weights.loc[cols])


def fit_thresholds(features, labels, weights=None, signed=False):
    """Per-trait threshold and direction that best reproduce `labels`.

    `labels` columns must be in TRAITS order; the label file uses its own
    spellings, so it is matched by position rather than by name.
    """
    s = score(features, weights)
    out = []
    for i in range(len(TRAITS)):
        col, y = s.iloc[:, i].values, labels.iloc[:, i].values
        best = max(((((col > t) if d == 1 else (col < t)) == y).mean(), t, d)
                   for d in ((1, -1) if signed else (1,)) for t in THRESHOLD_GRID)
        out.append({"threshold": best[1], "direction": best[2]})
    return pd.DataFrame(out, index=TRAITS)


def predict(features, weights=None, thresholds=None):
    """Binary trait predictions from the fitted threshold and direction."""
    if thresholds is None:
        thresholds = load_thresholds()
    s = score(features, weights)
    out = {}
    for t in TRAITS:
        th, d = thresholds.loc[t, "threshold"], thresholds.loc[t, "direction"]
        out[t] = ((s[t] > th) if d == 1 else (s[t] < th)).astype(int)
    return pd.DataFrame(out, index=s.index)


if __name__ == "__main__":
    # Fitted on the expert margin annotations, which are the inputs the weights
    # were written against and the only source of top and bottom margin classes.
    margins = pd.read_csv(os.path.join(HERE, "features_manual_29.csv"))
    margins.columns = [c.upper() for c in margins.columns]
    margins = margins.rename(columns={"RLM": "RFLM"})
    labels = pd.read_csv(os.path.join(HERE, "labels_traits_29.csv"))

    th = fit_thresholds(margins, labels)
    th.rename_axis("trait").to_csv(THRESHOLDS_CSV)
    print(f"fitted thresholds -> {os.path.basename(THRESHOLDS_CSV)}")
    print(th.to_string())
