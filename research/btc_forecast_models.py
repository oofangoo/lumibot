"""E021 model library: 91-day-ahead return distributions for a coin bought "now", from daily closes only.

Every model takes the closes up to day t (numpy array) and returns the distribution of log(close[t+H]/close[t]):
  * path models return simulated daily log-return paths, shape (n_sims, H)  -> can answer "did it dip -25% on the way?"
  * quantile models return only the quantiles at QLEVELS.
Models are fixed BEFORE any evaluation (no tuning on the walk-forward results):
  EMP    every past H-day window in the history (unconditional base rate; the baseline other models must beat)
  RW     random walk: normal, zero median, today's EWMA volatility (lambda 0.94), constant over the horizon
  BOOT   block bootstrap (blocks of 10 days) of the past daily log returns, INCLUDING the past drift (bull-market bias)
  BOOT0  the same with the past drift removed (median ~0)
  FHS    filtered historical simulation: past returns divided by their EWMA volatility, block-resampled, rescaled by a volatility that
         keeps updating along each path (volatility clustering); zero drift
  QLIN   linear quantile regression of the H-day return on trend / drawdown / momentum / volatility features (machine learning, walk-forward refit)
  QGBM   gradient-boosted quantile regression on the same features
  ENS    equal-weight average of all model quantiles
Plain numpy / scikit-learn: LumiBot has no multi-horizon distribution forecaster (it provides the data, backtests and charts).
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

H = 91
FEE = 0.0025
NET = (1 - FEE) / (1 + FEE)  # buy once, sell once, after the horizon
LAM = 0.94
BLOCK = 10
QLEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
PATH_MODELS = ("RW", "BOOT", "BOOT0", "FHS")
ML_MODELS = ("QLIN", "QGBM")
ALL_MODELS = ("EMP",) + PATH_MODELS + ML_MODELS


# ------------------------------------------------------------------------------------------------ helpers
def log_returns(closes):
    return np.diff(np.log(np.asarray(closes, dtype=float)))


def ewma_variance(r, lam=LAM):
    """s2[i] = variance forecast for r[i] from r[:i]; also the forecast for the day after the last return."""
    s2 = np.empty(len(r) + 1)
    s2[0] = np.var(r[:30]) if len(r) >= 30 else np.var(r)
    for i in range(len(r)):
        s2[i + 1] = lam * s2[i] + (1 - lam) * r[i] ** 2
    return s2  # len(r)+1: s2[len(r)] is the forecast for tomorrow


def block_indices(rng, n, n_sims, horizon=H, block=BLOCK):
    nb = -(-horizon // block)
    starts = rng.integers(0, n - block + 1, (n_sims, nb))
    return (starts[:, :, None] + np.arange(block)).reshape(n_sims, -1)[:, :horizon]


def quantiles_of(log_ret_end):
    return np.quantile(log_ret_end, QLEVELS)


# ------------------------------------------------------------------------------------------------ models (each returns (quantiles, paths_or_None))
def model_emp(closes, h=H):
    lc = np.log(np.asarray(closes, dtype=float))
    ret = lc[h:] - lc[:-h]
    return quantiles_of(ret), None


def model_rw(closes, rng, n_sims=2000, h=H):
    r = log_returns(closes)
    sigma = np.sqrt(ewma_variance(r)[-1])
    paths = np.cumsum(rng.standard_normal((n_sims, h)) * sigma, axis=1)
    return quantiles_of(paths[:, -1]), paths


def model_boot(closes, rng, n_sims=2000, h=H, demean=False):
    r = log_returns(closes)
    if demean:
        r = r - r.mean()
    paths = np.cumsum(r[block_indices(rng, len(r), n_sims, h)], axis=1)
    return quantiles_of(paths[:, -1]), paths


def model_fhs(closes, rng, n_sims=2000, h=H):
    r = log_returns(closes)
    s2 = ewma_variance(r)
    z = r / np.sqrt(s2[:-1])
    z = (z - z.mean()) / z.std()
    zs = z[block_indices(rng, len(z), n_sims, h)]
    var = np.full(n_sims, s2[-1])
    paths = np.empty((n_sims, h))
    acc = np.zeros(n_sims)
    for k in range(h):
        ret = np.sqrt(var) * zs[:, k]
        acc += ret
        paths[:, k] = acc
        var = LAM * var + (1 - LAM) * ret ** 2
    return quantiles_of(paths[:, -1]), paths


# ------------------------------------------------------------------------------------------------ machine-learning features
FEATURES = ("ret30", "ret90", "ret180", "dist50", "dist200", "drawdown", "logvol30")
MIN_HISTORY = 200


def feature_matrix(closes):
    """Row t uses closes[:t+1] only. NaN for t < MIN_HISTORY."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    lc = np.log(c)
    r = np.diff(lc, prepend=np.nan)
    out = np.full((n, len(FEATURES)), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(c)])
    peak = np.maximum.accumulate(c)
    for t in range(MIN_HISTORY, n):
        sma50 = (cs[t + 1] - cs[t - 49]) / 50
        sma200 = (cs[t + 1] - cs[t - 199]) / 200
        vol30 = np.std(r[t - 29: t + 1], ddof=1) * np.sqrt(365)
        out[t] = [lc[t] - lc[t - 30], lc[t] - lc[t - 90], lc[t] - lc[t - 180], np.log(c[t] / sma50), np.log(c[t] / sma200), np.log(c[t] / peak[t]), np.log(vol30)]
    return out


class QuantileML:
    """Fits one regressor per quantile level. kind: 'lin' (linear quantile regression) or 'gbm' (gradient boosting). Output quantiles are sorted."""

    def __init__(self, kind):
        self.kind = kind
        self.models, self.mu, self.sd = [], None, None

    def fit(self, X, y):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import QuantileRegressor

        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        self.models = []
        for q in QLEVELS:
            if self.kind == "lin":
                m = QuantileRegressor(quantile=float(q), alpha=1e-3, solver="highs")
            else:
                m = GradientBoostingRegressor(loss="quantile", alpha=float(q), n_estimators=80, learning_rate=0.05, max_depth=2, min_samples_leaf=50,
                                              subsample=0.7, random_state=0)
            self.models.append(m.fit(Xs, y))
        return self

    def predict(self, x):
        xs = ((np.atleast_2d(x) - self.mu) / self.sd)
        return np.sort(np.array([m.predict(xs)[0] for m in self.models]))


def training_set(closes, feats, t, h=H):
    """Rows i with i >= MIN_HISTORY and i + h <= t (the outcome was already known on day t: no look-ahead)."""
    lc = np.log(np.asarray(closes, dtype=float))
    idx = np.arange(MIN_HISTORY, t - h + 1)
    return feats[idx], lc[idx + h] - lc[idx]


def path_stats(paths, thresholds_down=(0.15, 0.25, 0.35), thresholds_up=(0.20,)):
    """Probability that the (gross) price touched a level at any time within the horizon, from simulated paths."""
    low, high = np.exp(paths.min(1)) - 1, np.exp(paths.max(1)) - 1
    out = {f"touch_down_{int(d * 100)}": float((low <= -d).mean()) for d in thresholds_down}
    out.update({f"touch_up_{int(u * 100)}": float((high >= u).mean()) for u in thresholds_up})
    return out
