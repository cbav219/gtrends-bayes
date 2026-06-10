# gtrends-bayes — Results & Model Notes

A deep-dive companion to the top-level [`README.md`](../README.md): what the model
produces, how it performs, which predictors drive it, and the honest caveats that
come with the approach.

> **Scope.** A 41-predictor Google Trends library, preprocessed per the OECD
> methodology (Woloszko 2020), feeds a Bayesian Structural Time Series model
> (Steven L. Scott's R `bsts` via `rpy2`, per Varian 2023) that forecasts **HY / IG
> corporate-bond ETF prices** (HYG, LQD — proxies for the underlying OAS spreads) at
> five horizons from one week to six months. It is framed as a **Trends-driven risk
> overlay** on an AR(4) backbone — a *supplement* to, not a replacement for, the AR
> baseline.

---

## What the model produces

For each target × horizon × `as_of` date, a probabilistic forecast — median plus an
80% credible band — in both ETF-price space and OAS-implied bps. Example shape
(numbers vary with `as_of`):

```
HY  [transform=levels, cadence=weekly, α=0.954]
as_of: 2026-05-03    last observed: 79.71
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w          +1.40      -3.73      +6.01         81.12      75.98      85.72
2w          +1.11      -5.45      +7.16         80.82      74.27      86.87
1m          +0.85      -7.34      +9.21         80.56      72.37      88.92
1q          +1.13     -13.84     +14.75         80.84      65.87      94.47

IG  [transform=levels, cadence=weekly, α=1.206]
as_of: 2026-05-03    last observed: 108.31
horizon    Δ median      Δ 5%     Δ 95%  level median   level 5%  level 95%
1w          +1.59      -7.68     +10.52        109.89     100.63     118.83
1m          +0.65     -14.06     +17.81        108.96      94.25     126.12
1q          +0.52     -29.04     +30.16        108.82      79.27     138.47
```

Each ETF forecast is also re-expressed as **OAS-implied bps** via an empirical
regression baked into the model. A separate small-N BSTS fit (`HY_OAS` / `IG_OAS`)
forecasts OAS bps directly from the 156-week FRED OAS history as a second opinion —
when the two views disagree, the disagreement is itself honest information about
regime uncertainty.

A published **Trends Risk Index** (weekly + daily cadence) accompanies the forecasts:
a PM-facing z-score of inclusion-weighted Trends signal, evaluated for crisis recall
and incremental Granger causality over VIX.

---

## Headline performance

**Hit rate on cumulative direction** — the PM-relevant metric (not RMSE, which is a
fool's errand on a near-random-walk target):

| Target | Model | 1w | 1m | 1q | 6m |
|---|---|---:|---:|---:|---:|
| HY | BSTS (Trends) | 0.50 | 0.49 | **0.60** | **0.61** |
| HY | Naive RW | 0.49 | 0.42 | 0.30 | 0.20 |
| IG | BSTS (Trends) | 0.48 | 0.54 | 0.55 | 0.53 |
| IG | Naive RW | 0.42 | 0.51 | 0.49 | 0.45 |

**Key result:** BSTS (Trends) wins on direction at long horizons — **~3× the signal
of a naive random walk on HY at six months (0.61 vs 0.20)**, and 0.60 vs 0.30 at one
quarter. The Trends signal pays off precisely when the autoregressive backbone has the
least to say.

**Calibrated coverage.** A conformal α multiplier is learned in-sample to hit 80%
nominal coverage; **all 8 (target, model) cells land in [0.75, 0.85]**. v5 α: HY
BSTS = 0.954, IG BSTS = 1.206.

**Stability.** The v5 refit on the full 41-predictor universe lands within 0.01 of
v4's categories-only numbers at every horizon — the topic predictors mainly rearrange
which features get weight without changing the headline calibration.

---

## Which predictors drive it

Top spike-and-slab inclusion predictors (v5, 41 categories + topics):

| Target | Predictor | P(γ=1) | β̄ when included |
|---|---|---:|---:|
| HY | Economic crisis (topic) | 1.00 | −0.20 |
| HY | Unemployment benefits (topic) | 1.00 | −1.44 |
| HY | Recruitment & Staffing (cat) | 1.00 | +1.62 |
| HY | vix (control) | 0.89 | −0.93 |
| HY | ust2y10y_slope (control) | 0.79 | −1.05 |
| IG | ust10y (control) | 1.00 | −2.77 |
| IG | Unemployment benefits (topic) | 1.00 | −2.74 |
| IG | Yield curve (topic) | 0.99 | +1.25 |
| IG | Recruitment & Staffing (cat) | 0.81 | +1.93 |
| IG | VIX (topic) | 0.56 | −0.95 |

Topic predictors dominate the top-inclusion list on both targets — the headline v5
improvement over v4's categories-only fit. The signs are economically sensible
(search interest in crisis ↑ → spread proxy ↓; labor demand ↑ → spread ↓).

## Trends Risk Index — Granger vs VIX

| Cadence | Target | F | p-value | ΔR² |
|---|---|---:|---:|---:|
| Weekly | HY | 1.71 | 0.145 | 0.007 |
| Weekly | IG | **3.79** | **0.005** | 0.016 |
| Daily | HY | 0.41 | 0.802 | 0.000 |
| Daily | IG | 0.56 | 0.695 | 0.001 |

The **weekly IG** Risk Index Granger-causes Δlog(LQD) incrementally over VIX. Weekly
HY is weaker but directionally consistent. The weekly cadence is the PM-facing
default; read the daily index as a watchlist visualization, not a daily significance
test (see caveat 3).

---

## Honest caveats — read these before forecasting

1. **ETF proxies, not OAS.** The model trains on HYG / LQD ETF prices because FRED's
   aggregate ICE BAML OAS series start only May 2023 — too short for the 2008+ training
   window. Proxy quality (2023-05 → 2026-05): HYG ↔ HY OAS Pearson **−0.69** (defensible);
   LQD ↔ IG OAS Pearson **−0.24** (weak — LQD carries duration noise the Trends predictors
   don't model). **Expect IG forecasts to be noisier in the OAS sense than HY forecasts.**

2. **BSTS posteriors are in level space**, not log-returns. The fit is on raw ETF prices;
   the freeze step auto-detects this and sets `target_transform="levels"`. Forecasts are
   valid level-space numbers. Wiring `TargetTransform` end-to-end is future work.

3. **Daily-cadence signal is weaker than weekly.** Expanding the daily predictor set from
   19 (categories only) to 39 added more idiosyncratic high-frequency noise than incremental
   signal, and the daily Granger-vs-VIX significance disappeared on both targets. The weekly
   Risk Index is the headline cadence; weekly IG remains significant (p = 0.005).

4. **StackedResidual underperforms the AR(p) baseline on information coefficient** (Spearman
   0.04–0.34 vs 0.39–0.55). An honest null result, reported for transparency — the headline
   deliverable is BSTS (Trends) standalone, which still wins on hit rate at long horizons.

5. **Crisis recall: 33% (1 of 3).** The Risk Index spiked in the four weeks before the 2020-03
   COVID shock but missed the 2022-09 UK gilt and 2023-03 SVB episodes. Treat it as a slow-burn
   macro-stress signal, not a point-event predictor.

6. **The OAS-direct sub-model is small-N.** It is a BSTS fit on ~154 weekly FRED OAS
   observations (2023-05+) — much smaller than the ETF fit's 487-week, 41-predictor envelope.
   Use it as a *second opinion* alongside the OAS-implied translation, not as a sole oracle.

---

## Methodology references

- **Steven L. Scott** — Bayesian Structural Time Series (the R `bsts` package).
- **Hal Varian (2023)** — nowcasting / forecasting with Google Trends.
- **Nicolas Woloszko (2020), OECD** — search-data preprocessing methodology.
