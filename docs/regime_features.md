# Inflation Features for the Macro Regime Model

## Purpose

This layer turns the analytical CPI panel into a compact country-month feature
contract. It does not classify latent regimes. The future regime engine should
estimate states jointly from inflation, growth, liquidity, volatility, and
market-price inputs.

Run:

```bash
PYTHONPATH=src python3 -m inflation_engine.features.regime_features
```

Output:

```text
data/processed/inflation_regime_features.parquet
```

## Feature contract

- `headline_yoy`, `core_yoy`: inflation level and persistence inputs.
- `headline_yoy_change_3m`, `core_yoy_change_3m`: seasonality-safe direction
  inputs, calculated with exact calendar lags.
- `headline_yoy_change_12m`, `core_yoy_change_12m`: slower disinflation or
  reacceleration inputs.
- `headline_core_gap`: core minus headline inflation.
- `inflation_level`: mean headline and core YoY inflation, requiring both.
- `inflation_direction_3m`: mean three-month change in headline and core YoY,
  requiring both.
- `headline_3m3m_ann_sa`, `core_3m3m_ann_sa`: fast momentum inputs exposed
  only for series that are seasonally adjusted.

Quality flags state whether the headline and core source series are adjusted,
whether both short-horizon rates are usable, and whether the core level and
direction feature set is complete.

## Availability contract

The export separates `observation_date`, `publication_date`, and
`information_date`. Actual release dates from the release calendar
(`fetch_release_calendar.py`, from 2016 onward) are used where it has coverage.
Older observations receive a conservative two-month information
lag and are labeled `conservative_month_lag`.

Historical vintage snapshots are still unavailable. The timing contract avoids
observation-month look-ahead, but a regime backtest must not claim vintage-safe
real-time performance until revision histories are stored.

The detailed Nordic component data mixes classification levels. Component
breadth and contribution measures must not be calculated until components are
assigned comparable hierarchy levels and paired with expenditure weights.

## Recommended next architecture

Build each domain as a separate, versioned feature block with a common monthly
contract:

```text
observation_date + information_date + geography + feature values + quality flags
```

Suggested order:

1. Finish point-in-time inflation metadata and component hierarchy.
2. Build a growth block using activity levels, momentum, and surprises.
3. Build a liquidity/policy block using rates, curve, credit, money, and
   funding conditions.
4. Build a market-state block using volatility, trend, correlation, breadth,
   and cross-asset prices.
5. Join blocks through an as-of feature store and standardize using expanding
   or rolling history only.
6. Fit a small HMM baseline, preserving state probabilities rather than only
   hard labels.
7. Evaluate state stability and downstream asset behavior out of sample.

Do not put asset returns into the slow macro-state label if those same returns
will be evaluated conditionally afterward. A two-layer macro state and market
state design avoids that circularity.
