# Focused Inflation Decomposition

The implementation covers every model geography using five exhaustive macro
buckets:

- food
- energy
- shelter
- core goods
- core services excluding housing

The country/source contracts are:

- US: annual CPI-U relative-importance weights from BLS
- Norway: hierarchy-safe CPI frontier and annual weights from SSB table 14700
- Germany, France, Italy, Spain, euro area, Denmark, and Sweden: HICP special
  aggregates from the ECB Data Portal and annual weights from Eurostat
  `prc_hicp_iw`
- UK: CPI special aggregates and annual weights from the ONS MM23 database

The US annual weights are CPI-U relative-importance values from the U.S. Bureau
of Labor Statistics. Food, energy, shelter, and core goods use published
component indexes. Core services excluding housing is an accounting residual,
so the five estimated contributions reconcile exactly to headline YoY
inflation.

For Norway, Statistics Norway table 14700 supplies the COICOP 2018 hierarchy,
component indexes, and annual per-mille weights. The engine selects one
non-overlapping frontier. Where SSB publishes detailed weights but no usable
detailed index, it selects the published parent once. This keeps the contract
broad and auditable instead of mixing overlapping parents and children.
Published one-decimal weights are normalized only after the frontier validates.
The one weighted child without a published index, other personal-transport
services, retains its own weight and uses the nearest parent rate as an explicit
`rate_source_component_code` proxy; motor fuels remain in energy.

For the harmonised European contract, food, energy, non-energy industrial
goods, and services are the four published exhaustive HICP special aggregates.
Actual rents are split out of services as shelter. This avoids traversing
overlapping detailed COICOP parents and children. Denmark and Sweden use HICP
components for decomposition comparability while retaining their national
headline/core series elsewhere in the model.

The UK uses the equivalent CPI special aggregates from ONS. The ONS services
weight is derived as all items less food, energy, and non-energy industrial
goods; actual rents are then removed to form core services excluding housing.
Since the ONS annual MM23 weight series represents the second, February-to-
December update under the double-update method introduced in 2017, January is
left unavailable rather than incorrectly assigned February weights.

The output includes weighted basket shares above 2%, 3%, and 4%, weighted
inflation breadth, services-versus-goods pressure, a heuristic cyclical versus
supply-sensitive split, and concentration of absolute contributions.

```bash
PYTHONPATH=src python3 -m inflation_engine.data.fetch_bls_weights \
  --reference-year 2025 \
  --snapshot data/reference/us_cpi_weights_2026.csv

PYTHONPATH=src python3 -m inflation_engine.data.fetch_ssb_weights \
  --reference-year 2026

PYTHONPATH=src python3 -m inflation_engine.data.fetch_eurostat_weights

PYTHONPATH=src python3 -m inflation_engine.data.fetch_ons_weights

PYTHONPATH=src python3 -m inflation_engine.features.decomposition
```

Outputs:

```text
data/processed/inflation_bucket_decomposition.parquet
data/processed/inflation_decomposition_summary.parquet
```

## Contribution interpretation

Direct bucket contributions use the transparent approximation:

```text
annual relative-importance weight × bucket YoY inflation / 100
```

Core services excluding housing is based on the published broad services rate
less shelter and receives the small fixed-weight/chain-index reconciliation
residual. The US residual is derived directly from headline. These are research
estimates, not official contribution series. Exact contributions across a
weight update require source-specific chaining in separate segments.

## Hierarchy guardrail

`data/reference/component_taxonomy.csv` separates source identifiers from
normalized labels and canonical macro buckets. It also records hierarchy
level, parent, effective dates, and whether a node is permitted in the
decomposition.

Overlapping aggregates are retained for analysis but marked as excluded. The
validator rejects more than one selected component for the same canonical
bucket and refuses to build a contract unless all five selected nodes have
effective annual weights.

## US PCE item decomposition

`src/inflation_engine/features/us_pce_item_decomposition.py` writes
`data/processed/us_pce_item_decomposition.parquet` from BEA's detail file. It
is independent of the US CPI decompositions and shares none of their inputs.

PCE is a chained Fisher index with monthly-changing weights, so it uses exact
Fisher contributions per monthly link (the method BEA uses for its own
contributions to percent change). Twelve links are chained into a YoY
contribution, each scaled by the aggregate's level at the start of the link
relative to t-12. The frontier is the deepest set of tree lines that have a
price and spending in both months of a link. It falls back to the nearest
covered ancestor where BEA has discontinued a split, for example the old
tenant-occupied homes and landlord durables split. "Less:" lines enter with
negative weights.

Core contributions use the same links against BEA's "PCE excluding food and
energy". Food and energy are the subtrees for food and beverages purchased for
off-premises consumption, gasoline and other energy goods, and electricity and
gas. Together they match BEA's own "PCE food and energy" spending line.

One reconciliation row per date holds the unexplained remainder, from
published-index rounding and BEA's aggregation details. It is about 0.01-0.02pp
in a typical month and up to about 0.2pp in 2020-21. Contributions plus the
reconciliation row equal headline (and core) PCE YoY exactly.

PCE item breadth (`build_item_breadth_us_pce` in `features/breadth.py`, rows
with `country_code = US_PCE` in `inflation_breadth_items.parquet`) uses the
same frontier, restated as a basket of positively weighted purchases. The NPISH
branch (gross output less receipts) collapses into BEA's single NPISH final
consumption line, and lines with a negative net sign or a negative spending
share are dropped. Examples are nonresidents' US travel spending and net
used-vehicle transactions. Weights are spending shares at t-12.

