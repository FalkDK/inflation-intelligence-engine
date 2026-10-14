"""BEA PCE product tree helpers shared by ingestion and decomposition.

Nodes are keyed by Table 2.4.5U line number (see fetch_bea_pce.py for why
never by series code). A line's net sign is the product of the
``aggregation_sign`` of itself and every ancestor, so "Less: ..." branches
subtract from the aggregate they sit under.
"""

import numpy as np
import pandas as pd


ROOT_LINE = 1


class PceTree:
    def __init__(self, structure: pd.DataFrame):
        self.structure = structure.drop_duplicates("table_line").set_index("table_line")
        self.parent = {
            int(line): (None if pd.isna(parent) else int(parent))
            for line, parent in self.structure["parent_line"].items()
        }
        self.children: dict[int, list[int]] = {}
        for line, parent in self.parent.items():
            if parent is not None:
                self.children.setdefault(parent, []).append(line)
        signs = self.structure["aggregation_sign"]
        self.net_sign = {
            line: int(np.prod([signs[ancestor] for ancestor in self.ancestry(line)]))
            for line in self.parent
        }

    def ancestry(self, line: int) -> list[int]:
        """``line`` itself followed by each ancestor up to the root."""
        chain = []
        current: int | None = line
        while current is not None:
            chain.append(current)
            current = self.parent[current]
        return chain

    def lines_for_nominal_series(self, series_ids) -> set[int]:
        return set(
            self.structure.index[self.structure["nominal_series_id"].isin(list(series_ids))]
        )

    def subtree(self, roots: set[int]) -> set[int]:
        """Every line at or below any of ``roots``."""
        return {line for line in self.parent if roots & set(self.ancestry(line))}

    def strict_ancestors(self, roots: set[int]) -> set[int]:
        return {ancestor for root in roots for ancestor in self.ancestry(root)[1:]}

    def frontier(self, covered: frozenset[int], *, start: int = ROOT_LINE) -> list[int]:
        """The deepest lines below ``start`` whose data is ``covered``,
        falling back to the nearest covered ancestor wherever a subtree
        isn't fully covered."""
        cache: dict[int, bool] = {}

        def fully_covered(line: int) -> bool:
            if line not in cache:
                children = self.children.get(line, [])
                cache[line] = line in covered or (
                    bool(children) and all(fully_covered(child) for child in children)
                )
            return cache[line]

        selected: list[int] = []

        def descend(line: int) -> None:
            children = self.children.get(line, [])
            if children and all(fully_covered(child) for child in children):
                for child in children:
                    descend(child)
            elif line in covered:
                selected.append(line)

        for child in self.children.get(start, []):
            descend(child)
        return selected


def link_coverage(prices: pd.DataFrame, nominal: pd.DataFrame) -> pd.DataFrame:
    """True where a line has a positive price and a spending level in both
    the month and the month before (i.e. a usable month-to-month link)."""
    available = prices.notna() & (prices > 0) & nominal.notna()
    return available & available.shift(1, fill_value=False)


def chained_fisher_index(
    tree: PceTree,
    prices: pd.DataFrame,
    nominal: pd.DataFrame,
    *,
    start: int,
    excluded_lines: set[int],
    base_year: int = 2017,
) -> pd.Series:
    """A monthly-chained Fisher price index over the frontier below
    ``start``, leaving out the subtrees under ``excluded_lines`` (and any
    line straddling them), rebased so ``base_year`` averages 100."""
    blocked = tree.subtree(excluded_lines) | tree.strict_ancestors(excluded_lines)
    coverage = link_coverage(prices, nominal)
    signs = pd.Series(tree.net_sign)
    relatives = pd.Series(np.nan, index=prices.index)
    frontier_cache: dict[frozenset[int], list[int]] = {}
    for position in range(1, len(prices.index)):
        date, previous = prices.index[position], prices.index[position - 1]
        covered = frozenset(coverage.columns[coverage.loc[date].to_numpy()])
        if covered not in frontier_cache:
            frontier_cache[covered] = tree.frontier(covered, start=start)
        lines = [line for line in frontier_cache[covered] if line not in blocked]
        if not lines:
            continue
        line_signs = signs.loc[lines].to_numpy()
        p0 = prices.loc[previous, lines].to_numpy(dtype=float)
        p1 = prices.loc[date, lines].to_numpy(dtype=float)
        q0 = nominal.loc[previous, lines].to_numpy(dtype=float) * line_signs / p0
        q1 = nominal.loc[date, lines].to_numpy(dtype=float) * line_signs / p1
        laspeyres = (p1 * q0).sum() / (p0 * q0).sum()
        paasche = (p1 * q1).sum() / (p0 * q1).sum()
        relatives.loc[date] = np.sqrt(laspeyres * paasche)
    # Chain only within the contiguous run of valid links ending at the
    # latest month, so a gap never silently splices two histories together.
    valid = relatives.notna()
    if not valid.any():
        return pd.Series(dtype=float)
    last_gap = valid[~valid].index.max()
    run = relatives.loc[relatives.index >= last_gap].copy()
    run.iloc[0] = 1.0
    level = run.cumprod()
    base = level.loc[level.index.year == base_year]
    if base.empty:
        raise ValueError(f"Chained index does not cover base year {base_year}")
    return level / base.mean() * 100.0
