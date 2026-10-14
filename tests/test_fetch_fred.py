import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from inflation_engine.data.fetch_fred import (
    BLS_EARLIEST_YEAR,
    US_CPI_SERIES,
    US_CPI_SPECIAL_AGGREGATE_SERIES,
    US_CPI_STRUCTURAL_SERIES,
    _bls_year_windows,
    build_us_cpi_series_registry,
    fetch_bls_cpi_series,
    fetch_bls_series_batch,
    fetch_fred_observations,
    fetch_us_cpi,
    normalize_bls_series,
    normalize_fred_series,
    save_us_cpi,
)
from inflation_engine.utils.schema import CPI_SCHEMA_COLUMNS


BLS_OBSERVATIONS = [
    {"year": "2026", "period": "M08", "periodName": "August", "value": "136.457"},
    {"year": "2026", "period": "M07", "periodName": "July", "value": "136.434"},
    {"year": "2026", "period": "M13", "periodName": "Annual", "value": "135.9"},
]


OBSERVATIONS = [
    {"date": "2024-01-01", "value": "300.1"},
    {"date": "2024-02-01", "value": "."},
    {"date": "2024-03-01", "value": "302.4"},
]


class FakeResponse:
    def __init__(self, payload, *, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResponse({"observations": OBSERVATIONS})


class FailingSession:
    def get(self, url, *, params, timeout):
        import requests

        raise requests.ConnectionError(f"request failed with api_key={params['api_key']}")


class NotFoundThenFakeSession:
    """A series that FRED doesn't mirror (400, "does not exist"), followed by
    normal successful responses for every other series."""

    def __init__(self, missing_series_id):
        self.missing_series_id = missing_series_id
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        if params["series_id"] == self.missing_series_id:
            return FakeResponse(
                {"error_code": 400, "error_message": "Bad Request. The series does not exist."},
                status_code=400,
                text="Bad Request. The series does not exist.",
            )
        return FakeResponse({"observations": OBSERVATIONS})


class FakeBlsResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeBlsSession:
    """Every requested series gets the fixture's 2026 observations back only
    for the window that actually covers 2026 -- like the real, non-
    overlapping year windows, exactly one window ever has data for a given
    date, so results don't collide across windows."""

    def __init__(self):
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append((url, json, timeout))
        covers_2026 = int(json["startyear"]) <= 2026 <= int(json["endyear"])
        data = BLS_OBSERVATIONS if covers_2026 else []
        series = [
            {"seriesID": series_id, "data": data} for series_id in json["seriesid"]
        ]
        return FakeBlsResponse({"status": "REQUEST_SUCCEEDED", "Results": {"series": series}})


class FailingBlsSession:
    def post(self, url, *, json, timeout):
        return FakeBlsResponse({"status": "REQUEST_NOT_PROCESSED", "message": ["bad key"]})


class TransientlyFailingBlsSession:
    """Raises a connection error on the first call, then succeeds -- proves
    the retry path recovers instead of failing the whole batch."""

    def __init__(self):
        self.calls = 0

    def post(self, url, *, json, timeout):
        import requests

        self.calls += 1
        if self.calls == 1:
            raise requests.ConnectionError("transient")
        series = [
            {"seriesID": series_id, "data": BLS_OBSERVATIONS}
            for series_id in json["seriesid"]
        ]
        return FakeBlsResponse({"status": "REQUEST_SUCCEEDED", "Results": {"series": series}})


class FetchFredTests(unittest.TestCase):
    def test_normalize_fred_series_uses_standard_schema(self):
        frame = normalize_fred_series(
            OBSERVATIONS,
            series_id="CPIAUCSL",
            series_type="headline",
            component="all_items",
        )

        self.assertEqual(frame.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(frame), 2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["date"]))
        self.assertEqual(frame["country_code"].unique().tolist(), ["US"])
        self.assertEqual(frame["value"].tolist(), [300.1, 302.4])
        self.assertEqual(frame["source_series_id"].unique().tolist(), ["CPIAUCSL"])
        self.assertEqual(
            frame["seasonal_adjustment"].unique().tolist(),
            ["seasonally_adjusted"],
        )

    def test_fetch_us_cpi_fetches_table_a_series(self):
        session = FakeSession()

        frame = fetch_us_cpi(api_key="test-key", session=session)

        self.assertEqual(len(session.calls), len(US_CPI_SERIES))
        self.assertEqual(
            {call[1]["series_id"] for call in session.calls},
            set(US_CPI_SERIES),
        )
        self.assertEqual(set(frame["series_type"]), {"headline", "core"})
        self.assertEqual(
            set(frame["component"]),
            {metadata["component"] for metadata in US_CPI_SERIES.values()},
        )
        self.assertEqual(len(frame), len(US_CPI_SERIES) * 2)

    def test_table_a_mapping_has_unique_components(self):
        components = [
            metadata["component"] for metadata in US_CPI_SERIES.values()
        ]

        # 21 fixed structural/aggregate series plus every item-level series
        # resolved in us_cpi_item_series_map.csv (see
        # build_us_cpi_item_series_map.py), minus the handful that are both.
        self.assertGreater(len(US_CPI_SERIES), 250)
        self.assertEqual(len(components), len(set(components)))
        self.assertEqual(
            {metadata["series_type"] for metadata in US_CPI_SERIES.values()},
            {"headline", "core"},
        )

    def test_build_us_cpi_series_registry_keeps_structural_component_on_collision(self):
        with TemporaryDirectory() as temporary_directory:
            item_map_path = Path(temporary_directory) / "us_cpi_item_series_map.csv"
            # "CUSR0000SAH1" is both a structural bucket (shelter) and would
            # otherwise be re-derived from the item map as "housing" -- the
            # structural entry must win, and everything else still merges in.
            pd.DataFrame(
                {
                    "level": [1, 3],
                    "label": ["Housing", "Rent of primary residence"],
                    "weight_pct": [44.469, 7.84],
                    "series_id": ["CUSR0000SAH1", "CUSR0000SEHA"],
                }
            ).to_csv(item_map_path, index=False)

            registry = build_us_cpi_series_registry(
                structural_series={
                    "CUSR0000SAH1": {"series_type": "core", "component": "shelter"}
                },
                special_aggregate_series={},
                item_series_map_path=item_map_path,
            )

        self.assertEqual(registry["CUSR0000SAH1"]["component"], "shelter")
        self.assertEqual(
            registry["CUSR0000SEHA"],
            {"series_type": "headline", "component": "rent_of_primary_residence"},
        )
        self.assertEqual(len(registry), 2)

    def test_special_aggregate_components_are_all_distinctly_prefixed(self):
        # These are BLS's published overlapping-combo series (Services less
        # rent of shelter, All items less shelter, etc.) -- reference-only,
        # never fed into the item frontier's weighted sum, so every one must
        # be identifiable as such by its component name alone.
        for metadata in US_CPI_SPECIAL_AGGREGATE_SERIES.values():
            self.assertTrue(metadata["component"].startswith("special_aggregate_"))
            self.assertEqual(metadata["series_type"], "headline")

    def test_special_aggregates_do_not_duplicate_structural_series(self):
        structural_ids = set(US_CPI_STRUCTURAL_SERIES)
        special_ids = set(US_CPI_SPECIAL_AGGREGATE_SERIES)
        self.assertEqual(structural_ids & special_ids, set())

    def test_special_aggregates_are_merged_into_the_full_registry(self):
        for series_id, metadata in US_CPI_SPECIAL_AGGREGATE_SERIES.items():
            self.assertEqual(US_CPI_SERIES[series_id], metadata)

    def test_build_us_cpi_series_registry_drops_a_special_aggregate_already_structural(self):
        with TemporaryDirectory() as temporary_directory:
            item_map_path = Path(temporary_directory) / "us_cpi_item_series_map.csv"
            pd.DataFrame(columns=["level", "label", "weight_pct", "series_id"]).to_csv(
                item_map_path, index=False
            )
            registry = build_us_cpi_series_registry(
                structural_series={
                    "CUSR0000SASLE": {
                        "series_type": "core",
                        "component": "services_less_energy_services",
                    }
                },
                special_aggregate_series={
                    "CUSR0000SASLE": {
                        "series_type": "headline",
                        "component": "special_aggregate_services_less_energy_services",
                    },
                    "CUSR0000SAS": {
                        "series_type": "headline",
                        "component": "special_aggregate_services",
                    },
                },
                item_series_map_path=item_map_path,
            )

        self.assertEqual(
            registry["CUSR0000SASLE"]["component"], "services_less_energy_services"
        )
        self.assertEqual(
            registry["CUSR0000SAS"]["component"], "special_aggregate_services"
        )
        self.assertEqual(len(registry), 2)

    def test_fetch_us_cpi_skips_a_series_fred_does_not_mirror_without_fallback(self):
        missing_series_id = next(iter(US_CPI_SERIES))
        session = NotFoundThenFakeSession(missing_series_id)

        frame = fetch_us_cpi(
            api_key="test-key", session=session, fetch_bls_fallback=False
        )

        self.assertEqual(len(session.calls), len(US_CPI_SERIES))
        self.assertNotIn(missing_series_id, set(frame["source_series_id"]))
        self.assertEqual(len(set(frame["source_series_id"])), len(US_CPI_SERIES) - 1)

    def test_fetch_us_cpi_recovers_missing_series_from_bls(self):
        missing_series_id = next(iter(US_CPI_SERIES))
        fred_session = NotFoundThenFakeSession(missing_series_id)
        bls_session = FakeBlsSession()

        frame = fetch_us_cpi(
            api_key="test-key",
            session=fred_session,
            bls_api_key="test-bls-key",
            bls_session=bls_session,
        )

        self.assertIn(missing_series_id, set(frame["source_series_id"]))
        recovered = frame.loc[frame["source_series_id"] == missing_series_id]
        self.assertEqual(set(recovered["source"]), {"BLS"})

    def test_bls_year_windows_cover_from_earliest_to_end_without_overlap(self):
        windows = _bls_year_windows(2026)
        self.assertEqual(windows[0][0], BLS_EARLIEST_YEAR)
        self.assertEqual(windows[-1][1], 2026)
        for (_, end), (next_start, _) in zip(windows, windows[1:]):
            self.assertEqual(next_start, end + 1)
        for start, end in windows:
            self.assertLessEqual(end - start + 1, 20)

    def test_fetch_bls_series_batch_rejects_more_than_25_series(self):
        with self.assertRaises(ValueError):
            fetch_bls_series_batch(
                [f"CUSR0000SE{i:02d}" for i in range(26)],
                "test-key",
                start_year=2020,
                end_year=2026,
                session=FakeBlsSession(),
            )

    def test_fetch_bls_series_batch_raises_on_failed_request(self):
        with self.assertRaisesRegex(ValueError, "did not succeed"):
            fetch_bls_series_batch(
                ["CUSR0000SEHA"],
                "test-key",
                start_year=2020,
                end_year=2026,
                session=FailingBlsSession(),
            )

    def test_fetch_bls_series_batch_retries_a_transient_failure(self):
        session = TransientlyFailingBlsSession()
        result = fetch_bls_series_batch(
            ["CUSR0000SEHA"],
            "test-key",
            start_year=2020,
            end_year=2026,
            session=session,
        )
        self.assertEqual(session.calls, 2)
        self.assertEqual(result["CUSR0000SEHA"], BLS_OBSERVATIONS)

    def test_normalize_bls_series_drops_annual_average_and_sets_seasonal_flag(self):
        frame = normalize_bls_series(
            BLS_OBSERVATIONS,
            series_id="CUUR0000SEHA",
            series_type="headline",
            component="rent_of_primary_residence",
        )

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["value"].tolist(), [136.434, 136.457])
        self.assertEqual(
            frame["seasonal_adjustment"].unique().tolist(), ["not_seasonally_adjusted"]
        )
        self.assertEqual(frame["source"].unique().tolist(), ["BLS"])

    def test_normalize_bls_series_handles_empty_observations(self):
        frame = normalize_bls_series(
            [],
            series_id="CUSR0000SEHA",
            series_type="headline",
            component="rent_of_primary_residence",
        )
        self.assertTrue(frame.empty)
        self.assertEqual(frame.columns.tolist(), CPI_SCHEMA_COLUMNS)

    def test_fetch_error_does_not_expose_api_key(self):
        with self.assertRaisesRegex(RuntimeError, "credentials were redacted") as context:
            fetch_fred_observations(
                "CPIAUCSL",
                "sensitive-key",
                session=FailingSession(),
            )

        self.assertNotIn("sensitive-key", str(context.exception))

    def test_save_us_cpi_writes_parquet(self):
        frame = normalize_fred_series(
            OBSERVATIONS,
            series_id="CPIAUCSL",
            series_type="headline",
            component="all_items",
        )

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "fred_us_cpi.parquet"
            destination = save_us_cpi(frame, output)
            reloaded = pd.read_parquet(destination)

        self.assertEqual(destination, output)
        self.assertEqual(reloaded.columns.tolist(), CPI_SCHEMA_COLUMNS)
        self.assertEqual(len(reloaded), 2)


if __name__ == "__main__":
    unittest.main()
