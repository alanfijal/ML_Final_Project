import logging
from pathlib import Path

import pytest

from src.data import load_split, DATA_PATH
from src.logging_config import setup_logging

setup_logging(level=logging.WARNING)


# Expected values from Table 5 of Arias Chao et al. (2020).
EXPECTED_EOL = {
    2: 75, 5: 89, 10: 82, 16: 63, 18: 71, 20: 66,
    11: 59, 14: 76, 15: 67,
}
DEV_UNITS  = {2, 5, 10, 16, 18, 20}
TEST_UNITS = {11, 14, 15}

# Required scenario descriptors and physical sensors (paper Tables 1 & 2).
REQUIRED_W  = {"alt", "Mach", "TRA", "T2"}
REQUIRED_XS = {"Wf", "Nf", "Nc", "T24", "T30", "T48", "T50",
               "P15", "P21", "P24", "Ps30", "P40", "P50"}
# Virtual sensors that must NOT leak in by default (paper Table 3).
VIRTUAL_VARS = {"T40", "P30", "P45", "W21", "W22", "W25",
                "W31", "W32", "W48", "W50"}


@pytest.fixture(scope="module")
def h5_path() -> Path:
    """Locate the dataset file. Skip the suite if it isn't present —
    so CI doesn't fail on machines that don't have the 1 GB file."""
    candidates = list(DATA_PATH.glob("*.h5"))
    if not candidates:
        pytest.skip(f"No .h5 file found in {DATA_PATH}")
    if len(candidates) > 1:
        pytest.fail(f"Ambiguous: multiple .h5 files in {DATA_PATH}: {candidates}")
    return candidates[0]


@pytest.fixture(scope="module")
def dev_df(h5_path):
    return load_split(h5_path, split="dev")


@pytest.fixture(scope="module")
def test_df(h5_path):
    return load_split(h5_path, split="test")


class TestUnitComposition:
    def test_dev_units(self, dev_df):
        assert set(dev_df["unit"].unique()) == DEV_UNITS

    def test_test_units(self, test_df):
        assert set(test_df["unit"].unique()) == TEST_UNITS


class TestEndOfLife:
    """max(cycle) per unit must match Table 5 of the paper. If this breaks,
    either the dataset file changed or the loader is misaligning rows."""

    @pytest.mark.parametrize("unit,expected", [
        (u, e) for u, e in EXPECTED_EOL.items() if u in DEV_UNITS
    ])
    def test_dev_eol(self, dev_df, unit, expected):
        actual = int(dev_df.loc[dev_df["unit"] == unit, "cycle"].max())
        assert actual == expected, f"unit {unit}: max cycle {actual} != {expected}"

    @pytest.mark.parametrize("unit,expected", [
        (u, e) for u, e in EXPECTED_EOL.items() if u in TEST_UNITS
    ])
    def test_test_eol(self, test_df, unit, expected):
        actual = int(test_df.loc[test_df["unit"] == unit, "cycle"].max())
        assert actual == expected, f"unit {unit}: max cycle {actual} != {expected}"


class TestSampleCounts:
    """Loose bounds — paper reports 5.3M dev / 1.2M test."""

    def test_dev_size(self, dev_df):
        assert 5.0e6 < len(dev_df) < 5.6e6

    def test_test_size(self, test_df):
        assert 1.0e6 < len(test_df) < 1.4e6


class TestSchema:
    def test_required_columns_dev(self, dev_df):
        cols = set(dev_df.columns)
        assert REQUIRED_W.issubset(cols)
        assert REQUIRED_XS.issubset(cols)
        assert {"unit", "cycle", "RUL"}.issubset(cols)

    def test_virtuals_excluded_by_default(self, dev_df):
        """Construct validity: virtual sensors must not leak in unless
        the caller explicitly opts in via include_virtual=True."""
        leaked = VIRTUAL_VARS & set(dev_df.columns)
        assert not leaked, f"virtual sensors leaked into default load: {leaked}"

    def test_virtuals_included_when_requested(self, h5_path):
        df = load_split(h5_path, split="test", include_virtual=True)
        assert VIRTUAL_VARS & set(df.columns)


class TestRULMonotonicity:
    """RUL must decrease (weakly) within each unit as cycle increases.
    A failure here would mean the rows are scrambled."""

    def test_rul_decreasing_per_unit(self, dev_df):
        unit_df = dev_df[dev_df["unit"] == 2].sort_values("cycle")
        rul_per_cycle = unit_df.groupby("cycle")["RUL"].first()
        assert rul_per_cycle.is_monotonic_decreasing