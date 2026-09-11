"""Tests for ccs_scripts.co2_dynamic_map.

These tests are organized to double as documentation for how the module works:

* Argument parser: what CLI flags are required/optional and their defaults.
* Config loading/validation: what a valid YAML config must look like.
* Pure helper functions: the small building blocks (thresholds, distances,
  normalization, aggregation) used by the calculation pipeline.
* Path resolution: how EGRID/UNRST paths are derived from CLI args vs YAML.
* End-to-end: running main_entry_point() against a real Eclipse case (the
  same "Reek" dataset used by tests/test_co2_plume_extent.py) to show how a
  YAML config translates into distance calculations and written target files.
"""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from ccs_scripts.co2_dynamic_map.manager import (
    _aggregate_scalar_values,
    _distance_2d_from_xy,
    _distance_to_line,
    _format_date_suffix,
    _format_report_date,
    _format_threshold_suffix,
    _load_config,
    _normalization_parameters,
    _normalize_optimization_direction,
    _normalized_value,
    _optimization_multiplier,
    _parse_string_or_list,
    _parse_thresholds,
    _resolve_egrid_path,
    _resolve_unrst_path,
    _scalar_value_for_type,
    _validate_config_schema,
    main_entry_point,
)
from ccs_scripts.co2_dynamic_map.parser import build_argument_parser

REEK_CASE = str(
    Path(__file__).parents[1]
    / "tests"
    / "data"
    / "reek"
    / "eclipse"
    / "model"
    / "2_R001_REEK-0"
)


class _Args:
    """Minimal stand-in for the argparse.Namespace consumed by the
    EGRID/UNRST path-resolution helpers, without needing a real CLI parse."""

    def __init__(
        self, egrid=None, unrst=None, case_name=None, output_date=None, step=None
    ):
        self.egrid = egrid
        self.unrst = unrst
        self.case_name = case_name
        self.output_date = output_date
        self.step = step


def _minimal_config(**overrides):
    config = {
        "property": "SGAS",
        "threshold": 0.1,
        "output_date": "2001-08-01",
        "distance_calculations": [
            {
                "type": "plume_extent",
                "output_name": "plume_dist",
                "optimization_direction": "max",
                "x": 0.0,
                "y": 0.0,
            }
        ],
    }
    config.update(overrides)
    return config


def _write_config(tmp_path, content):
    config_path = tmp_path / "config.yml"
    config_path.write_text(content)
    return config_path


# ---------------------------------------------------------------------------
# Argument parser: -c/-cn are required, everything else has sane defaults.
# ---------------------------------------------------------------------------


def test_parser_requires_config_and_case_name():
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_defaults():
    parser = build_argument_parser()
    args = parser.parse_args(["-c", "conf.yml", "-cn", "CASE"])
    assert args.config == Path("conf.yml")
    assert args.case_name == Path("CASE")
    assert args.egrid is None
    assert args.unrst is None
    assert args.output_date is None
    assert args.step is None
    assert args.lint is False


def test_parser_lint_flag():
    parser = build_argument_parser()
    args = parser.parse_args(["-c", "conf.yml", "-cn", "CASE", "--lint"])
    assert args.lint is True


def test_parser_skip_type_keeps_strings_instead_of_path():
    parser = build_argument_parser(skip_type=True)
    args = parser.parse_args(["-c", "conf.yml", "-cn", "CASE"])
    assert args.config == "conf.yml"
    assert args.case_name == "CASE"


# ---------------------------------------------------------------------------
# Config loading: YAML must be a mapping, and duplicate keys are rejected.
# ---------------------------------------------------------------------------


def test_load_config_reads_yaml(tmp_path):
    config_path = _write_config(tmp_path, "property: SGAS\nthreshold: 0.1\n")
    config = _load_config(config_path)
    assert config == {"property": "SGAS", "threshold": 0.1}


def test_load_config_rejects_non_mapping(tmp_path):
    config_path = _write_config(tmp_path, "- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        _load_config(config_path)


def test_load_config_rejects_duplicate_keys(tmp_path):
    config_path = _write_config(tmp_path, "property: SGAS\nproperty: SWAT\n")
    with pytest.raises(ValueError, match="Invalid YAML"):
        _load_config(config_path)


# ---------------------------------------------------------------------------
# Config schema validation: required/allowed keys at top level and per-calc.
# ---------------------------------------------------------------------------


def test_validate_config_schema_accepts_minimal_config():
    _validate_config_schema(_minimal_config())  # should not raise


def test_validate_config_schema_rejects_unknown_top_level_key():
    config = _minimal_config()
    config["unknown"] = 1
    with pytest.raises(ValueError, match="Unsupported top-level key"):
        _validate_config_schema(config)


def test_validate_config_schema_rejects_missing_required_top_level_key():
    config = _minimal_config()
    del config["threshold"]
    with pytest.raises(ValueError, match="Missing required top-level key"):
        _validate_config_schema(config)


def test_validate_config_schema_rejects_empty_distance_calculations():
    config = _minimal_config(distance_calculations=[])
    with pytest.raises(ValueError, match="non-empty list"):
        _validate_config_schema(config)


def test_validate_config_schema_rejects_calc_missing_required_key():
    config = _minimal_config()
    del config["distance_calculations"][0]["x"]
    with pytest.raises(ValueError, match="missing required key"):
        _validate_config_schema(config)


def test_validate_config_schema_rejects_unknown_calc_type():
    config = _minimal_config()
    config["distance_calculations"][0]["type"] = "circle"
    with pytest.raises(ValueError, match="unknown type"):
        _validate_config_schema(config)


# ---------------------------------------------------------------------------
# Small pure-logic helpers used throughout the calculation pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, []),
        ("SGAS", ["SGAS"]),
        ("SGAS, SWAT", ["SGAS", "SWAT"]),
        (["SGAS", "SWAT"], ["SGAS", "SWAT"]),
        (5, ["5"]),
    ],
)
def test_parse_string_or_list(value, expected):
    assert _parse_string_or_list(value) == expected


def test_parse_thresholds_parses_numbers():
    assert _parse_thresholds("0.1, 0.2") == [0.1, 0.2]


def test_parse_thresholds_rejects_non_numeric():
    with pytest.raises(ValueError):
        _parse_thresholds("abc")


@pytest.mark.parametrize(
    "threshold,expected",
    [
        (0.1, "0_1"),
        (5, "5_0"),
        (0.000000001, "0_000000001"),
        (-2.5, "-2_5"),
    ],
)
def test_format_threshold_suffix(threshold, expected):
    assert _format_threshold_suffix(threshold) == expected


@pytest.mark.parametrize(
    "direction,expected",
    [
        (None, "max"),
        ("max", "max"),
        ("Maximize", "max"),
        ("min", "min"),
        ("MINIMIZE", "min"),
    ],
)
def test_normalize_optimization_direction(direction, expected):
    assert _normalize_optimization_direction(direction, "ctx") == expected


def test_normalize_optimization_direction_rejects_invalid_value():
    with pytest.raises(ValueError, match="invalid optimization_direction"):
        _normalize_optimization_direction("sideways", "ctx")


def test_optimization_multiplier():
    assert _optimization_multiplier("max") == 1
    assert _optimization_multiplier("min") == -1


@pytest.mark.parametrize(
    "calc_type,expected",
    [
        ("plume_extent", 5.0),  # plume_extent picks the farthest point -> max
        ("point", 1.0),  # point/line care about the closest point -> min
        ("line", 1.0),
    ],
)
def test_scalar_value_for_type(calc_type, expected):
    values = np.array([1.0, 5.0, 2.0])
    assert _scalar_value_for_type(values, calc_type) == expected


def test_scalar_value_for_type_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown type"):
        _scalar_value_for_type(np.array([1.0]), "circle")


def test_distance_2d_from_xy():
    centers = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    distances = _distance_2d_from_xy(centers, 0.0, 0.0)
    np.testing.assert_allclose(distances, [0.0, 5.0])


def test_distance_to_line_zero_at_anchor_point():
    centers = np.array([[100.0, 100.0, 0.0]])
    distances = _distance_to_line(
        centers, angle_deg=0.0, x0=100.0, y0=100.0, line_length=50.0
    )
    np.testing.assert_allclose(distances, [0.0])


def test_distance_to_line_perpendicular_offset():
    # angle=0 means the line runs due north-south through (0, 0); a point
    # 10m to the east should be exactly 10m away from the line.
    centers = np.array([[10.0, 0.0, 0.0]])
    distances = _distance_to_line(
        centers, angle_deg=0.0, x0=0.0, y0=0.0, line_length=50.0
    )
    np.testing.assert_allclose(distances, [10.0])


def test_distance_to_line_clamps_to_segment_endpoint():
    # The line is finite (line_length=10 -> total length 20m), so a point far
    # beyond the segment's end is measured from the nearest endpoint, not the
    # infinite line.
    centers = np.array([[0.0, 1000.0, 0.0]])
    distances = _distance_to_line(
        centers, angle_deg=0.0, x0=0.0, y0=0.0, line_length=10.0
    )
    np.testing.assert_allclose(distances, [990.0])


def test_normalization_parameters_returns_none_without_obj_keys():
    assert _normalization_parameters({}, 1) == (None, None, None)


def test_normalization_parameters_requires_min_and_max_together():
    with pytest.raises(ValueError, match="obj_min.*obj_max"):
        _normalization_parameters({"obj_min": 0.0}, 1)


def test_normalization_parameters_requires_mean_or_ref():
    with pytest.raises(ValueError, match="obj_mean"):
        _normalization_parameters({"obj_min": 0.0, "obj_max": 1.0}, 1)


def test_normalization_parameters_accepts_obj_ref_as_mean():
    assert _normalization_parameters(
        {"obj_min": 0.0, "obj_max": 2.0, "obj_ref": 1.0}, 1
    ) == (0.0, 2.0, 1.0)


def test_normalization_parameters_rejects_equal_min_and_max():
    with pytest.raises(ValueError, match="must be different"):
        _normalization_parameters({"obj_min": 1.0, "obj_max": 1.0, "obj_mean": 1.0}, 1)


def test_normalized_value_passthrough_without_params():
    assert _normalized_value(5.0, None, None, None) == 5.0


def test_normalized_value_normalizes_between_min_and_max():
    # (raw - obj_mean) / (obj_max - obj_min)
    assert _normalized_value(1269.1237856341, 0.0, 2000.0, 1000.0) == pytest.approx(
        0.1345618928
    )


@pytest.mark.parametrize(
    "mode,values,expected",
    [
        ("max", [1.0, 3.0, 2.0], 3.0),
        ("min", [1.0, 3.0, 2.0], 1.0),
        ("mean", [1.0, 2.0, 3.0], 2.0),
    ],
)
def test_aggregate_scalar_values(mode, values, expected):
    assert _aggregate_scalar_values(values, mode) == expected


def test_aggregate_scalar_values_rejects_empty_list():
    with pytest.raises(ValueError, match="No scalar values"):
        _aggregate_scalar_values([], "max")


def test_aggregate_scalar_values_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported aggregation mode"):
        _aggregate_scalar_values([1.0], "median")


def test_format_report_date_handles_datetime_and_date():
    assert _format_report_date(datetime(2001, 8, 1, 12, 0)) == "2001-08-01"
    assert _format_report_date(date(2001, 8, 1)) == "2001-08-01"


def test_format_date_suffix_replaces_dashes():
    assert _format_date_suffix(date(2001, 8, 1)) == "2001_08_01"


# ---------------------------------------------------------------------------
# EGRID/UNRST path resolution: explicit args win over --case_name, which
# wins over the YAML config's own 'egrid'/'unrst'/'case' entries.
# ---------------------------------------------------------------------------


def test_resolve_egrid_path_prefers_explicit_egrid_arg():
    args = _Args(egrid="explicit.EGRID", case_name="CASE")
    assert _resolve_egrid_path(args, {"egrid": "config.EGRID"}) == "explicit.EGRID"


def test_resolve_egrid_path_falls_back_to_case_name():
    args = _Args(case_name="CASE")
    assert _resolve_egrid_path(args, {}) == "CASE.EGRID"


def test_resolve_egrid_path_falls_back_to_config_egrid():
    args = _Args()
    assert _resolve_egrid_path(args, {"egrid": "config.EGRID"}) == "config.EGRID"


def test_resolve_egrid_path_falls_back_to_config_case():
    args = _Args()
    assert _resolve_egrid_path(args, {"case": "CONFIGCASE"}) == "CONFIGCASE.EGRID"


def test_resolve_egrid_path_raises_when_nothing_provided():
    args = _Args()
    with pytest.raises(ValueError, match="Provide --egrid"):
        _resolve_egrid_path(args, {})


def test_resolve_unrst_path_returns_none_when_nothing_provided():
    # Unlike EGRID, UNRST is optional: a config without any 'property'
    # filtering never needs to open an UNRST file.
    args = _Args()
    assert _resolve_unrst_path(args, {}) is None


def test_resolve_unrst_path_prefers_explicit_unrst_arg():
    args = _Args(unrst="explicit.UNRST", case_name="CASE")
    assert _resolve_unrst_path(args, {}) == "explicit.UNRST"


# ---------------------------------------------------------------------------
# End-to-end: run main_entry_point() against the Reek Eclipse dataset also
# used by tests/test_co2_plume_extent.py, to show a full config -> output
# round trip on real simulation data.
# ---------------------------------------------------------------------------


def test_main_entry_point_plume_extent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(
        tmp_path,
        """
property: SGAS
threshold: 0.1
output_date: 2001-08-01
distance_calculations:
  - type: plume_extent
    output_name: plume_dist
    optimization_direction: max
    x: 462500.0
    y: 5933100.0
""",
    )
    rc = main_entry_point(["-c", str(config_path), "-cn", REEK_CASE])
    assert rc == 0

    value = float((tmp_path / "plume_dist").read_text())
    # Matches the max SGAS-plume distance from the same injection point/date
    # computed independently in
    # test_co2_plume_extent.py::test_calc_plume_extents.
    assert value == pytest.approx(1269.1237856341113, rel=1e-6)


def test_main_entry_point_point_and_line_use_min_distance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(
        tmp_path,
        """
property: SGAS
threshold: 0.1
output_date: 2001-08-01
distance_calculations:
  - type: point
    output_name: point_dist
    optimization_direction: min
    x: 467000.0
    y: 5934000.0
  - type: line
    output_name: line_dist
    optimization_direction: min
    x: 467000.0
    y: 5934000.0
    angle: 90
    line_length: 5000
""",
    )
    rc = main_entry_point(["-c", str(config_path), "-cn", REEK_CASE])
    assert rc == 0

    # optimization_direction=min flips the sign of the written scalar
    # (multiplier=-1), so the raw minimum distance is written as a negative
    # optimization target (i.e. maximizing it minimizes the real distance).
    point_value = float((tmp_path / "point_dist").read_text())
    assert point_value == pytest.approx(-4465.9534468947, rel=1e-6)

    line_value = float((tmp_path / "line_dist").read_text())
    assert line_value == pytest.approx(-757.6734829622, rel=1e-6)


def test_main_entry_point_normalizes_optimization_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(
        tmp_path,
        """
property: SGAS
threshold: 0.1
output_date: 2001-08-01
distance_calculations:
  - type: plume_extent
    output_name: norm_dist
    optimization_direction: max
    x: 462500.0
    y: 5933100.0
    obj_min: 0.0
    obj_max: 2000.0
    obj_mean: 1000.0
""",
    )
    rc = main_entry_point(["-c", str(config_path), "-cn", REEK_CASE])
    assert rc == 0

    value = float((tmp_path / "norm_dist").read_text())
    # (raw - obj_mean) / (obj_max - obj_min) = (1269.1237856341 - 1000) / 2000
    assert value == pytest.approx(0.1345618928, rel=1e-6)


def test_main_entry_point_lint_exits_without_running(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(
        tmp_path,
        """
property: SGAS
threshold: 0.1
output_date: 2001-08-01
distance_calculations:
  - type: plume_extent
    output_name: plume_dist
    optimization_direction: max
    x: 462500.0
    y: 5933100.0
""",
    )
    with pytest.raises(SystemExit) as exc_info:
        main_entry_point(["-c", str(config_path), "-cn", REEK_CASE, "--lint"])
    assert exc_info.value.code == 0
    assert not (tmp_path / "plume_dist").exists()


def test_main_entry_point_writes_polygons_when_requested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(
        tmp_path,
        """
property: SGAS
threshold: 0.1
output_date: 2001-08-01
writing_polygons: true
distance_calculations:
  - type: plume_extent
    output_name: plume_dist
    optimization_direction: max
    x: 462500.0
    y: 5933100.0
""",
    )
    rc = main_entry_point(["-c", str(config_path), "-cn", REEK_CASE])
    assert rc == 0

    polygon_path = tmp_path / "plume_dist_maxdist.pol"
    assert polygon_path.exists()
    lines = polygon_path.read_text().splitlines()
    # RMS-style polygon: start point, end point, "999.00 999.00 999.00" terminator.
    assert len(lines) == 3
    assert lines[-1] == "999.00 999.00 999.00"
