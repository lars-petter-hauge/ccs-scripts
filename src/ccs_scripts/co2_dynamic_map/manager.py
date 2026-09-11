from __future__ import annotations

import argparse
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
from resdata.grid import Grid
from resdata.resfile import ResdataFile
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from ccs_scripts.co2_dynamic_map.parser import build_argument_parser

ALLOWED_TOP_LEVEL_KEYS = {
    "property",
    "threshold",
    "output_date",
    "writing_polygons",
    "distance_calculations",
}
REQUIRED_TOP_LEVEL_KEYS = {
    "property",
    "threshold",
    "output_date",
    "distance_calculations",
}
ALLOWED_CALC_KEYS = {
    "type",
    "output_name",
    "optimization_direction",
    "angle",
    "line_length",
    "x",
    "y",
    "obj_min",
    "obj_max",
    "obj_mean",
    "obj_ref",
    "output_date",
    "step",
    "line_polygon_file",
}
REQUIRED_CALC_KEYS = {
    "type",
    "output_name",
    "optimization_direction",
    "x",
    "y",
}
SUPPORTED_CALC_TYPES = {"plume_extent", "point", "line"}


def _parse_string_or_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip()]
    return [str(value).strip()]


def _load_config(config_path: Path) -> dict:
    yaml_loader = YAML(typ="safe", pure=True)
    yaml_loader.allow_duplicate_keys = False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml_loader.load(handle) or {}
    except YAMLError as exc:
        raise ValueError(f"Invalid YAML config file '{config_path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Top-level YAML must be a mapping")

    return config


def _validate_config_schema(config: dict) -> None:
    unknown_top_level = set(config.keys()) - ALLOWED_TOP_LEVEL_KEYS
    if unknown_top_level:
        raise ValueError(
            "Unsupported top-level key(s): "
            f"{', '.join(sorted(unknown_top_level))}. "
            f"Allowed keys: {', '.join(sorted(ALLOWED_TOP_LEVEL_KEYS))}"
        )

    missing_top_level = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
    if missing_top_level:
        raise ValueError(
            f"Missing required top-level key(s): {', '.join(sorted(missing_top_level))}"
        )

    calculations = config.get("distance_calculations")
    if not isinstance(calculations, list) or len(calculations) == 0:
        raise ValueError("YAML must contain non-empty list 'distance_calculations'")

    for index, calc in enumerate(calculations, start=1):
        if not isinstance(calc, dict):
            raise ValueError(
                f"Calculation {index}: expected mapping, got {type(calc).__name__}"
            )

        unknown_calc = set(calc.keys()) - ALLOWED_CALC_KEYS
        if unknown_calc:
            raise ValueError(
                f"Calculation {index}: unsupported key(s): {', '.join(sorted(unknown_calc))}. "
                f"Allowed keys: {', '.join(sorted(ALLOWED_CALC_KEYS))}"
            )

        missing_calc = REQUIRED_CALC_KEYS - set(calc.keys())
        if missing_calc:
            raise ValueError(
                f"Calculation {index}: missing required key(s): {', '.join(sorted(missing_calc))}"
            )

        calc_type = str(calc.get("type", "")).strip().lower()
        if calc_type not in SUPPORTED_CALC_TYPES:
            raise ValueError(
                f"Calculation {index}: unknown type '{calc_type}'. "
                "Use plume_extent, point, or line"
            )


def _get_calc_output_name(calc: dict, index: int) -> str:
    raw_output_name = calc.get("output_name")
    output_name = str(raw_output_name).strip() if raw_output_name is not None else ""
    if not output_name:
        raise ValueError(f"Calculation {index}: missing required key 'output_name'")

    return output_name


def _parse_thresholds(value: object) -> list[float]:
    tokens = _parse_string_or_list(value)
    if not tokens:
        return []
    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid threshold value(s): {value}") from exc


def _format_threshold_suffix(threshold: float) -> str:
    text = f"{float(threshold):.12f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace(".", "_")


def _normalize_optimization_direction(direction: object, context: str) -> str:
    if direction is None:
        return "max"

    direction_text = str(direction).strip().lower()
    if direction_text in {"max", "maximize"}:
        return "max"
    if direction_text in {"min", "minimize"}:
        return "min"

    raise ValueError(
        f"{context}: invalid optimization_direction='{direction}'. "
        "Use one of: max, maximize, min, minimize"
    )


def _optimization_multiplier(optimization_direction: str) -> int:
    if optimization_direction == "min":
        return -1
    return 1


def _scalar_value_for_type(values: np.ndarray, calc_type: str) -> float:
    if calc_type == "plume_extent":
        return float(np.max(values))
    if calc_type in {"point", "line"}:
        return float(np.min(values))
    raise ValueError(f"Unknown type '{calc_type}'. Use plume_extent, point, or line")


def _polygon_z_value(centers: np.ndarray) -> float:
    return float(np.min(centers[:, 2]))


def _active_to_global_index(grid: Grid, active_index: int) -> int:
    """Map active index to global index for the loaded grid."""
    if hasattr(grid, "get_global_index"):
        return int(grid.get_global_index(active_index=active_index))
    if hasattr(grid, "actnum_indices"):
        return int(grid.actnum_indices[active_index])
    return int(active_index)


def _print_minimum_z_cell_info(grid: Grid, centers: np.ndarray) -> None:
    min_z_active_index = int(np.argmin(centers[:, 2]))
    min_z_global_index = _active_to_global_index(grid, min_z_active_index)
    i_idx, j_idx, k_idx = grid.get_ijk(active_index=min_z_active_index)
    x_cell, y_cell, z_cell = grid.get_xyz(active_index=min_z_active_index)
    print(
        "Minimum polygon z reference cell: "
        f"active_index={min_z_active_index}, "
        f"global1={min_z_global_index + 1}, "
        f"ijk1=({i_idx + 1},{j_idx + 1},{k_idx + 1}), "
        f"xyz=({x_cell:.6f},{y_cell:.6f},{z_cell:.6f})"
    )


def _distance_2d_from_xy(centers: np.ndarray, x0: float, y0: float) -> np.ndarray:
    return np.sqrt((centers[:, 0] - x0) ** 2 + (centers[:, 1] - y0) ** 2)


def _distance_to_line(
    centers: np.ndarray,
    angle_deg: float,
    x0: float,
    y0: float,
    line_length: float,
) -> np.ndarray:
    """Distance from each cell centre to a finite line segment."""
    theta = math.radians(angle_deg % 360)
    dx = math.sin(theta)
    dy = math.cos(theta)
    half_len = float(line_length)

    ax = x0 - half_len * dx
    ay = y0 - half_len * dy
    bx = x0 + half_len * dx
    by = y0 + half_len * dy

    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0.0:
        return np.sqrt((centers[:, 0] - x0) ** 2 + (centers[:, 1] - y0) ** 2)

    apx = centers[:, 0] - ax
    apy = centers[:, 1] - ay
    t = (apx * abx + apy * aby) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)

    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return np.sqrt((centers[:, 0] - closest_x) ** 2 + (centers[:, 1] - closest_y) ** 2)


def _resolve_egrid_path(args: argparse.Namespace, config: dict) -> str:
    if args.egrid:
        return str(args.egrid)

    if args.case_name:
        return str(args.case_name) + ".EGRID"

    if "egrid" in config:
        return str(config["egrid"])

    if "case" in config:
        return str(config["case"]) + ".EGRID"

    raise ValueError("Provide --egrid, or add 'egrid' (or 'case') in YAML")


def _resolve_unrst_path(args: argparse.Namespace, config: dict) -> str | None:
    if args.unrst:
        return str(args.unrst)

    if args.case_name:
        return str(args.case_name) + ".UNRST"

    if "unrst" in config:
        return str(config["unrst"])

    if "case" in config:
        return str(config["case"]) + ".UNRST"

    return None


def _select_step_index(
    unrst: ResdataFile,
    args: argparse.Namespace,
    config: dict,
    output_date_override: object = None,
    step_override: int | None = None,
) -> int:
    if output_date_override is not None:
        raw_date = output_date_override
    elif args.output_date:
        raw_date = args.output_date
    else:
        date_list = _parse_string_or_list(config.get("output_date"))
        raw_date = date_list[0] if date_list else None

    if raw_date is not None:
        output_date_str = str(raw_date).strip()
        lowered_output_date = output_date_str.lower()
        if lowered_output_date in {"min", "max", "mean"}:
            raw_date = None
        else:
            report_dates = [
                _format_report_date(report_date) for report_date in unrst.report_dates
            ]
            try:
                return report_dates.index(output_date_str)
            except ValueError as exc:
                raise ValueError(
                    f"output_date='{output_date_str}' not found in UNRST report dates"
                ) from exc

    requested_step = step_override if step_override is not None else args.step
    if requested_step is None:
        requested_step = int(config.get("step", -1))

    n_steps = len(unrst.report_dates)
    if n_steps == 0:
        raise ValueError("UNRST has no report dates/time steps")

    if requested_step < 0:
        requested_step = n_steps + requested_step

    if requested_step < 0 or requested_step >= n_steps:
        raise ValueError(
            f"step={requested_step} is out of range. Valid range: 0..{n_steps - 1} (or negative index)"
        )
    return requested_step


def _resolve_step_indices(
    unrst: ResdataFile,
    args: argparse.Namespace,
    config: dict,
    output_date_override: object = None,
    step_override: int | None = None,
) -> tuple[list[int], str]:
    n_steps = len(unrst.report_dates)
    if n_steps == 0:
        raise ValueError("UNRST has no report dates/time steps")

    if step_override is not None:
        requested_step = step_override
        if requested_step < 0:
            requested_step = n_steps + requested_step
        if requested_step < 0 or requested_step >= n_steps:
            raise ValueError(
                f"step={requested_step} is out of range. Valid range: 0..{n_steps - 1} (or negative index)"
            )
        return [requested_step], f"step={requested_step}"

    if output_date_override is not None:
        raw_date = output_date_override
    elif args.output_date:
        raw_date = args.output_date
    else:
        date_list = _parse_string_or_list(config.get("output_date"))
        raw_date = date_list[0] if date_list else None

    if raw_date is not None:
        output_date_text = str(raw_date).strip()
        lowered_output_date = output_date_text.lower()
        if lowered_output_date in {"min", "max", "mean"}:
            return list(range(n_steps)), lowered_output_date

        step_index = _select_step_index(
            unrst,
            args,
            config,
            output_date_override=output_date_text,
            step_override=None,
        )
        return [step_index], f"date={output_date_text}"

    step_index = _select_step_index(
        unrst,
        args,
        config,
        output_date_override=None,
        step_override=None,
    )
    return [step_index], "last"


def _format_report_date(report_date: object) -> str:
    if isinstance(report_date, datetime):
        return report_date.strftime("%Y-%m-%d")
    if isinstance(report_date, date):
        return report_date.isoformat()

    if hasattr(report_date, "strftime"):
        return report_date.strftime("%Y-%m-%d")

    return str(report_date)[:10]


def _format_date_suffix(report_date: object) -> str:
    return _format_report_date(report_date).replace("-", "_")


def _build_property_masks(
    centers: np.ndarray,
    args: argparse.Namespace,
    config: dict,
    output_date_override: object = None,
    step_override: int | None = None,
) -> dict[str, np.ndarray]:
    properties = _parse_string_or_list(config.get("property"))
    thresholds = _parse_thresholds(config.get("threshold"))
    output_dates_list = _parse_string_or_list(config.get("output_date"))

    if not properties:
        return {"all": np.ones(len(centers), dtype=bool)}

    unrst_path = _resolve_unrst_path(args, config)
    if unrst_path is None:
        raise ValueError(
            "Property filtering requested but no UNRST path found. Provide --unrst, --case_name, or 'unrst'/'case' in YAML."
        )

    unrst = ResdataFile(unrst_path)

    seen: list[tuple[str, float, object]] = []
    for i, prop in enumerate(properties):
        keyword = prop.strip().upper()
        threshold = thresholds[i] if i < len(thresholds) else 0.0
        if output_date_override is not None:
            pair_date = output_date_override
        elif i < len(output_dates_list):
            pair_date = output_dates_list[i]
        elif output_dates_list:
            pair_date = output_dates_list[-1]
        else:
            pair_date = None
        triplet = (
            keyword,
            threshold,
            str(pair_date).strip() if pair_date is not None else None,
        )
        if triplet not in seen:
            seen.append(triplet)

    masks: dict[str, np.ndarray] = {}
    for keyword, threshold, pair_date in seen:
        if keyword not in unrst:
            print(f"Warning: property '{keyword}' not found in UNRST; skipping.")
            continue
        step_idx = _select_step_index(
            unrst,
            args,
            config,
            output_date_override=pair_date,
            step_override=step_override,
        )
        values = np.asarray(unrst[keyword][step_idx].numpy_copy(), dtype=float)
        prop_mask = values >= threshold
        selected_count = int(np.count_nonzero(prop_mask))
        threshold_suffix = _format_threshold_suffix(threshold)
        report_date_suffix = _format_date_suffix(unrst.report_dates[step_idx])
        mask_key = (
            f"{keyword.lower()}_{threshold_suffix}_{report_date_suffix}"
            if len(seen) > 1
            else "all"
        )
        print(
            f"  {keyword} >= {threshold} @ {pair_date}: {selected_count} cells selected"
        )
        if selected_count == 0:
            print(
                "Warning: "
                f"Filter {keyword} >= {threshold} @ {pair_date} selected 0 cells. "
                "Check threshold/property/date."
            )
        masks[mask_key] = prop_mask

    return masks or {"all": np.zeros(len(centers), dtype=bool)}


def _build_property_masks_for_step(
    centers: np.ndarray,
    unrst: ResdataFile,
    config: dict,
    step_index: int,
) -> dict[str, np.ndarray]:
    properties = _parse_string_or_list(config.get("property"))
    thresholds = _parse_thresholds(config.get("threshold"))

    if not properties:
        return {"all": np.ones(len(centers), dtype=bool)}

    seen: list[tuple[str, float]] = []
    for i, prop in enumerate(properties):
        keyword = prop.strip().upper()
        threshold = thresholds[i] if i < len(thresholds) else 0.0
        pair = (keyword, threshold)
        if pair not in seen:
            seen.append(pair)

    masks: dict[str, np.ndarray] = {}
    for keyword, threshold in seen:
        if keyword not in unrst:
            print(f"Warning: property '{keyword}' not found in UNRST; skipping.")
            continue
        values = np.asarray(unrst[keyword][step_index].numpy_copy(), dtype=float)
        prop_mask = values >= threshold
        selected_count = int(np.count_nonzero(prop_mask))
        threshold_suffix = _format_threshold_suffix(threshold)
        mask_key = f"{keyword.lower()}_{threshold_suffix}" if len(seen) > 1 else "all"
        print(
            f"  {keyword} >= {threshold} @ {_format_report_date(unrst.report_dates[step_index])}: "
            f"{selected_count} cells selected"
        )
        if selected_count == 0:
            print(
                "Warning: "
                f"Filter {keyword} >= {threshold} @ {_format_report_date(unrst.report_dates[step_index])} selected 0 cells. "
                "Check threshold/property/date."
            )
        masks[mask_key] = prop_mask

    return masks or {"all": np.zeros(len(centers), dtype=bool)}


def _normalization_parameters(
    calc: dict, index: int
) -> tuple[float | None, float | None, float | None]:
    has_obj_min = "obj_min" in calc
    has_obj_max = "obj_max" in calc
    has_obj_mean = "obj_mean" in calc
    has_obj_ref = "obj_ref" in calc
    has_any_normalization = has_obj_min or has_obj_max or has_obj_mean or has_obj_ref

    if not has_any_normalization:
        return None, None, None

    if not has_obj_min or not has_obj_max:
        raise ValueError(
            f"Calculation {index}: when normalization is used, both 'obj_min' and 'obj_max' must be provided"
        )

    obj_mean_raw = calc.get("obj_mean", calc.get("obj_ref"))
    if obj_mean_raw is None:
        raise ValueError(
            f"Calculation {index}: when normalization is used, provide 'obj_mean' or 'obj_ref'"
        )

    try:
        obj_min = float(calc["obj_min"])
        obj_max = float(calc["obj_max"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Calculation {index}: obj_min and obj_max must be numeric"
        ) from exc

    try:
        obj_mean = float(obj_mean_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Calculation {index}: obj_mean (or obj_ref) must be numeric"
        ) from exc

    denominator = obj_max - obj_min
    if denominator == 0.0:
        raise ValueError(
            f"Calculation {index}: obj_max and obj_min must be different for normalization"
        )

    return obj_min, obj_max, obj_mean


def _normalized_value(
    raw_value: float,
    obj_min: float | None,
    obj_max: float | None,
    obj_mean: float | None,
) -> float:
    if obj_min is None or obj_max is None or obj_mean is None:
        return raw_value
    return (raw_value - obj_mean) / (obj_max - obj_min)


def _compute_all_distances(
    centers: np.ndarray,
    args: argparse.Namespace,
    config: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    calculations = config.get("distance_calculations")
    if not isinstance(calculations, list) or len(calculations) == 0:
        raise ValueError("YAML must contain non-empty list 'distance_calculations'")

    result: dict[str, np.ndarray] = {}
    property_masks_cache: dict[
        tuple[str | None, int | None], dict[str, np.ndarray]
    ] = {}
    column_active_indices: dict[str, np.ndarray] = {}

    for index, calc in enumerate(calculations, start=1):
        calc_type = str(calc.get("type", "")).lower()
        output_name = _get_calc_output_name(calc, index)
        calc_output_date = calc.get("output_date")
        calc_step = calc.get("step")
        calc_step_value = int(calc_step) if calc_step is not None else None
        cache_key = (
            str(calc_output_date).strip() if calc_output_date is not None else None,
            calc_step_value,
        )
        if cache_key not in property_masks_cache:
            property_masks_cache[cache_key] = _build_property_masks(
                centers,
                args,
                config,
                output_date_override=calc_output_date,
                step_override=calc_step_value,
            )
        property_masks = property_masks_cache[cache_key]

        calc_properties = list(property_masks.keys())

        for prop in calc_properties:
            if prop not in property_masks:
                raise ValueError(
                    f"Calculation {index}: property '{prop}' not available in global property selection"
                )

            prop_active_indices = np.flatnonzero(property_masks[prop])
            prop_centers = centers[prop_active_indices]
            if len(prop_centers) == 0:
                continue

            output_column = output_name if prop == "all" else f"{output_name}_{prop}"
            column_active_indices[output_column] = prop_active_indices

            if calc_type in ("plume_extent", "point"):
                if "x" not in calc or "y" not in calc:
                    raise ValueError(
                        f"Calculation {index}: '{calc_type}' requires x and y"
                    )
                x0 = float(calc["x"])
                y0 = float(calc["y"])
                result[output_column] = _distance_2d_from_xy(prop_centers, x0, y0)

            elif calc_type == "line":
                if "x" not in calc or "y" not in calc:
                    raise ValueError(
                        f"Calculation {index}: 'line' type requires both 'x' and 'y' "
                        "as the anchor point on the line"
                    )
                x0 = float(calc["x"])
                y0 = float(calc["y"])
                angle_deg = float(calc.get("angle", 0.0))
                line_length = float(calc.get("line_length", 500.0))
                result[output_column] = _distance_to_line(
                    prop_centers, angle_deg, x0, y0, line_length
                )
            else:
                raise ValueError(
                    f"Calculation {index}: unknown type '{calc_type}'. Use plume_extent, point, or line"
                )

    return result, column_active_indices


def _resolve_calc_scalar_value(
    distance_results: dict[str, np.ndarray], calc: dict, index: int
) -> float:
    output_name = _get_calc_output_name(calc, index)

    calc_type = str(calc.get("type", "")).lower()
    matching_columns = [
        column
        for column in distance_results
        if column == output_name or column.startswith(f"{output_name}_")
    ]
    if not matching_columns:
        raise ValueError(
            f"Calculation {index}: no output columns found for output_name='{output_name}'"
        )

    scalar_values = [
        _scalar_value_for_type(distance_results[column], calc_type)
        for column in matching_columns
    ]
    if calc_type == "plume_extent":
        return max(scalar_values)
    return min(scalar_values)


def _aggregate_scalar_values(values: list[float], mode: str) -> float:
    if not values:
        raise ValueError("No scalar values available for aggregation")
    if mode == "max":
        return max(values)
    if mode == "min":
        return min(values)
    if mode == "mean":
        return float(sum(values) / len(values))
    raise ValueError(f"Unsupported aggregation mode '{mode}'")


def _compute_distance_results_for_calc_step(
    centers: np.ndarray,
    unrst: ResdataFile,
    config: dict,
    calc: dict,
    index: int,
    step_index: int,
) -> dict[str, np.ndarray]:
    calc_type = str(calc.get("type", "")).lower()
    output_name = _get_calc_output_name(calc, index)
    property_masks = _build_property_masks_for_step(centers, unrst, config, step_index)
    result: dict[str, np.ndarray] = {}

    for prop in property_masks:
        prop_active_indices = np.flatnonzero(property_masks[prop])
        prop_centers = centers[prop_active_indices]
        if len(prop_centers) == 0:
            continue

        output_column = output_name if prop == "all" else f"{output_name}_{prop}"
        if calc_type in ("plume_extent", "point"):
            x0 = float(calc["x"])
            y0 = float(calc["y"])
            result[output_column] = _distance_2d_from_xy(prop_centers, x0, y0)
        elif calc_type == "line":
            x0 = float(calc["x"])
            y0 = float(calc["y"])
            angle_deg = float(calc.get("angle", 0.0))
            line_length = float(calc.get("line_length", 500.0))
            result[output_column] = _distance_to_line(
                prop_centers, angle_deg, x0, y0, line_length
            )
        else:
            raise ValueError(
                f"Calculation {index}: unknown type '{calc_type}'. Use plume_extent, point, or line"
            )

    return result


def _write_target_value(output_name: str, value: float) -> None:
    with open(output_name, "w", encoding="utf-8") as handle:
        handle.write(f"{value:.10f}\n")


def _write_optimization_targets(
    distance_results: dict[str, np.ndarray],
    config: dict,
    centers: np.ndarray,
    args: argparse.Namespace,
) -> None:
    calculations = config.get("distance_calculations")
    if not isinstance(calculations, list):
        return

    unrst_path = _resolve_unrst_path(args, config)
    unrst = ResdataFile(unrst_path) if unrst_path is not None else None

    for index, calc in enumerate(calculations, start=1):
        output_name = _get_calc_output_name(calc, index)
        obj_min, obj_max, obj_mean = _normalization_parameters(calc, index)

        raw_direction = calc.get("optimization_direction")
        optimization_direction = _normalize_optimization_direction(
            raw_direction, f"Calculation {index}"
        )
        multiplier = _optimization_multiplier(optimization_direction)

        calc_output_date = calc.get("output_date")
        calc_step = calc.get("step")
        calc_step_value = int(calc_step) if calc_step is not None else None

        step_indices = None
        selection = None
        if unrst is not None:
            step_indices, selection = _resolve_step_indices(
                unrst,
                args,
                config,
                output_date_override=calc_output_date,
                step_override=calc_step_value,
            )

        if selection in {"min", "max", "mean"} and step_indices is not None:
            per_step_results = [
                _compute_distance_results_for_calc_step(
                    centers, unrst, config, calc, index, step_index
                )
                for step_index in step_indices
            ]
            scalar_values = [
                _resolve_calc_scalar_value(step_result, calc, index)
                for step_result in per_step_results
                if step_result
            ]
            if not scalar_values:
                print(
                    "Warning: "
                    f"Calculation {index} skipped because no scalar values were produced "
                    f"for output_name='{output_name}'."
                )
                continue

            raw_value = _aggregate_scalar_values(scalar_values, selection)
            final_value = (
                _normalized_value(raw_value, obj_min, obj_max, obj_mean) * multiplier
            )
            _write_target_value(output_name, final_value)
            print(
                f"Wrote target {output_name}: "
                f"type={str(calc.get('type', '')).lower()}, "
                f"selection={selection}, direction={optimization_direction}, multiplier={multiplier:+d}, "
                f"original_value={raw_value:.10f}, "
                f"transform={'raw' if obj_min is None else 'normalized'}, "
                f"value={final_value:.10f}"
            )
            continue

        matching_columns = [
            column
            for column in distance_results
            if column == output_name or column.startswith(f"{output_name}_")
        ]
        if not matching_columns:
            print(
                "Warning: "
                f"Calculation {index} skipped because no output columns were produced "
                f"for output_name='{output_name}'."
            )
            continue

        raw_value = _resolve_calc_scalar_value(distance_results, calc, index)
        final_value = (
            _normalized_value(raw_value, obj_min, obj_max, obj_mean) * multiplier
        )
        _write_target_value(output_name, final_value)
        print(
            f"Wrote target {output_name}: "
            f"type={str(calc.get('type', '')).lower()}, "
            f"direction={optimization_direction}, multiplier={multiplier:+d}, original_value={raw_value:.10f}, "
            f"transform={'raw' if obj_min is None else 'normalized'}, "
            f"value={final_value:.10f}"
        )

        property_specific_columns = [
            col for col in matching_columns if col != output_name
        ]
        for column_name in property_specific_columns:
            calc_type = str(calc.get("type", "")).lower()
            raw_column_value = _scalar_value_for_type(
                distance_results[column_name], calc_type
            )
            final_column_value = (
                _normalized_value(raw_column_value, obj_min, obj_max, obj_mean)
                * multiplier
            )
            _write_target_value(column_name, final_column_value)
            print(
                f"Wrote target {column_name}: "
                f"type={str(calc.get('type', '')).lower()}, "
                f"direction={optimization_direction}, multiplier={multiplier:+d}, original_value={raw_column_value:.10f}, "
                f"transform={'raw' if obj_min is None else 'normalized'}, "
                f"value={final_column_value:.10f}"
            )


def _line_segment_from_calculation(
    calc: dict,
    z_value: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    x0 = float(calc["x"])
    y0 = float(calc["y"])
    angle_deg = float(calc.get("angle", 0.0))
    half_len = float(calc.get("line_length", 500.0))
    theta = math.radians(angle_deg % 360)
    dx = math.sin(theta)
    dy = math.cos(theta)
    return (
        (x0 - half_len * dx, y0 - half_len * dy, z_value),
        (x0 + half_len * dx, y0 + half_len * dy, z_value),
    )


def _closest_point_on_line_segment(
    calc: dict, x: float, y: float
) -> tuple[float, float]:
    """Closest point on the configured finite line segment to (x, y)."""
    x0 = float(calc["x"])
    y0 = float(calc["y"])
    angle_deg = float(calc.get("angle", 0.0))
    half_len = float(calc.get("line_length", 500.0))

    theta = math.radians(angle_deg % 360)
    dx = math.sin(theta)
    dy = math.cos(theta)

    ax = x0 - half_len * dx
    ay = y0 - half_len * dy
    bx = x0 + half_len * dx
    by = y0 + half_len * dy

    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0.0:
        return x0, y0

    apx = x - ax
    apy = y - ay
    t = (apx * abx + apy * aby) / ab_len_sq
    t = float(np.clip(t, 0.0, 1.0))
    return ax + t * abx, ay + t * aby


def _write_point_min_distance_polygons(
    config: dict,
    distance_results: dict[str, np.ndarray],
    column_active_indices: dict[str, np.ndarray],
    centers: np.ndarray,
    grid: Grid,
) -> None:
    """For each 'point' calculation, write a .pol segment from the anchor to the nearest cell."""
    calculations = config.get("distance_calculations")
    if not isinstance(calculations, list) or len(calculations) == 0:
        return

    origin_z_value = _polygon_z_value(centers)

    for index, calc in enumerate(calculations, start=1):
        if str(calc.get("type", "")).strip().lower() != "point":
            continue

        output_name = _get_calc_output_name(calc, index)

        x0 = float(calc["x"])
        y0 = float(calc["y"])

        matching_columns = [
            column
            for column in distance_results
            if column == output_name or column.startswith(f"{output_name}_")
        ]

        for column in matching_columns:
            if column not in distance_results:
                continue
            if column not in column_active_indices:
                continue

            distances = distance_results[column]
            if len(distances) == 0:
                continue

            prop_active_indices = column_active_indices[column]
            prop_centers = centers[prop_active_indices]
            endpoint_z_value = float(np.min(prop_centers[:, 2])) - 1.0
            argmin_idx = int(np.argmin(distances))
            closest = prop_centers[argmin_idx]
            closest_active_index = int(prop_active_indices[argmin_idx])
            closest_global_index = _active_to_global_index(grid, closest_active_index)
            closest_global_index_1 = closest_global_index + 1
            i_idx, j_idx, k_idx = grid.get_ijk(active_index=closest_active_index)
            i_res, j_res, k_res = i_idx + 1, j_idx + 1, k_idx + 1
            x_cell, y_cell, z_cell = grid.get_xyz(active_index=closest_active_index)

            output_path = f"{column}_mindist.pol"

            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(f"{x0:.6f} {y0:.6f} {origin_z_value:.6f}\n")
                handle.write(
                    f"{closest[0]:.6f} {closest[1]:.6f} {endpoint_z_value:.6f}\n"
                )
                handle.write("999.00 999.00 999.00\n")

            print(
                f"Wrote point min-distance polygon: {output_path} "
                f"(min dist = {float(distances[argmin_idx]):.2f} m, "
                f"origin_z={origin_z_value:.6f}, endpoint_z={endpoint_z_value:.6f}, "
                f"nearest cell active_index={closest_active_index}, "
                f"global1={closest_global_index_1}, "
                f"ijk1=({i_res},{j_res},{k_res}), "
                f"xyz=({x_cell:.2f},{y_cell:.2f},{z_cell:.2f}))"
            )


def _write_line_polygon_for_visualization(config: dict, centers: np.ndarray) -> None:
    calculations = config.get("distance_calculations")
    if not isinstance(calculations, list) or len(calculations) == 0:
        return

    z_value = _polygon_z_value(centers)
    for index, calc in enumerate(calculations, start=1):
        if str(calc.get("type", "")).strip().lower() != "line":
            continue

        output_name = _get_calc_output_name(calc, index)

        output_path = str(calc.get("line_polygon_file", f"{output_name}.pol")).strip()
        if not output_path:
            output_path = f"{output_name}.pol"

        point_a, point_b = _line_segment_from_calculation(calc, z_value)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(f"{point_a[0]:.6f} {point_a[1]:.6f} {point_a[2]:.6f}\n")
            handle.write(f"{point_b[0]:.6f} {point_b[1]:.6f} {point_b[2]:.6f}\n")
            handle.write("999.00 999.00 999.00\n")

        print(f"Wrote line polygon for visualization: {output_path}")


def _write_line_min_distance_polygons(
    config: dict,
    distance_results: dict[str, np.ndarray],
    column_active_indices: dict[str, np.ndarray],
    centers: np.ndarray,
    grid: Grid,
) -> None:
    """For each 'line' calculation, write a .pol segment for the minimum distance pair."""
    calculations = config.get("distance_calculations")
    if not isinstance(calculations, list) or len(calculations) == 0:
        return

    origin_z_value = _polygon_z_value(centers)

    for index, calc in enumerate(calculations, start=1):
        if str(calc.get("type", "")).strip().lower() != "line":
            continue

        output_name = _get_calc_output_name(calc, index)

        matching_columns = [
            column
            for column in distance_results
            if column == output_name or column.startswith(f"{output_name}_")
        ]

        for column in matching_columns:
            if column not in distance_results:
                continue
            if column not in column_active_indices:
                continue

            distances = distance_results[column]
            if len(distances) == 0:
                continue

            prop_active_indices = column_active_indices[column]
            prop_centers = centers[prop_active_indices]
            endpoint_z_value = float(np.min(prop_centers[:, 2])) - 1.0

            argmin_idx = int(np.argmin(distances))
            closest = prop_centers[argmin_idx]
            closest_active_index = int(prop_active_indices[argmin_idx])
            closest_global_index = _active_to_global_index(grid, closest_active_index)
            closest_global_index_1 = closest_global_index + 1
            i_idx, j_idx, k_idx = grid.get_ijk(active_index=closest_active_index)
            i_res, j_res, k_res = i_idx + 1, j_idx + 1, k_idx + 1
            x_cell, y_cell, z_cell = grid.get_xyz(active_index=closest_active_index)

            line_x, line_y = _closest_point_on_line_segment(
                calc, float(closest[0]), float(closest[1])
            )
            output_path = f"{column}_mindist.pol"

            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(f"{line_x:.6f} {line_y:.6f} {origin_z_value:.6f}\n")
                handle.write(
                    f"{closest[0]:.6f} {closest[1]:.6f} {endpoint_z_value:.6f}\n"
                )
                handle.write("999.00 999.00 999.00\n")

            print(
                f"Wrote line min-distance polygon: {output_path} "
                f"(min dist = {float(distances[argmin_idx]):.2f} m, "
                f"origin_z={origin_z_value:.6f}, endpoint_z={endpoint_z_value:.6f}, "
                f"nearest cell active_index={closest_active_index}, "
                f"global1={closest_global_index_1}, "
                f"ijk1=({i_res},{j_res},{k_res}), "
                f"xyz=({x_cell:.2f},{y_cell:.2f},{z_cell:.2f}))"
            )


def _write_plume_extent_max_distance_polygons(
    config: dict,
    distance_results: dict[str, np.ndarray],
    column_active_indices: dict[str, np.ndarray],
    centers: np.ndarray,
    grid: Grid,
) -> None:
    """For each 'plume_extent' calculation, write a .pol segment for the maximum distance pair."""
    calculations = config.get("distance_calculations")
    if not isinstance(calculations, list) or len(calculations) == 0:
        return

    origin_z_value = _polygon_z_value(centers)

    for index, calc in enumerate(calculations, start=1):
        if str(calc.get("type", "")).strip().lower() != "plume_extent":
            continue

        output_name = _get_calc_output_name(calc, index)

        x0 = float(calc["x"])
        y0 = float(calc["y"])

        matching_columns = [
            column
            for column in distance_results
            if column == output_name or column.startswith(f"{output_name}_")
        ]

        for column in matching_columns:
            if column not in distance_results:
                continue
            if column not in column_active_indices:
                continue

            distances = distance_results[column]
            if len(distances) == 0:
                continue

            prop_active_indices = column_active_indices[column]
            prop_centers = centers[prop_active_indices]
            endpoint_z_value = float(np.min(prop_centers[:, 2])) - 1.0

            argmax_idx = int(np.argmax(distances))
            farthest = prop_centers[argmax_idx]
            farthest_active_index = int(prop_active_indices[argmax_idx])
            farthest_global_index = _active_to_global_index(grid, farthest_active_index)
            farthest_global_index_1 = farthest_global_index + 1
            i_idx, j_idx, k_idx = grid.get_ijk(active_index=farthest_active_index)
            i_res, j_res, k_res = i_idx + 1, j_idx + 1, k_idx + 1
            x_cell, y_cell, z_cell = grid.get_xyz(active_index=farthest_active_index)

            output_path = f"{column}_maxdist.pol"
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(f"{x0:.6f} {y0:.6f} {origin_z_value:.6f}\n")
                handle.write(
                    f"{farthest[0]:.6f} {farthest[1]:.6f} {endpoint_z_value:.6f}\n"
                )
                handle.write("999.00 999.00 999.00\n")

            print(
                f"Wrote plume_extent max-distance polygon: {output_path} "
                f"(max dist = {float(distances[argmax_idx]):.2f} m, "
                f"origin_z={origin_z_value:.6f}, endpoint_z={endpoint_z_value:.6f}, "
                f"farthest cell active_index={farthest_active_index}, "
                f"global1={farthest_global_index_1}, "
                f"ijk1=({i_res},{j_res},{k_res}), "
                f"xyz=({x_cell:.2f},{y_cell:.2f},{z_cell:.2f}))"
            )


def main_entry_point(args=None):
    parser = build_argument_parser()
    options = parser.parse_args(args=args)

    if options.lint:
        parser.exit()

    config_path = Path(options.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = _load_config(config_path)
    _validate_config_schema(config)

    egrid_path = _resolve_egrid_path(options, config)
    grid = Grid(egrid_path)
    nactive = grid.get_num_active()
    centers = np.asarray(
        [grid.get_xyz(active_index=active_index) for active_index in range(nactive)],
        dtype=float,
    )

    print("--- GRID/UNRST properties ---")
    print(f"EGRID: {egrid_path}")
    print(f"Cells (active): {len(centers)}")
    _print_minimum_z_cell_info(grid, centers)

    distance_results, column_active_indices = _compute_all_distances(
        centers,
        options,
        config,
    )

    if config.get("writing_polygons", False):
        print("--- Distance visualization ---")
        _write_line_polygon_for_visualization(config, centers)
        _write_plume_extent_max_distance_polygons(
            config,
            distance_results,
            column_active_indices,
            centers,
            grid,
        )
        _write_line_min_distance_polygons(
            config, distance_results, column_active_indices, centers, grid
        )
        _write_point_min_distance_polygons(
            config, distance_results, column_active_indices, centers, grid
        )

    print("--- Distance summary ---")
    for column_name, values in distance_results.items():
        print(
            f"  {column_name}: min={np.min(values):.2f}, mean={np.mean(values):.2f}, max={np.max(values):.2f}"
        )

    print("--- Optimization scalars ---")
    _write_optimization_targets(distance_results, config, centers, options)

    return 0


if __name__ == "__main__":
    raise SystemExit(main_entry_point())
