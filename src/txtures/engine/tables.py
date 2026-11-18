"""Output tables: assembling them from results, and checking which are current.

An output table directory, `DERIVATIVES/NAME/WORKFLOW-DIGEST12/`, holds:

- `NAME.tsv`: one row per region, with location columns, the input's metadata
  columns, and the requested feature columns;
- `NAME.columns.tsv`: one row per feature column, naming its module, feature,
  labels, and variant identity;
- `manifest.json`: the table's identity, the request that produced it, and
  the resolved config and resources of every variant involved.

Tables are views over the item store. Deleting one loses nothing; running
the same request again rebuilds it without recomputing stored items.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from txtures import __version__
from txtures.corpus import SEGMENT_LEVELS
from txtures.definitions import Definitions
from txtures.engine.execution import Executor
from txtures.engine.planning import Plan, PlannedVariant, plan_request
from txtures.errors import TxturesError
from txtures.store import MANIFEST, lock, make_shared_dir, now, read_json, write_json

if TYPE_CHECKING:
    import pandas as pd

COLUMNS_SUFFIX = ".columns.tsv"


def format_value(value: Any) -> str:
    """Format one cell for TSV output in a form R and pandas both read back exactly.

    Missing values and NaN become `NA`, infinities `Inf` and `-Inf`, Booleans
    `TRUE` and `FALSE`. Floats use the shortest text that round-trips at their
    own precision.

    Raises:
        TxturesError: A text value contains a tab or line break.
    """
    if value is None:
        return "NA"
    if isinstance(value, bool | np.bool_):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float | np.floating):
        if math.isnan(value):
            return "NA"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return str(value) if isinstance(value, np.floating) else repr(value)
    if isinstance(value, int | np.integer):
        return str(int(value))
    text = str(value)
    if any(char in text for char in "\t\n\r"):
        raise TxturesError(f"value {text!r} contains a tab or line break")
    return text


def _row(values: Iterable[Any]) -> str:
    return "\t".join(format_value(value) for value in values) + "\n"


def _variant_record(planned: PlannedVariant) -> dict[str, Any]:
    return {
        "module": planned.name,
        "config_id": planned.variant.config_id,
        "config": planned.variant.config.model_dump(mode="json"),
        "contract_version": planned.module.contract_version,
        "labels": [label.value for label in planned.variant.labels],
        "resources": {
            name: {"kind": r.kind, "identity": r.identity, "spec": dict(r.resolved)}
            for name, r in planned.resources.items()
        },
        "upstream": {name: upstream.identity for name, upstream in planned.upstream.items()},
    }


def table_manifest(plan: Plan, modules: Sequence[str] | None) -> dict[str, Any]:
    """Describe a plan's output table: its identity, request, columns, and variants."""
    return {
        "digest": plan.table_identity,
        "input": {"name": plan.corpus.name, "digest": plan.corpus.digest},
        "workflow": plan.workflow,
        "modules": list(modules) if modules is not None else None,
        "columns": [
            {
                "name": column.name,
                "module": planned.name,
                "feature": column.feature,
                "variant": planned.identity,
            }
            for column, planned in plan.column_variants()
        ],
        "variants": {planned.identity: _variant_record(planned) for planned in plan.variants},
        "notes": list(plan.notes),
        "txtures_version": __version__,
        "created": now(),
    }


def table_is_current(plan: Plan) -> bool:
    """Report whether the plan's table directory holds a complete table with its identity."""
    try:
        manifest = read_json(plan.table_directory / MANIFEST)
    except (OSError, ValueError):
        return False
    name = plan.corpus.name
    return (
        isinstance(manifest, dict)
        and manifest.get("digest") == plan.table_identity
        and (plan.table_directory / f"{name}.tsv").is_file()
        and (plan.table_directory / f"{name}{COLUMNS_SUFFIX}").is_file()
    )


def _segment_levels(corpus) -> list[str]:
    """The segment levels that some document of `corpus` defines, outermost first."""
    return [level for level in SEGMENT_LEVELS if corpus.segment_count(level)]


def table_header(plan: Plan) -> list[str]:
    """The output table's column names: location, metadata, then features.

    Each section, paragraph, or sentence ID column is present only when the
    input defines that level, and `region_id` only when the input gives one.
    """
    corpus = plan.corpus
    ids = [f"{level}_id" for level in _segment_levels(corpus)]
    if corpus.has_region_ids:
        ids.append("region_id")
    header = ["document_id", *ids, "region_index", "text", "char_start", "char_end"]
    return [*header, *corpus.metadata_columns, *(column.name for column in plan.columns)]


def table_rows(plan: Plan, executor: Executor) -> Iterator[list[Any]]:
    """Yield the output table's rows, one per region, computing results as needed.

    Results are released after each document, so memory use does not grow
    with the input.

    Raises:
        TxturesError: A module returned columns of the wrong length or omitted
            a feature.
    """
    corpus = plan.corpus
    levels = _segment_levels(corpus)
    column_variants = plan.column_variants()
    for document in corpus.documents:
        features = _document_columns(document, column_variants, executor)
        segment_ids = [document.segment_ids(level) for level in levels]
        if corpus.has_region_ids:
            segment_ids.append(list(document.region_ids))
        for index, (start, end) in enumerate(document.regions):
            metadata = document.region_metadata[index] if document.region_metadata else {}
            values: list[Any] = [document.document_id]
            values += [ids[index] if ids else None for ids in segment_ids]
            values += [index + 1, document.text[start:end], start, end]
            values += [metadata.get(name) for name in corpus.metadata_columns]
            values += [feature[index] for feature in features]
            yield values
        executor.forget()


def table_frame(plan: Plan, executor: Executor) -> pd.DataFrame:
    """Assemble the output table in memory, with the columns and values `write_table` writes."""
    import pandas as pd

    header = table_header(plan)
    frame = pd.DataFrame(list(table_rows(plan, executor)), columns=header)
    return _typed(frame, [column.name for column in plan.columns])


def write_table(plan: Plan, executor: Executor, modules: Sequence[str] | None) -> Path:
    """Assemble the plan's output table from results and publish it atomically.

    Raises:
        TxturesError: A module returned columns of the wrong length or omitted
            a feature.
    """
    corpus = plan.corpus
    directory = plan.table_directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        with (staging / f"{corpus.name}.tsv").open("w", encoding="utf-8", newline="") as handle:
            handle.write("\t".join(table_header(plan)) + "\n")
            for values in table_rows(plan, executor):
                handle.write(_row(values))
        with (staging / f"{corpus.name}{COLUMNS_SUFFIX}").open("w", encoding="utf-8") as handle:
            handle.write("column\tmodule\tfeature\tlabels\tvariant\n")
            for column, planned in plan.column_variants():
                labels = ";".join(
                    f"{label.module}.{label.source}={label.value}"
                    for label in column.variant.labels
                )
                handle.write(
                    _row([column.name, planned.name, column.feature, labels, planned.identity])
                )
        write_json(staging / MANIFEST, table_manifest(plan, modules))
        if directory.exists():
            shutil.rmtree(directory)
        os.rename(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory


# Reading tables back

INTEGER_COLUMNS = ("region_index", "char_start", "char_end")


def _typed(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Give location columns fixed types and feature columns numeric or Boolean types."""
    for name in frame.columns:
        if name in INTEGER_COLUMNS:
            frame[name] = frame[name].astype("int64")
        elif name not in features:
            frame[name] = frame[name].astype(object)
    return frame


def _parse_feature(values: pd.Series) -> pd.Series:
    """Parse a feature column written by `format_value`: Boolean, integer, float, or text."""
    import pandas as pd

    present = values[values != "NA"]
    if len(present) and present.isin(["TRUE", "FALSE"]).all():
        return values.map({"TRUE": True, "FALSE": False, "NA": pd.NA}).astype("boolean")
    try:
        if len(present) == len(values) and len(values):
            return values.map(int).astype("int64")
    except ValueError:
        pass
    try:
        return values.map(lambda value: math.nan if value == "NA" else float(value))
    except ValueError:
        return values.where(values != "NA", None)


def read_table(path: Path | str) -> pd.DataFrame:
    """Read an output table written by txtures, keeping its values exact.

    `path` is a table directory or its `NAME.tsv` file. Location and metadata
    columns are read as text, so a region such as `NA` or `null` stays a
    string; only feature columns, listed in `NAME.columns.tsv`, treat `NA` as
    missing and `Inf` as infinite.

    Raises:
        TxturesError: `path` is not an output table.
    """
    path = Path(path)
    if path.is_dir():
        tables = [p for p in path.glob("*.tsv") if not p.name.endswith(COLUMNS_SUFFIX)]
        if len(tables) != 1:
            raise TxturesError(f"{path} is not an output table directory")
        path = tables[0]
    columns_file = path.with_name(path.name.removesuffix(".tsv") + COLUMNS_SUFFIX)
    if not path.is_file() or not columns_file.is_file():
        raise TxturesError(f"{path} is not an output table: {columns_file.name} is missing")
    import pandas as pd

    options = {"sep": "\t", "dtype": str, "keep_default_na": False, "quoting": 3}
    features = list(pd.read_csv(columns_file, **options)["column"])
    frame = pd.read_csv(path, **options)
    for name in features:
        frame[name] = _parse_feature(frame[name])
    return _typed(frame, features)


def _document_columns(document, column_variants, executor: Executor) -> list[Sequence[Any]]:
    """Return each requested feature column's values for one document."""
    by_variant: dict[str, Mapping[str, Any]] = {}
    features = []
    for column, planned in column_variants:
        if planned.identity not in by_variant:
            result = executor.result(planned, document)
            by_variant[planned.identity] = planned.module.columns(
                planned.variant.config, document, result
            )
        values = by_variant[planned.identity].get(column.feature)
        if values is None:
            raise TxturesError(f"module {planned.name} did not return feature {column.feature!r}")
        if len(values) != len(document.regions):
            raise TxturesError(
                f"module {planned.name} returned {len(values)} {column.feature} values for "
                f"document {document.document_id!r}, which has {len(document.regions)} regions"
            )
        features.append(values)
    return features


# Status and cleanup


@dataclass(frozen=True)
class TableStatus:
    """Whether a table directory matches what its request would produce now."""

    path: Path
    state: str
    """`current`, `stale`, or `broken`."""
    reason: str = ""


def table_statuses(
    derivatives: Path,
    replan: Callable[[str, str, Sequence[str] | None], Plan],
    names: Sequence[str] | None = None,
) -> list[TableStatus]:
    """Check every table directory under `derivatives`, or only those of `names`.

    `replan` plans a request from an input name, workflow, and module list, as
    recorded in a table's manifest. A table is current when replanning gives
    its identity, stale when it does not or cannot be replanned, and broken
    when its directory is incomplete.
    """
    if not derivatives.is_dir():
        return []
    statuses = []
    inputs = [derivatives / name for name in names] if names else sorted(derivatives.iterdir())
    for input_directory in inputs:
        if not input_directory.is_dir() or input_directory.name.startswith("."):
            continue
        for directory in sorted(input_directory.iterdir()):
            if not directory.is_dir():
                continue
            if directory.name.startswith("."):
                statuses.append(TableStatus(directory, "broken", "interrupted write"))
                continue
            try:
                manifest = read_json(directory / MANIFEST)
                request = (manifest["input"]["name"], manifest["workflow"], manifest["modules"])
            except (OSError, ValueError, KeyError, TypeError):
                statuses.append(TableStatus(directory, "broken", "missing or invalid manifest"))
                continue
            try:
                plan = replan(*request)
            except TxturesError as error:
                reason = str(error).splitlines()[0] if str(error) else type(error).__name__
                statuses.append(TableStatus(directory, "stale", f"cannot replan: {reason}"))
                continue
            if plan.table_directory != directory or not table_is_current(plan):
                statuses.append(TableStatus(directory, "stale", "inputs or definitions changed"))
            else:
                statuses.append(TableStatus(directory, "current"))
    return statuses


def run_request(plan: Plan, executor: Executor, modules: Sequence[str] | None) -> tuple[Path, bool]:
    """Run a plan and write its table, unless an identical table already exists.

    Returns the table directory and whether anything was written.
    """
    if table_is_current(plan):
        return plan.table_directory, False
    executor.prepare_resources()
    executor.compute_items()
    directory = plan.table_directory
    make_shared_dir(directory.parent)
    # Concurrent runs of one request (or the workers of one job) write it once.
    with lock(directory.with_name(directory.name + ".lock")):
        if table_is_current(plan):
            return directory, False
        return write_table(plan, executor, modules), True


def replanner(settings, definitions: Definitions):
    """Return a function that replans a recorded request with current settings and definitions."""

    def replan(name: str, workflow: str, modules: Sequence[str] | None) -> Plan:
        return plan_request(settings, definitions, name, workflow, modules)

    return replan
