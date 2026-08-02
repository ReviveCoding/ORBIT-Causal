from __future__ import annotations

import csv
import gzip
import hashlib
import hmac
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, SplitConfig

CRITEO_REQUIRED = {
    *(f"f{i}" for i in range(12)),
    "treatment",
    "exposure",
    "visit",
    "conversion",
}
MAX_CRITEO_HEADER_BYTES = 1 << 20
CRITEO_V21_EXPECTED_ROWS = 13_979_592
CRITEO_V21_EXPECTED_GZIP_BYTES = 311_422_618
CRITEO_V21_EXPECTED_GZIP_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    feature_count: int
    missing_count: int
    duplicate_profiles: int
    invalid_binary_count: int
    invalid_numeric_count: int
    invalid_categorical_count: int
    exposure_constraint_violations: int
    conversion_constraint_violations: int
    extreme_numeric_count: int
    treatment_rate: float
    visit_rate: float
    conversion_rate: float
    leakage_check: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class CriteoFileValidationReport:
    schema_version: str
    row_count: int
    feature_count: int
    treatment_rate: float
    visit_rate: float
    conversion_rate: float
    source_size_bytes: int
    source_sha256: str
    gzip_encoded: bool
    gzip_crc_validated: bool
    full_file_validated: bool
    official_file_match: bool | None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def generate_synthetic_data(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate a randomized, nonlinear HTE fixture with observed ground truth.

    The fixture is deliberately not a proxy for financial clients. It exists to
    validate estimators, policy metrics, calibration, and failure gates.
    """

    if n_rows < 1000:
        raise ValueError("synthetic fixture requires at least 1,000 rows")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 6))
    cat_a = rng.integers(0, 8, size=n_rows)
    cat_b = rng.integers(0, 5, size=n_rows)
    treatment = rng.binomial(1, 0.50, size=n_rows)

    baseline_logit = (
        -1.35
        + 0.55 * x[:, 0]
        - 0.35 * x[:, 1]
        + 0.25 * x[:, 2] * x[:, 3]
        + 0.12 * (cat_a == 2)
        - 0.10 * (cat_b == 4)
    )
    mu0 = 1.0 / (1.0 + np.exp(-baseline_logit))
    true_tau = (
        0.035
        + 0.090 / (1.0 + np.exp(-1.8 * x[:, 0]))
        + 0.045 * np.sin(x[:, 2])
        + 0.035 * (cat_a == 5)
        - 0.060 * (x[:, 1] > 1.0)
    )
    true_tau = np.clip(true_tau, -0.08, 0.20)
    mu1 = np.clip(mu0 + true_tau, 0.005, 0.98)
    visit_probability = np.where(treatment == 1, mu1, mu0)
    visit = rng.binomial(1, visit_probability)

    compliance_probability = 1.0 / (1.0 + np.exp(-(-0.2 + 0.6 * x[:, 4])))
    exposure = treatment * rng.binomial(1, compliance_probability)
    conversion = visit * rng.binomial(1, np.clip(0.08 + 0.05 * mu0, 0.02, 0.25))

    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    frame["f6"] = cat_a
    frame["f7"] = cat_b
    frame["treatment"] = treatment.astype(np.int8)
    frame["exposure"] = exposure.astype(np.int8)
    frame["visit"] = visit.astype(np.int8)
    frame["conversion"] = conversion.astype(np.int8)
    frame["true_mu0"] = mu0
    frame["true_mu1"] = mu1
    frame["true_tau"] = true_tau
    return frame


def _criteo_header(path: Path) -> tuple[list[str], bool]:
    try:
        with path.open("rb") as raw:
            gzip_encoded = raw.read(2) == b"\x1f\x8b"
        with gzip.open(path, "rb") if gzip_encoded else path.open("rb") as handle:
            header_bytes = handle.readline(MAX_CRITEO_HEADER_BYTES + 1)
        if len(header_bytes) > MAX_CRITEO_HEADER_BYTES:
            raise ValueError("Criteo CSV header exceeds the 1 MiB safety limit")
        header = next(csv.reader([header_bytes.decode("utf-8-sig")]))
    except (
        OSError,
        EOFError,
        gzip.BadGzipFile,
        UnicodeDecodeError,
        csv.Error,
        StopIteration,
    ) as error:
        raise ValueError("Criteo input must contain a valid UTF-8 CSV header") from error
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise ValueError(f"Criteo CSV contains duplicate columns: {duplicates}")
    missing = sorted(CRITEO_REQUIRED - set(header))
    if missing:
        raise ValueError(f"not CRITEO-UPLIFTv2 schema; missing columns: {missing}")
    return header, gzip_encoded


def _sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def authenticated_criteo_snapshot(
    path: str | Path,
    *,
    expected_size_bytes: int = CRITEO_V21_EXPECTED_GZIP_BYTES,
    expected_sha256: str = CRITEO_V21_EXPECTED_GZIP_SHA256,
) -> Iterator[Path]:
    """Freeze authenticated source bytes so validation and loading cannot diverge."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Criteo source not found: {path}")
    if (
        not isinstance(expected_size_bytes, int)
        or isinstance(expected_size_bytes, bool)
        or expected_size_bytes < 1
    ):
        raise ValueError("expected_size_bytes must be a positive integer")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)
    ):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")

    snapshot: Path | None = None
    digest = hashlib.sha256()
    size = 0
    try:
        try:
            with (
                path.open("rb") as source,
                tempfile.NamedTemporaryFile(
                    prefix="orbit-criteo-snapshot-",
                    suffix=".csv.gz",
                    delete=False,
                ) as target,
            ):
                snapshot = Path(target.name)
                while chunk := source.read(1 << 20):
                    size += len(chunk)
                    if size > expected_size_bytes:
                        raise ValueError("Criteo source exceeds the pinned official v2.1 size")
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        except OSError as error:
            raise ValueError("cannot create an authenticated Criteo source snapshot") from error
        if size != expected_size_bytes or not hmac.compare_digest(
            digest.hexdigest(), expected_sha256.lower()
        ):
            raise ValueError("Criteo source does not match the pinned official v2.1 size/SHA-256")
        yield snapshot
    finally:
        if snapshot is not None:
            snapshot.unlink(missing_ok=True)


def _source_row_priorities(source_row_ids: np.ndarray) -> np.ndarray:
    """Return deterministic SplitMix64 priorities for bounded whole-file sampling."""

    values = np.asarray(source_row_ids, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _validate_sample_window(max_rows: int | None, sample_offset: int) -> None:
    if not isinstance(sample_offset, int) or isinstance(sample_offset, bool) or sample_offset < 0:
        raise ValueError("sample_offset must be a nonnegative integer")
    if max_rows is None:
        if sample_offset != 0:
            raise ValueError("sample_offset requires max_rows")
        return
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer when provided")


def deterministic_sample_row_ids(
    total_rows: int,
    max_rows: int | None,
    sample_offset: int = 0,
    *,
    chunk_rows: int = 250_000,
) -> np.ndarray:
    """Return a deterministic, disjoint SplitMix64 priority window.

    Consecutive windows with the same ``max_rows`` and offsets ``0``,
    ``max_rows``, ... are disjoint by construction.  This helper derives the
    admitted source-row contract without reading outcome or feature values.
    """

    _validate_sample_window(max_rows, sample_offset)
    if not isinstance(total_rows, int) or isinstance(total_rows, bool) or total_rows <= 0:
        raise ValueError("total_rows must be a positive integer")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if max_rows is None:
        return np.arange(total_rows, dtype=np.int64)
    target = sample_offset + max_rows
    if target > total_rows:
        raise ValueError("sample window exceeds the available source rows")

    selected_ids = np.empty(0, dtype=np.int64)
    selected_priorities = np.empty(0, dtype=np.uint64)
    for start in range(0, total_rows, chunk_rows):
        source_ids = np.arange(start, min(start + chunk_rows, total_rows), dtype=np.int64)
        priorities = _source_row_priorities(source_ids)
        selected_ids = np.concatenate((selected_ids, source_ids))
        selected_priorities = np.concatenate((selected_priorities, priorities))
        if len(selected_ids) > target:
            keep = np.argpartition(selected_priorities, target - 1)[:target]
            selected_ids = selected_ids[keep]
            selected_priorities = selected_priorities[keep]

    order = np.argsort(selected_priorities, kind="stable")
    window = selected_ids[order][sample_offset:target]
    return np.sort(window, kind="mergesort")


def sample_row_ids_sha256(source_row_ids: np.ndarray) -> str:
    values = np.asarray(source_row_ids)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("source_row_ids must be a nonempty one-dimensional vector")
    if not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
        raise ValueError("source_row_ids must contain nonnegative integers")
    canonical = np.sort(values.astype("<i8", copy=False), kind="mergesort")
    digest = hashlib.sha256()
    digest.update(len(canonical).to_bytes(8, "little", signed=False))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def load_criteo_csv(
    path: str | Path,
    max_rows: int | None = None,
    sample_offset: int = 0,
) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Criteo source not found: {path}")
    _validate_sample_window(max_rows, sample_offset)
    _, gzip_encoded = _criteo_header(path)
    read_options = {
        "compression": "gzip" if gzip_encoded else None,
        "usecols": sorted(CRITEO_REQUIRED),
    }
    try:
        if max_rows is None:
            frame = pd.read_csv(path, **read_options)
            frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
            return frame

        target = sample_offset + max_rows
        selected_frames: list[pd.DataFrame] = []
        selected_priorities: list[np.ndarray] = []
        offset = 0
        # Keep peak reader memory bounded independently of the requested sample
        # size.  A large --max-rows value must not silently become a single
        # multi-million-row CSV chunk.
        reader = pd.read_csv(path, chunksize=250_000, **read_options)
        for chunk in reader:
            source_ids = np.arange(offset, offset + len(chunk), dtype=np.int64)
            offset += len(chunk)
            priorities = _source_row_priorities(source_ids)
            chunk.insert(0, "source_row_id", source_ids)
            if len(chunk) > target:
                keep = np.argpartition(priorities, target - 1)[:target]
                chunk = chunk.iloc[keep]
                priorities = priorities[keep]
            selected_frames.append(chunk)
            selected_priorities.append(priorities)
            if sum(len(part) for part in selected_frames) > target:
                combined = pd.concat(selected_frames, ignore_index=True)
                combined_priorities = np.concatenate(selected_priorities)
                keep = np.argpartition(combined_priorities, target - 1)[:target]
                selected_frames = [combined.iloc[keep]]
                selected_priorities = [combined_priorities[keep]]
        if not selected_frames:
            raise ValueError("Criteo input contains no data rows")
        if offset < target:
            raise ValueError("sample window exceeds the available source rows")
        combined = pd.concat(selected_frames, ignore_index=True)
        combined_priorities = np.concatenate(selected_priorities)
        order = np.argsort(combined_priorities, kind="stable")
        sampled = combined.iloc[order[sample_offset:target]]
        return sampled.sort_values("source_row_id", kind="mergesort").reset_index(drop=True)
    except (OSError, EOFError, gzip.BadGzipFile, pd.errors.ParserError) as error:
        raise ValueError("Criteo input is truncated, corrupt, or malformed") from error


def validate_criteo_file(
    path: str | Path,
    config: DataConfig,
    *,
    max_rows: int | None = None,
    chunk_rows: int = 250_000,
    require_official_file: bool = False,
    expected_rows: int = CRITEO_V21_EXPECTED_ROWS,
    expected_size_bytes: int = CRITEO_V21_EXPECTED_GZIP_BYTES,
    expected_sha256: str = CRITEO_V21_EXPECTED_GZIP_SHA256,
) -> CriteoFileValidationReport:
    """Validate a Criteo file, freezing authenticated official bytes first.

    Public callers must never parse one inode and later authenticate another.
    Official validation therefore copies and authenticates the source once, then
    performs every schema, CRC, row-count, and statistic check on that snapshot.
    """

    if require_official_file:
        with authenticated_criteo_snapshot(
            path,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        ) as snapshot:
            return _validate_criteo_contents(
                snapshot,
                config,
                max_rows=max_rows,
                chunk_rows=chunk_rows,
                require_official_file=True,
                expected_rows=expected_rows,
                expected_size_bytes=expected_size_bytes,
                expected_sha256=expected_sha256,
            )
    return _validate_criteo_contents(
        path,
        config,
        max_rows=max_rows,
        chunk_rows=chunk_rows,
        require_official_file=False,
        expected_rows=expected_rows,
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
    )


def _validate_criteo_contents(
    path: str | Path,
    config: DataConfig,
    *,
    max_rows: int | None = None,
    chunk_rows: int = 250_000,
    require_official_file: bool = False,
    expected_rows: int = CRITEO_V21_EXPECTED_ROWS,
    expected_size_bytes: int = CRITEO_V21_EXPECTED_GZIP_BYTES,
    expected_sha256: str = CRITEO_V21_EXPECTED_GZIP_SHA256,
) -> CriteoFileValidationReport:
    """Stream one stable Criteo path without retaining rows in memory."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Criteo source not found: {path}")
    if config.kind != "criteo_v2":
        raise ValueError("streaming validation requires data.kind='criteo_v2'")
    if max_rows is not None and (
        not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0
    ):
        raise ValueError("max_rows must be a positive integer when provided")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if require_official_file and max_rows is not None:
        raise ValueError("official-file validation requires the complete file")
    header, gzip_encoded = _criteo_header(path)
    if require_official_file and set(header) != CRITEO_REQUIRED:
        extra = sorted(set(header) - CRITEO_REQUIRED)
        raise ValueError(f"official Criteo file contains unexpected columns: {extra}")

    row_count = 0
    treatment_sum = 0
    visit_sum = 0
    conversion_sum = 0
    try:
        reader = pd.read_csv(
            path,
            compression="gzip" if gzip_encoded else None,
            nrows=max_rows,
            usecols=sorted(CRITEO_REQUIRED),
            chunksize=chunk_rows,
        )
        for chunk in reader:
            validate_frame(chunk, config)
            row_count += len(chunk)
            treatment_sum += int(chunk[config.treatment].sum())
            visit_sum += int(chunk["visit"].sum())
            conversion_sum += int(chunk["conversion"].sum())
    except (OSError, EOFError, gzip.BadGzipFile, pd.errors.ParserError) as error:
        raise ValueError("Criteo input is truncated, corrupt, or malformed") from error
    if row_count == 0:
        raise ValueError("Criteo input contains no data rows")

    source_size = path.stat().st_size
    source_sha256 = _sha256_path(path)
    official_match: bool | None = None
    if require_official_file:
        official_match = bool(
            gzip_encoded
            and row_count == expected_rows
            and source_size == expected_size_bytes
            and source_sha256 == expected_sha256
        )
        if not official_match:
            raise ValueError(
                "Criteo file does not match the pinned official v2.1 row-count/size/SHA-256 contract"
            )
    return CriteoFileValidationReport(
        schema_version="1.0",
        row_count=row_count,
        feature_count=len(config.feature_columns),
        treatment_rate=treatment_sum / row_count,
        visit_rate=visit_sum / row_count,
        conversion_rate=conversion_sum / row_count,
        source_size_bytes=source_size,
        source_sha256=source_sha256,
        gzip_encoded=gzip_encoded,
        gzip_crc_validated=bool(gzip_encoded and max_rows is None),
        full_file_validated=max_rows is None,
        official_file_match=official_match,
    )


def validate_frame(frame: pd.DataFrame, config: DataConfig) -> DataQualityReport:
    _validate_data_contract(config)
    if frame.empty:
        raise ValueError("data admission failed: frame must contain at least one row")
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"data admission failed: duplicate columns: {duplicate_columns}")
    prohibited = {config.treatment, config.exposure, "visit", "conversion", "split"}
    leaked = sorted(
        column
        for column in config.feature_columns
        if column in prohibited or column.startswith("true_")
    )
    if leaked:
        raise ValueError(f"data admission failed: leaked feature columns: {leaked}")
    required = set(config.feature_columns) | {
        config.treatment,
        config.exposure,
        "visit",
        "conversion",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    binary_columns = [config.treatment, config.exposure, "visit", "conversion"]
    invalid_binary = 0
    for column in binary_columns:
        valid = frame[column].isin([0, 1])
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if np.iscomplexobj(numeric.to_numpy()):
            valid &= False
        invalid_binary += int((~valid).sum())
    invalid_numeric = 0
    invalid_categorical = 0
    for column in config.feature_columns:
        if column in config.categorical_columns:
            for value in frame[column]:
                try:
                    _canonical_scalar(value)
                except (TypeError, ValueError):
                    invalid_categorical += 1
            continue
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_numeric += int((original.notna() & numeric.isna()).sum())
        numeric_values = numeric.to_numpy()
        if np.iscomplexobj(numeric_values):
            invalid_numeric += int(original.notna().sum())
        else:
            invalid_numeric += int(np.isinf(numeric_values.astype(float)).sum())
    if invalid_binary:
        exposure_violations = 0
        conversion_violations = 0
    else:
        exposure_violations = int((frame[config.exposure] > frame[config.treatment]).sum())
        conversion_violations = int((frame["conversion"] > frame["visit"]).sum())
    if (
        invalid_binary
        or invalid_numeric
        or invalid_categorical
        or exposure_violations
        or conversion_violations
    ):
        raise ValueError(
            "data admission failed: "
            f"invalid_binary={invalid_binary}, invalid_numeric={invalid_numeric}, "
            f"invalid_categorical={invalid_categorical}, "
            f"exposure_violations={exposure_violations}, "
            f"conversion_violations={conversion_violations}"
        )

    profiles = profile_ids(frame, config)
    duplicate_profiles = int(profiles.duplicated().sum())
    extreme_numeric_count = 0
    for column in config.feature_columns:
        if column in config.categorical_columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        q25, q75 = numeric.quantile([0.25, 0.75])
        spread = q75 - q25
        if np.isfinite(spread) and spread > 0:
            extreme_numeric_count += int(
                ((numeric < q25 - 10 * spread) | (numeric > q75 + 10 * spread)).sum()
            )
    return DataQualityReport(
        row_count=len(frame),
        feature_count=len(config.feature_columns),
        missing_count=int(frame.loc[:, config.feature_columns].isna().sum().sum()),
        duplicate_profiles=duplicate_profiles,
        invalid_binary_count=invalid_binary,
        invalid_numeric_count=invalid_numeric,
        invalid_categorical_count=invalid_categorical,
        exposure_constraint_violations=exposure_violations,
        conversion_constraint_violations=conversion_violations,
        extreme_numeric_count=extreme_numeric_count,
        treatment_rate=float(frame[config.treatment].mean()),
        visit_rate=float(frame["visit"].mean()),
        conversion_rate=float(frame["conversion"].mean()),
        leakage_check="PASS: exposure and outcomes excluded from feature contract",
    )


def _canonical_scalar(value: object) -> list[object]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return ["missing"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["number", str(value)]
    if isinstance(value, float):
        if np.isnan(value):
            return ["missing"]
        if not np.isfinite(value):
            raise ValueError("profile values must be finite or missing")
        if value == 0 or value.is_integer():
            return ["number", str(int(value))]
        return ["number", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    raise TypeError(f"unsupported profile scalar type: {type(value).__name__}")


def _profile_hash(values: Iterable[object]) -> str:
    canonical = json.dumps(
        [_canonical_scalar(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


def profile_ids(frame: pd.DataFrame, config: DataConfig) -> pd.Series:
    """Return vectorized profile IDs and fail closed on a hash collision.

    The vectorized primary hash keeps the 14M-row public path practical. Any
    duplicated 64-bit identifier is rechecked with the canonical 128-bit
    serializer so distinct profiles are never silently merged.
    """

    _validate_data_contract(config)
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"duplicate profile columns: {duplicate_columns}")
    missing = sorted(set(config.feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing profile features: {missing}")
    features = frame.loc[:, config.feature_columns]
    normalized = pd.DataFrame(index=features.index)
    for column in config.feature_columns:
        if column in config.categorical_columns:
            normalized[column] = features[column].map(_categorical_token)
        else:
            normalized[column] = pd.to_numeric(features[column], errors="raise").astype(float)
    identifiers = pd.util.hash_pandas_object(normalized, index=False).rename("profile_id")
    possible_collisions = identifiers.duplicated(keep=False)
    if possible_collisions.any():
        # Recheck the original typed values, not their float model projection.
        # Otherwise integers above 2**53 can round to the same float and evade
        # the canonical collision audit even though they are distinct profiles.
        audit = features.loc[possible_collisions].copy()
        audit.insert(0, "profile_id", identifiers.loc[possible_collisions].to_numpy())
        audit = audit.drop_duplicates()
        audit["canonical"] = audit.loc[:, config.feature_columns].apply(
            lambda row: _profile_hash(row.values), axis=1
        )
        if (audit.groupby("profile_id", sort=False)["canonical"].nunique() > 1).any():
            raise RuntimeError("pretreatment profile hash collision detected")
    return identifiers


def _categorical_token(value: object) -> str:
    """Type-safe token that preserves the frozen numeric-category encoding."""

    if isinstance(value, np.generic):
        value = value.item()
    typed = _canonical_scalar(value)
    if typed[0] == "missing":
        return "<NA>"
    if isinstance(value, str):
        return f"<STRING>{value}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    raise TypeError(f"unsupported categorical scalar type: {type(value).__name__}")


def _validate_data_contract(config: DataConfig) -> None:
    if not config.feature_columns or any(
        not isinstance(column, str) or not column for column in config.feature_columns
    ):
        raise ValueError("data feature_columns must contain nonempty strings")
    if len(set(config.feature_columns)) != len(config.feature_columns):
        raise ValueError("data feature_columns must be unique")
    if len(set(config.categorical_columns)) != len(config.categorical_columns):
        raise ValueError("data categorical_columns must be unique")
    if not set(config.categorical_columns).issubset(config.feature_columns):
        raise ValueError("data categorical_columns must be a subset of feature_columns")


def assign_splits(
    frame: pd.DataFrame,
    data_config: DataConfig,
    split_config: SplitConfig,
    seed: int,
) -> pd.Series:
    """Deterministically allocate identical pretreatment profiles together.

    Groups are ordered by a salted hash, then allocated within treatment/outcome
    strata. The target fractions are approximate when duplicate groups are large.
    """

    split_config.validate()
    _validate_data_contract(data_config)
    profile_hashes = profile_ids(frame, data_config)
    group_frame = pd.DataFrame(
        {
            "profile_hash": profile_hashes,
            "treatment": frame[data_config.treatment].to_numpy(),
            "outcome": frame[data_config.outcome].to_numpy(),
        }
    )
    group_summary = group_frame.groupby("profile_hash", sort=False).agg(
        treatment_rate=("treatment", "mean"),
        outcome_rate=("outcome", "mean"),
        row_count=("treatment", "size"),
    )
    group_summary = group_summary.reset_index()
    group_summary["treatment"] = (group_summary.pop("treatment_rate") >= 0.5).astype(int)
    group_summary["outcome"] = (group_summary.pop("outcome_rate") >= 0.5).astype(int)
    salted = pd.DataFrame(
        {
            "seed": np.full(len(group_summary), seed, dtype=np.uint32),
            "profile_hash": group_summary["profile_hash"].to_numpy(dtype=np.uint64),
        }
    )
    group_summary["order_hash"] = pd.util.hash_pandas_object(salted, index=False).to_numpy()
    group_summary = group_summary.sort_values(
        ["treatment", "outcome", "order_hash", "profile_hash"],
        kind="mergesort",
    )
    stratum = group_summary.groupby(["treatment", "outcome"], sort=False)["row_count"]
    midpoint = stratum.cumsum() - 0.5 * group_summary["row_count"]
    fraction = midpoint / stratum.transform("sum")
    names = np.array(["train", "calibration", "policy", "final_test"], dtype=object)
    shares = np.array(
        [split_config.train, split_config.calibration, split_config.policy, split_config.final_test]
    )
    boundaries = np.cumsum(shares)
    split_index = np.searchsorted(boundaries, fraction.to_numpy(), side="right")
    split_index = np.minimum(split_index, len(names) - 1)
    labels = pd.Series(
        names[split_index],
        index=group_summary["profile_hash"].to_numpy(),
    )
    result = profile_hashes.map(labels)
    if result.isna().any():
        raise RuntimeError("split assignment produced missing labels")
    return result.rename("split")


def dataset_fingerprint(frame: pd.DataFrame, config: DataConfig) -> str:
    digest = hashlib.sha256()
    candidate_columns = [
        *(column for column in ("source_row_id",) if column in frame.columns),
        *config.feature_columns,
        config.treatment,
        config.exposure,
        "visit",
        "conversion",
        *sorted(column for column in frame.columns if column.startswith("true_")),
    ]
    columns = list(dict.fromkeys(candidate_columns))
    digest.update(str((len(frame), len(columns))).encode("utf-8"))
    digest.update("|".join(f"{column}:{frame[column].dtype}" for column in columns).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(frame.loc[:, columns], index=False)
    digest.update(hashed.to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def split_summary(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    rows = []
    for split_name, part in frame.groupby("split", sort=False):
        rows.append(
            {
                "split": split_name,
                "n": len(part),
                "share": len(part) / len(frame),
                "treatment_rate": part[config.treatment].mean(),
                "outcome_rate": part[config.outcome].mean(),
                "missing_features": int(part.loc[:, config.feature_columns].isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def subgroup_summary(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Return technical categorical slices; this is not a fairness audit."""

    rows: list[dict[str, object]] = []
    for column in config.categorical_columns:
        values = frame[column].map(_categorical_token)
        for value, count in values.value_counts(dropna=False).items():
            mask = values == value
            rows.append(
                {
                    "feature": column,
                    "value": str(value),
                    "n": int(count),
                    "share": float(count / len(frame)),
                    "treatment_rate": float(frame.loc[mask, config.treatment].mean()),
                    "outcome_rate": float(frame.loc[mask, config.outcome].mean()),
                    "interpretation": "technical subgroup; not a protected-class fairness label",
                }
            )
    columns = [
        "feature",
        "value",
        "n",
        "share",
        "treatment_rate",
        "outcome_rate",
        "interpretation",
    ]
    return pd.DataFrame(rows, columns=columns)
